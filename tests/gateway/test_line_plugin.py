"""Tests for the LINE platform adapter plugin.

Covers the seven synthesis areas from the PR review:

1. webhook signature verification (HMAC-SHA256, base64) + tampering rejection
2. inbound chat-id resolution for user / group / room sources
3. three-allowlist gating (users / groups / rooms / allow_all)
4. inbound dedup via webhookEventId
5. RequestCache state machine (PENDING → READY → DELIVERED, ERROR)
6. Markdown stripping with URL preservation + LINE-sized chunking
7. send routing: reply token preferred → push fallback → batched at 5/call
8. register() metadata + standalone_send shape
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/line/adapter.py under plugin_adapter_line so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_line = load_plugin_adapter("line")

verify_line_signature = _line.verify_line_signature
strip_markdown_preserving_urls = _line.strip_markdown_preserving_urls
split_for_line = _line.split_for_line
build_postback_button_message = _line.build_postback_button_message
build_workbench_handoff_message = _line.build_workbench_handoff_message
build_workbench_ticket_message = _line.build_workbench_ticket_message
needs_workbench_output = _line.needs_workbench_output
_resolve_chat = _line._resolve_chat
_allowed_for_source = _line._allowed_for_source
_is_system_bypass = _line._is_system_bypass
RequestCache = _line.RequestCache
State = _line.State
LineAdapter = _line.LineAdapter
register = _line.register
check_requirements = _line.check_requirements
validate_config = _line.validate_config
_standalone_send = _line._standalone_send
_env_enablement = _line._env_enablement
_MessageDeduplicator = _line._MessageDeduplicator
WORKBENCH_TICKET_RETURN_PREFIX = _line.WORKBENCH_TICKET_RETURN_PREFIX
LineWorkbenchTicketTurn = _line.LineWorkbenchTicketTurn
workbench_create_ticket = _line.workbench_create_ticket
bind_line_ticket_turn_for_gateway = _line.bind_line_ticket_turn_for_gateway
block_line_kanban_ticket_write = _line.block_line_kanban_ticket_write
peek_line_ticket_card = _line.peek_line_ticket_card
record_line_ticket_card = _line.record_line_ticket_card


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def _sign(self, body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def test_valid_signature_passes(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert verify_line_signature(body, sig, "secret")

    def test_tampered_body_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body + b" ", sig, "secret")

    def test_wrong_secret_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body, sig, "different")

    def test_empty_signature_rejected(self):
        assert not verify_line_signature(b"x", "", "secret")

    def test_empty_secret_rejected(self):
        assert not verify_line_signature(b"x", "AAAA", "")

    def test_garbage_signature_rejected(self):
        assert not verify_line_signature(b"hello", "not base64 at all!!", "s")


# ---------------------------------------------------------------------------
# 2. Chat-id / source resolution
# ---------------------------------------------------------------------------

class TestSourceResolution:

    def test_user_source(self):
        chat_id, ctype = _resolve_chat({"type": "user", "userId": "U123"})
        assert chat_id == "U123"
        assert ctype == "dm"

    def test_group_source(self):
        chat_id, ctype = _resolve_chat({"type": "group", "groupId": "C456", "userId": "U123"})
        assert chat_id == "C456"
        assert ctype == "group"

    def test_room_source(self):
        chat_id, ctype = _resolve_chat({"type": "room", "roomId": "R789", "userId": "U123"})
        assert chat_id == "R789"
        assert ctype == "room"

    def test_unknown_source_falls_back_to_dm(self):
        chat_id, ctype = _resolve_chat({"type": "weird"})
        assert chat_id == ""
        assert ctype == "dm"

    def test_empty_source(self):
        chat_id, ctype = _resolve_chat({})
        assert chat_id == ""
        assert ctype == "dm"


# ---------------------------------------------------------------------------
# 3. Three-allowlist gating
# ---------------------------------------------------------------------------

class TestAllowlist:

    def test_allow_all_short_circuits(self):
        for src in [
            {"type": "user", "userId": "Ufoo"},
            {"type": "group", "groupId": "Cfoo"},
            {"type": "room", "roomId": "Rfoo"},
        ]:
            assert _allowed_for_source(src, allow_all=True, user_ids=set(), group_ids=set(), room_ids=set())

    def test_user_in_allowlist_passes(self):
        src = {"type": "user", "userId": "Uok"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())

    def test_user_not_in_allowlist_rejected(self):
        src = {"type": "user", "userId": "Uother"}
        assert not _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())

    def test_group_uses_group_list_not_user_list(self):
        src = {"type": "group", "groupId": "Cok", "userId": "Uany"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uany"}, group_ids={"Cok"}, room_ids=set())
        assert not _allowed_for_source(src, allow_all=False, user_ids={"Uany"}, group_ids=set(), room_ids=set())

    def test_room_uses_room_list(self):
        src = {"type": "room", "roomId": "Rok"}
        assert _allowed_for_source(src, allow_all=False, user_ids=set(), group_ids=set(), room_ids={"Rok"})
        assert not _allowed_for_source(src, allow_all=False, user_ids=set(), group_ids=set(), room_ids=set())

    def test_unknown_type_rejected(self):
        src = {"type": "weird"}
        assert not _allowed_for_source(src, allow_all=False, user_ids=set(), group_ids=set(), room_ids=set())


# ---------------------------------------------------------------------------
# 4. Inbound dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_first_event_not_duplicate(self):
        d = _MessageDeduplicator()
        assert not d.is_duplicate("evt1")

    def test_repeat_event_marked_duplicate(self):
        d = _MessageDeduplicator()
        d.is_duplicate("evt1")
        assert d.is_duplicate("evt1")

    def test_blank_id_not_treated_as_duplicate(self):
        d = _MessageDeduplicator()
        # Blank IDs should always pass through (don't lock out unidentifiable events).
        assert not d.is_duplicate("")
        assert not d.is_duplicate("")

    def test_lru_eviction_under_pressure(self):
        d = _MessageDeduplicator(max_size=10)
        for i in range(20):
            d.is_duplicate(f"evt{i}")
        # Exact eviction order isn't specified, but the cap must be enforced.
        # Insert one more and assert the bookkeeping doesn't grow without bound.
        d.is_duplicate("evt20")
        assert len(d._seen) <= 20  # bounded — exact cap depends on eviction policy


# ---------------------------------------------------------------------------
# 5. RequestCache state machine
# ---------------------------------------------------------------------------

class TestRequestCache:

    def test_register_pending_is_pending(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        assert c.get(rid).state is State.PENDING
        assert c.get(rid).chat_id == "Uchat"

    def test_set_ready_transitions(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "the answer")
        assert c.get(rid).state is State.READY
        assert c.get(rid).payload == "the answer"

    def test_set_error_transitions(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_error(rid, "boom")
        assert c.get(rid).state is State.ERROR
        assert c.get(rid).payload == "boom"

    def test_mark_delivered_from_ready(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED

    def test_mark_delivered_from_error(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_error(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED

    def test_set_ready_on_delivered_is_noop(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "first")
        c.mark_delivered(rid)
        c.set_ready(rid, "second")
        # DELIVERED is terminal — no further mutation
        assert c.get(rid).payload == "first"
        assert c.get(rid).state is State.DELIVERED

    def test_find_pending_for_chat(self):
        c = RequestCache()
        rid_a = c.register_pending("Ua")
        rid_b = c.register_pending("Ub")
        assert c.find_pending_for_chat("Ua") == rid_a
        assert c.find_pending_for_chat("Ub") == rid_b
        assert c.find_pending_for_chat("Uc") is None
        c.set_ready(rid_a, "x")
        # No longer PENDING — should not be found
        assert c.find_pending_for_chat("Ua") is None


# ---------------------------------------------------------------------------
# 6. Markdown stripping + chunking
# ---------------------------------------------------------------------------

class TestMarkdownAndChunking:

    def test_bold_stripped(self):
        assert strip_markdown_preserving_urls("**hello**") == "hello"

    def test_italic_stripped(self):
        assert strip_markdown_preserving_urls("*hello*") == "hello"

    def test_inline_code_unfenced(self):
        assert strip_markdown_preserving_urls("run `ls -la`") == "run ls -la"

    def test_link_preserved_with_url(self):
        out = strip_markdown_preserving_urls("see [here](https://x.com)")
        assert "https://x.com" in out
        assert "here (https://x.com)" in out

    def test_heading_prefix_stripped(self):
        out = strip_markdown_preserving_urls("# Title\n## Sub")
        assert out == "Title\nSub"

    def test_bullet_marker_replaced(self):
        out = strip_markdown_preserving_urls("- a\n- b")
        assert out == "• a\n• b"

    def test_code_fence_content_kept(self):
        # Source files often contain code snippets — the agent should still
        # see the content as plain text, just without backticks.
        md = "```python\nprint('hi')\n```"
        out = strip_markdown_preserving_urls(md)
        assert "print('hi')" in out
        assert "```" not in out

    def test_split_short_returns_single_chunk(self):
        assert split_for_line("hi") == ["hi"]

    def test_split_long_chunks_at_paragraph_boundary(self):
        text = "para1\n\npara2\n\npara3"
        chunks = split_for_line(text, max_chars=8)
        assert all(len(c) <= 8 for c in chunks), chunks
        assert len(chunks) >= 2

    def test_split_caps_at_five_chunks(self):
        # 1000 paragraphs of 100 chars each — must cap at 5 LINE bubbles.
        text = "\n\n".join(["x" * 100 for _ in range(1000)])
        chunks = split_for_line(text)
        assert len(chunks) <= 5

    def test_workbench_output_uses_only_strong_presentation_signals(self):
        assert not needs_workbench_output("查詢完成：今天是 32 元。")
        assert not needs_workbench_output("我用了工具，但答案很短。")
        assert needs_workbench_output("| 方案 | 成本 |\n| --- | --- |\n| A | 100 |")
        assert needs_workbench_output("```python\nprint('hello')\n```")
        assert needs_workbench_output("長內容" * 1000)


# ---------------------------------------------------------------------------
# 7. Send routing (reply -> push fallback, batching, system-bypass)
# ---------------------------------------------------------------------------

class TestSendRouting:

    @pytest.fixture
    def adapter(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
            "sent_message_ids_path": str(tmp_path / "sent-message-ids.json"),
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad

    def test_system_bypass_recognized(self):
        assert _is_system_bypass("⚡ Interrupting current run")
        assert _is_system_bypass("⏳ Queued — agent is busy")
        assert _is_system_bypass("⏩ Steered toward new task")
        assert not _is_system_bypass("Hello world")
        assert not _is_system_bypass("")

    def test_send_uses_reply_when_token_present(self, adapter):
        import time as _time
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert result.success
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_not_called()
        # Token consumed (single-use)
        assert "Uchat" not in adapter._reply_tokens

    def test_send_records_line_message_id_for_future_quote(self, adapter):
        import time as _time
        adapter._reply_tokens["Cgroup"] = ("rt-token", _time.time() + 30)
        adapter._client.reply.return_value = ["bot-message-1"]

        result = asyncio.run(adapter.send("Cgroup", "hello"))

        assert result.success
        assert adapter._has_required_mention(
            "follow up", {"quotedMessageId": "bot-message-1"}
        )

    def test_group_response_quotes_the_triggering_user_message(self, adapter):
        import time as _time
        adapter._reply_tokens["Cgroup"] = (
            "rt-token",
            _time.time() + 30,
            "user-message-quote-token",
        )

        result = asyncio.run(adapter.send("Cgroup", "hello"))

        assert result.success
        messages = adapter._client.reply.call_args.args[1]
        assert messages[0]["quoteToken"] == "user-message-quote-token"

    def test_message_scoped_reply_token_survives_other_group_chatter(self, adapter):
        import time as _time
        adapter._reply_tokens["Cgroup"] = (
            "newer-unrelated-token",
            _time.time() + 30,
            "newer-quote-token",
        )
        adapter._reply_contexts["trigger-message"] = (
            "Cgroup",
            "original-trigger-token",
            _time.time() + 30,
            "original-quote-token",
        )

        result = asyncio.run(
            adapter.send("Cgroup", "answer", reply_to="trigger-message")
        )

        assert result.success
        token, messages = adapter._client.reply.call_args.args
        assert token == "original-trigger-token"
        assert messages[0]["quoteToken"] == "original-quote-token"

    def test_unmentioned_group_text_is_observed_without_dispatch(self, adapter):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        session_entry = MagicMock(session_id="shared-group-session")
        adapter._session_store = MagicMock()
        adapter._session_store.get_or_create_session.return_value = session_entry
        adapter.handle_message = AsyncMock()
        event = {
            "replyToken": "ignored-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ualice"},
            "message": {
                "id": "message-1",
                "type": "text",
                "quoteToken": "ignored-quote-token",
                "text": "Alice 的普通群組發言",
            },
        }

        asyncio.run(adapter._handle_message_event(event))

        adapter.handle_message.assert_not_awaited()
        shared_source = adapter._session_store.get_or_create_session.call_args.args[0]
        assert shared_source.chat_id == "Cgroup"
        assert shared_source.user_id is None
        observed = adapter._session_store.append_to_transcript.call_args.args[1]
        assert observed["observed"] is True
        assert "Ualice" in observed["content"]
        assert "Alice 的普通群組發言" in observed["content"]
        assert "Cgroup" not in adapter._reply_tokens

    def test_unmentioned_bookkeeping_command_is_dispatched(self, adapter):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()

        asyncio.run(adapter._handle_message_event({
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ualice"},
            "message": {"id": "message-1", "type": "text", "text": "記帳 午餐 125"},
        }))

        assert adapter.handle_message.await_args.args[0].text.endswith("記帳 午餐 125")

    def test_bare_unmentioned_bookkeeping_word_stays_observed(self, adapter):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter._session_store = MagicMock()
        adapter._session_store.get_or_create_session.return_value = MagicMock(
            session_id="shared-group-session"
        )
        adapter.handle_message = AsyncMock()

        asyncio.run(adapter._handle_message_event({
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ualice"},
            "message": {"id": "message-1", "type": "text", "text": "記帳"},
        }))

        adapter.handle_message.assert_not_awaited()
        adapter._session_store.append_to_transcript.assert_called_once()

    def test_bare_bookkeeping_reply_is_dispatched_without_mention(self, adapter):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()

        asyncio.run(adapter._handle_message_event({
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ualice"},
            "message": {
                "id": "message-1",
                "type": "text",
                "quotedMessageId": "quoted-message",
                "text": "記帳",
            },
        }))

        assert adapter.handle_message.await_args.args[0].reply_to_message_id == "quoted-message"

    def test_addressed_group_turn_uses_shared_observed_context_session(self, adapter):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()
        mention_text = "@Methu"
        event = {
            "replyToken": "trigger-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ubob"},
            "message": {
                "id": "trigger-message",
                "type": "text",
                "quoteToken": "trigger-quote-token",
                "quotedMessageId": "bot-message-1",
                "text": f"{mention_text} 幫我整理大家剛才說的內容",
                "mention": {
                    "mentionees": [{
                        "index": 0,
                        "length": len(mention_text),
                        "isSelf": True,
                    }],
                },
            },
        }
        adapter._remember_sent_message_ids(["bot-message-1"])

        asyncio.run(adapter._handle_message_event(event))

        forwarded = adapter.handle_message.await_args.args[0]
        assert forwarded.source.chat_id == "Cgroup"
        assert forwarded.source.user_id is None
        assert forwarded.source.role_authorized is True
        assert forwarded.text.startswith(
            "[Trusted LINE source: sender_id=Ubob; scope_id=Cgroup; "
            "source_event_id=line:message:trigger-message]\n"
        )
        assert "observed LINE group context" in forwarded.channel_prompt
        assert "first Trusted LINE source line" in forwarded.channel_prompt
        assert forwarded.reply_to_message_id == "bot-message-1"
        assert forwarded.reply_to_is_own_message is True
        assert adapter._reply_contexts["trigger-message"][1] == "trigger-reply-token"

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        assert runner._is_user_authorized(forwarded.source) is True
        anonymous_source = adapter.build_source(
            chat_id="Cgroup",
            chat_type="group",
            user_id=None,
        )
        assert runner._is_user_authorized(anonymous_source) is False

    def test_group_reply_with_mention_includes_observed_message_text(self, adapter):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()
        adapter._session_store = MagicMock()
        adapter._session_store.get_or_create_session.return_value = MagicMock(
            session_id="shared-group-session"
        )
        adapter._session_store.load_transcript.return_value = [{
            "role": "user",
            "content": "[Ualice|Ualice]\n午餐 $215",
            "message_id": "quoted-message",
            "observed": True,
        }]
        mention_text = "@Methu"
        event = {
            "replyToken": "trigger-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ubob"},
            "message": {
                "id": "trigger-message",
                "type": "text",
                "quotedMessageId": "quoted-message",
                "text": mention_text,
                "mention": {"mentionees": [{
                    "index": 0,
                    "length": len(mention_text),
                    "isSelf": True,
                }]},
            },
        }

        asyncio.run(adapter._handle_message_event(event))

        forwarded = adapter.handle_message.await_args.args[0]
        assert forwarded.reply_to_message_id == "quoted-message"
        assert forwarded.reply_to_text == (
            "[Trusted quoted LINE source: sender_id=Ualice; "
            "source_event_id=line:message:quoted-message]\n午餐 $215"
        )

    def test_group_reply_with_mention_attaches_observed_image(self, adapter, tmp_path):
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()
        adapter._session_store = MagicMock()
        adapter._session_store.get_or_create_session.return_value = MagicMock(
            session_id="shared-group-session"
        )
        adapter._session_store.load_transcript.return_value = [{
            "role": "user",
            "content": "[Ualice|Ualice]\n[image]",
            "message_id": "quoted-image",
            "observed": True,
        }]
        image_path = tmp_path / "receipt.jpg"
        image_path.write_bytes(b"image")
        adapter.workbench_media_log = str(tmp_path / "workbench-media.jsonl")
        adapter._record_workbench_media(
            chat_id="Cgroup",
            chat_type="group",
            user_id="Ualice",
            message_id="quoted-image",
            msg_type="image",
            local_path=str(image_path),
        )
        mention_text = "@Methu"

        asyncio.run(adapter._handle_message_event({
            "replyToken": "trigger-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ubob"},
            "message": {
                "id": "trigger-message",
                "type": "text",
                "quotedMessageId": "quoted-image",
                "text": mention_text,
                "mention": {"mentionees": [{
                    "index": 0,
                    "length": len(mention_text),
                    "isSelf": True,
                }]},
            },
        }))

        forwarded = adapter.handle_message.await_args.args[0]
        assert forwarded.media_urls == [str(image_path)]
        assert forwarded.media_types == ["image/jpeg"]

        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner
        from gateway.session import build_session_key

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(group_sessions_per_user=False)
        runner.adapters = {}
        runner._model = "test-model"
        runner._base_url = None
        runner._decide_image_input_mode = lambda: "native"
        asyncio.run(runner._prepare_inbound_message_text(
            event=forwarded,
            source=forwarded.source,
            history=[],
        ))
        assert runner._consume_pending_native_image_paths(
            build_session_key(forwarded.source, group_sessions_per_user=False)
        ) == [str(image_path)]

    def test_group_reply_resolves_observed_message_after_session_reset(self, adapter, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        source = adapter._group_observe_source("Cgroup", "group")
        old_session = store.get_or_create_session(source)
        store.append_to_transcript(old_session.session_id, {
            "role": "user",
            "content": "[Ualice|Ualice]\n午餐 $215",
            "message_id": "quoted-message",
            "observed": True,
        })
        store.reset_session(old_session.session_key)

        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()
        adapter._session_store = store
        mention_text = "@Methu"
        asyncio.run(adapter._handle_message_event({
            "replyToken": "trigger-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ubob"},
            "message": {
                "id": "trigger-message",
                "type": "text",
                "quotedMessageId": "quoted-message",
                "text": mention_text,
                "mention": {"mentionees": [{
                    "index": 0,
                    "length": len(mention_text),
                    "isSelf": True,
                }]},
            },
        }))

        forwarded = adapter.handle_message.await_args.args[0]
        assert forwarded.reply_to_text == (
            "[Trusted quoted LINE source: sender_id=Ualice; "
            "source_event_id=line:message:quoted-message]\n午餐 $215"
        )

    def test_group_reply_resolves_legacy_addressed_message_after_session_reset(
        self, adapter, tmp_path
    ):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        source = adapter._group_observe_source("Cgroup", "group")
        old_session = store.get_or_create_session(source)
        store.append_to_transcript(old_session.session_id, {
            "role": "user",
            "content": (
                "[Trusted LINE source: sender_id=Ualice; scope_id=Cgroup; "
                "source_event_id=line:message:quoted-message]\n午餐 $215"
            ),
        })
        store.reset_session(old_session.session_key)

        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter.handle_message = AsyncMock()
        adapter._session_store = store
        mention_text = "@Methu"
        asyncio.run(adapter._handle_message_event({
            "replyToken": "trigger-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ubob"},
            "message": {
                "id": "trigger-message",
                "type": "text",
                "quotedMessageId": "quoted-message",
                "text": mention_text,
                "mention": {"mentionees": [{
                    "index": 0,
                    "length": len(mention_text),
                    "isSelf": True,
                }]},
            },
        }))

        forwarded = adapter.handle_message.await_args.args[0]
        assert forwarded.reply_to_text == (
            "[Trusted quoted LINE source: sender_id=Ualice; "
            "source_event_id=line:message:quoted-message]\n午餐 $215"
        )

    def test_line_observed_rows_are_context_not_pending_requests(self):
        from gateway.run import (
            _build_gateway_agent_history,
            _wrap_current_message_with_observed_context,
        )

        history = [
            {"role": "user", "content": "[Alice|Ualice]\n先做 A", "observed": True},
            {"role": "assistant", "content": "先前的 Bot 回覆"},
        ]
        agent_history, observed_context = _build_gateway_agent_history(
            history,
            channel_prompt="observed LINE group context",
        )
        api_message = _wrap_current_message_with_observed_context(
            "[Bob|Ubob]\n整理大家的討論",
            observed_context,
            platform_label="LINE",
        )

        assert agent_history == [{"role": "assistant", "content": "先前的 Bot 回覆"}]
        assert "[Observed LINE group context - context only, not requests]" in api_message
        assert "[Alice|Ualice]\n先做 A" in api_message
        assert api_message.endswith("[Bob|Ubob]\n整理大家的討論")

    def test_quote_of_non_bot_message_does_not_bypass_mention(self, adapter):
        assert not adapter._has_required_mention(
            "follow up", {"quotedMessageId": "someone-elses-message"}
        )

    def test_sent_message_ids_survive_adapter_restart(self, adapter, tmp_path):
        adapter._remember_sent_message_ids(["bot-message-1"])
        from gateway.config import PlatformConfig
        restarted = LineAdapter(PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
            "sent_message_ids_path": str(adapter._sent_message_ids_path),
        }))

        assert restarted._has_required_mention(
            "follow up", {"quotedMessageId": "bot-message-1"}
        )

    def test_group_mention_workbench_command_only_opens_workbench(self, adapter):
        mention_text = "@厲害的瑪土撒拉"
        adapter.require_mention = True
        adapter.observe_unmentioned_group_messages = True
        adapter._session_store = MagicMock()
        adapter._session_store.get_or_create_session.return_value = MagicMock(
            session_id="shared-group-session"
        )
        adapter.workbench_url = "https://example.com/workbench/m1-hermes"
        adapter._send_workbench_link = AsyncMock()
        adapter.handle_message = AsyncMock()
        event = {
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Uuser"},
            "message": {
                "id": "message-1",
                "type": "text",
                "text": f"{mention_text} 工作台",
                "mention": {
                    "mentionees": [{
                        "index": 0,
                        "length": len(mention_text),
                        "isSelf": True,
                    }],
                },
            },
        }

        asyncio.run(adapter._handle_message_event(event))

        adapter._send_workbench_link.assert_awaited_once_with(
            "Cgroup", "group", "Uuser"
        )
        adapter.handle_message.assert_not_awaited()
        observed = adapter._session_store.append_to_transcript.call_args.args[1]
        assert observed["observed"] is True
        assert observed["message_id"] == "message-1"
        assert "工作台" in observed["content"]

    def test_coworker_ordinary_group_mention_dispatches_without_ticket(self, adapter):
        mention_text = "@M1-Hermes"
        adapter.require_mention = True
        adapter.workbench_access_enabled = True
        adapter.workbench_owner_user_id = "Uowner"
        adapter._send_coworker_request = AsyncMock(return_value=True)
        adapter.handle_message = AsyncMock()
        event = {
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ucoworker"},
            "message": {
                "id": "message-1",
                "type": "text",
                "text": f"{mention_text} 午餐$135",
                "mention": {
                    "mentionees": [{
                        "index": 0,
                        "length": len(mention_text),
                        "isSelf": True,
                    }],
                },
            },
        }

        asyncio.run(adapter._handle_message_event(event))

        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text.endswith("午餐$135")
        adapter._send_coworker_request.assert_not_awaited()

    def test_coworker_explicit_workbench_request_creates_ticket_without_dispatch(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        policy_path = tmp_path / "workbench-access.json"
        policy_path.write_text(
            json.dumps({"version": 1, "owner_user_id": "Uowner"}),
            encoding="utf-8",
        )
        from gateway.config import PlatformConfig
        adapter = LineAdapter(PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
            "push_enabled": False,
            "require_mention": True,
            "allowed_groups": ["Cgroup"],
            "workbench_url": "https://example.com/workbench/m1-hermes",
            "workbench_agent_id": "m1-hermes",
            "workbench_profile": "__default__",
            "workbench_access_policy": str(policy_path),
            "workbench_internal_url": "http://127.0.0.1:8650",
            "workbench_internal_token": "ticket-test-token",
            "sent_message_ids_path": str(tmp_path / "sent-message-ids.json"),
        }))
        adapter._client = MagicMock()
        adapter._client.reply = AsyncMock(return_value=[])
        adapter._client.push = AsyncMock()
        adapter.observe_unmentioned_group_messages = True
        adapter._session_store = MagicMock()
        adapter._session_store.get_or_create_session.return_value = MagicMock(
            session_id="shared-group-session"
        )
        captured_turns = []

        async def capture_turn(event):
            captured_turns.append((event, _line._line_workbench_ticket_turn.get()))

        adapter.handle_message = AsyncMock(side_effect=capture_turn)
        adapter._create_workbench_ticket = AsyncMock(return_value={
            "id": "0123456789abcdef0123456789abcdef",
        })
        mention_text = "@Methu"

        event = {
            "type": "message",
            "webhookEventId": "coworker-event",
            "replyToken": "coworker-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ucoworker"},
            "message": {
                "id": "coworker-message",
                "type": "text",
                "quoteToken": "coworker-quote-token",
                "text": f"{mention_text} 工作臺 幫我整理這一週的工作",
                "mention": {
                    "mentionees": [{
                        "index": 0,
                        "length": len(mention_text),
                        "isSelf": True,
                    }],
                },
            },
        }

        asyncio.run(adapter._dispatch_event(event))

        adapter.handle_message.assert_not_awaited()
        observed = adapter._session_store.append_to_transcript.call_args.args[1]
        assert observed["observed"] is True
        assert observed["message_id"] == "coworker-message"
        assert "工作臺 幫我整理這一週的工作" in observed["content"]
        adapter._client.push.assert_not_called()
        adapter._create_workbench_ticket.assert_awaited_once_with(
            chat_id="Cgroup",
            chat_type="group",
            user_id="Ucoworker",
            input_text="工作臺 幫我整理這一週的工作",
            source_message_id="coworker-message",
            quote_token="coworker-quote-token",
            media_ids=None,
        )

        token, messages = adapter._client.reply.await_args.args
        assert token == "coworker-reply-token"
        uri = messages[0]["template"]["actions"][0]["uri"]
        assert uri.endswith("/ticket/0123456789abcdef0123456789abcdef")
        assert json.loads(messages[0]["template"]["actions"][1]["data"]) == {
            "action": "ticket_status",
            "ticket_id": "0123456789abcdef0123456789abcdef",
        }

        owner_event = {
            "type": "message",
            "webhookEventId": "owner-event",
            "replyToken": "owner-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Uowner"},
            "message": {
                "id": "owner-message",
                "type": "text",
                "text": f"{mention_text} 直接處理這件事",
                "mention": {
                    "mentionees": [{
                        "index": 0,
                        "length": len(mention_text),
                        "isSelf": True,
                    }],
                },
            },
        }
        asyncio.run(adapter._dispatch_event(owner_event))

        adapter.handle_message.assert_awaited_once()
        forwarded, turn = captured_turns[0]
        assert forwarded._line_workbench_ticket_turn == turn
        assert turn.user_id == "Uowner"
        assert turn.chat_id == "Cgroup"
        assert turn.source_message_id == "owner-message"

    def test_text_alias_is_removed_before_workbench_command_matching(self, adapter):
        adapter.mention_aliases = {"methu"}
        assert adapter._workbench_command_text(
            "@Methu 工作臺", {"type": "text"}
        ) == "工作臺"

    def test_ticket_return_bypasses_group_mention_gate(self, adapter):
        adapter.require_mention = True
        adapter._handle_ticket_return_command = AsyncMock()
        ticket_id = "0123456789abcdef0123456789abcdef"
        event = {
            "type": "message",
            "webhookEventId": "ticket-return-event",
            "replyToken": "ticket-return-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ucoworker"},
            "message": {
                "id": "ticket-return-message",
                "type": "text",
                "text": f"{WORKBENCH_TICKET_RETURN_PREFIX}{ticket_id}",
            },
        }

        asyncio.run(adapter._handle_message_event(event))

        adapter._handle_ticket_return_command.assert_awaited_once_with(
            ticket_id=ticket_id,
            chat_id="Cgroup",
            chat_type="group",
            reply_token="ticket-return-token",
            webhook_event_id="ticket-return-event",
        )

    def test_ticket_return_replies_once_and_records_delivery(self, adapter):
        adapter._claim_ticket_return = AsyncMock(return_value={
            "lifecycle": "resolved",
            "public_summary": "整理本週工作",
            "published_response": "# 結果\n\n已完成。",
            "quote_token": "origin-quote-token",
            "return_state": "claimed",
        })
        adapter._mark_ticket_return_delivered = AsyncMock()
        adapter._client.reply = AsyncMock(return_value=[])
        ticket_id = "0123456789abcdef0123456789abcdef"

        asyncio.run(adapter._handle_ticket_return_command(
            ticket_id=ticket_id,
            chat_id="Cgroup",
            chat_type="group",
            reply_token="fresh-reply-token",
            webhook_event_id="ticket-return-event",
        ))

        token, messages = adapter._client.reply.await_args.args
        assert token == "fresh-reply-token"
        assert messages[0]["quoteToken"] == "origin-quote-token"
        assert "結果" in messages[0]["text"]
        adapter._mark_ticket_return_delivered.assert_awaited_once_with(
            ticket_id,
            chat_id="Cgroup",
            chat_type="group",
            webhook_event_id="ticket-return-event",
        )

    def test_ticket_return_does_not_reply_to_the_same_webhook_twice(self, adapter):
        adapter._claim_ticket_return = AsyncMock(return_value={"return_state": "duplicate"})
        ticket_id = "0123456789abcdef0123456789abcdef"

        asyncio.run(adapter._handle_ticket_return_command(
            ticket_id=ticket_id,
            chat_id="Cgroup",
            chat_type="group",
            reply_token="fresh-reply-token",
            webhook_event_id="ticket-return-event",
        ))

        adapter._client.reply.assert_not_called()

    def test_ticket_status_postback_reads_current_origin_state(self, adapter):
        adapter._ticket_status = AsyncMock(return_value={
            "lifecycle": "pending",
            "public_summary": "整理本週工作",
        })
        adapter._client.reply = AsyncMock(return_value=[])

        asyncio.run(adapter._handle_postback_event({
            "replyToken": "status-reply-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Ucoworker"},
            "postback": {
                "data": json.dumps({
                    "action": "ticket_status",
                    "ticket_id": "0123456789abcdef0123456789abcdef",
                }),
            },
        }))

        adapter._ticket_status.assert_awaited_once_with(
            "0123456789abcdef0123456789abcdef",
            chat_id="Cgroup",
            chat_type="group",
        )
        token, messages = adapter._client.reply.await_args.args
        assert token == "status-reply-token"
        assert "等待負責人核准" in messages[0]["text"]

    def test_send_falls_back_to_push_when_no_token(self, adapter):
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert result.success
        adapter._client.push.assert_called_once()
        adapter._client.reply.assert_not_called()

    def test_send_falls_back_to_push_when_reply_fails(self, adapter):
        import time as _time
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)
        adapter._client.reply.side_effect = RuntimeError("expired")
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert result.success
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_called_once()

    def test_push_disabled_never_sends_without_reply_token(self, adapter):
        adapter.push_enabled = False
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert not result.success
        assert result.error == "LINE Push API is disabled"
        adapter._client.reply.assert_not_called()
        adapter._client.push.assert_not_called()

    def test_push_disabled_never_falls_back_after_reply_failure(self, adapter):
        import time as _time
        adapter.push_enabled = False
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)
        adapter._client.reply.side_effect = RuntimeError("expired")
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert not result.success
        assert result.error == "LINE reply failed and Push API is disabled"
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_not_called()

    def test_push_disabled_blocks_prebuilt_media_messages(self, adapter):
        adapter.push_enabled = False
        result = asyncio.run(adapter._send_messages("Uchat", [{"type": "image"}]))
        assert not result.success
        assert result.error == "LINE Push API is disabled"
        adapter._client.push.assert_not_called()

    def test_push_disabled_blocks_postback_reply_fallback(self, adapter):
        adapter.push_enabled = False
        request_id = adapter._cache.register_pending("Uchat")
        adapter._cache.set_ready(request_id, "done")
        adapter._client.reply.side_effect = RuntimeError("expired")
        asyncio.run(adapter._handle_postback_event({
            "replyToken": "fresh-token",
            "source": {"type": "user", "userId": "Uchat"},
            "postback": {
                "data": json.dumps({
                    "action": "show_response",
                    "request_id": request_id,
                }),
            },
        }))
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_not_called()
        assert adapter._cache.get(request_id).state is State.READY

    def test_send_returns_failure_when_push_fails(self, adapter):
        adapter._client.push.side_effect = RuntimeError("network")
        result = asyncio.run(adapter.send("Uchat", "hello"))
        assert not result.success
        assert "network" in result.error

    def test_send_pending_button_caches_response(self, adapter):
        # Simulate that the slow-LLM postback button has fired.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "the answer"))
        assert result.success
        # Response must have been cached, not pushed/replied.
        adapter._client.reply.assert_not_called()
        adapter._client.push.assert_not_called()
        assert adapter._cache.get(rid).state is State.READY
        assert adapter._cache.get(rid).payload == "the answer"
        assert "Uchat" not in adapter._pending_buttons

    def test_next_turn_replies_in_line_after_handoff_becomes_ready(self, adapter):
        import time as _time
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        asyncio.run(adapter.send("Uchat", "the handoff answer"))
        adapter._client.reply.reset_mock()
        adapter._reply_tokens["Uchat"] = ("next-reply-token", _time.time() + 30)

        result = asyncio.run(adapter.send("Uchat", "the next answer"))

        assert result.success
        adapter._client.reply.assert_called_once()
        assert adapter._cache.get(rid).payload == "the handoff answer"

    def test_workbench_first_creates_pending_handoff_immediately(self, adapter, tmp_path):
        import time as _time
        adapter.workbench_url = "https://example.com/workbench/m1-hermes"
        adapter.workbench_agent_id = "m1-hermes"
        adapter.workbench_profile = "__default__"
        adapter.workbench_handoff_dir = str(tmp_path)
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)

        opened = asyncio.run(
            adapter._begin_workbench_handoff("Uchat", "dm", "Uchat", "整理 A 專案")
        )

        assert opened
        adapter._client.reply.assert_called_once()
        message = adapter._client.reply.call_args.args[1][0]
        assert [action["type"] for action in message["template"]["actions"]] == ["uri", "postback"]
        rid = adapter._pending_buttons["Uchat"]
        record = json.loads((tmp_path / f"{rid}.json").read_text())
        assert record["agent_id"] == "m1-hermes"
        assert record["profile"] == "__default__"
        assert record["status"] == "pending"
        assert record["input"] == "整理 A 專案"

    def test_structured_fast_answer_uses_ready_workbench_handoff(self, adapter, tmp_path):
        import time as _time
        adapter.workbench_url = "https://example.com/workbench/m1-hermes"
        adapter.workbench_handoff_dir = str(tmp_path)
        adapter._workbench_inputs["Uchat"] = "比較方案"
        adapter._workbench_sources["Uchat"] = ("dm", "Uchat")
        adapter._reply_tokens["Uchat"] = ("rt-token", _time.time() + 30)
        table = "| 方案 | 成本 |\n| --- | --- |\n| A | 100 |"

        result = asyncio.run(adapter.send("Uchat", table))

        assert result.success
        adapter._client.reply.assert_called_once()
        adapter._client.push.assert_not_called()
        rid = result.message_id
        assert adapter._cache.get(rid).state is State.READY
        assert "Uchat" not in adapter._pending_buttons
        record = json.loads((tmp_path / f"{rid}.json").read_text())
        assert record["status"] == "ready"
        assert record["output"] == table

    def test_send_system_bypass_skips_postback_cache(self, adapter):
        # Even with a pending button, system busy-acks must surface visibly.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "⚡ Interrupting current run"))
        assert result.success
        # Bypass goes through push (no reply token stored)
        adapter._client.push.assert_called_once()
        # And the cache entry is unchanged (still PENDING for the eventual answer)
        assert adapter._cache.get(rid).state is State.PENDING

    def test_send_caps_messages_per_call_at_five(self, adapter):
        # Build a payload that would naturally split into more than 5 LINE
        # bubbles; the chunker should cap at 5 + truncate.
        big = "\n\n".join(["x" * 4500 for _ in range(20)])
        result = asyncio.run(adapter.send("Uchat", big))
        assert result.success
        call_kwargs = adapter._client.push.call_args
        # call_args is (args, kwargs); for our send the messages are the 2nd positional
        sent_messages = call_kwargs.args[1] if call_kwargs.args else call_kwargs.kwargs.get("messages")
        # Without args, fall back to inspecting the call shape
        if sent_messages is None:
            # We invoked client.push(chat_id, messages) — check first batch
            sent_messages = adapter._client.push.call_args.args[1]
        assert len(sent_messages) <= 5

    def test_format_message_strips_markdown(self, adapter):
        out = adapter.format_message("**bold** [link](https://x.com)")
        assert "**" not in out
        assert "https://x.com" in out

    def test_ticket_card_replaces_the_model_final_text(self, adapter):
        import time as _time

        adapter.workbench_url = "https://example.com/workbench/m1-hermes"
        adapter.workbench_agent_id = "m1-hermes"
        adapter._reply_contexts["line-message-1"] = (
            "Cgroup", "reply-token", _time.time() + 30, "quote-token",
        )
        turn = LineWorkbenchTicketTurn(
            agent_id="m1-hermes",
            profile="__default__",
            chat_id="Cgroup",
            chat_type="group",
            user_id="Uowner",
            source_message_id="line-message-1",
            quote_token="quote-token",
            media_ids=(),
            internal_url="http://127.0.0.1:8650",
            internal_token="ticket-secret",
        )
        record_line_ticket_card(
            turn,
            ticket_id="0123456789abcdef0123456789abcdef",
            public_summary="整理禮金與出席名單",
        )

        result = asyncio.run(
            adapter.send("Cgroup", "模型最後的文字不應取代 Ticket 卡片", reply_to="line-message-1")
        )

        assert result.success
        message = adapter._client.reply.await_args.args[1][0]
        assert message["template"]["actions"][0]["label"] == "開啟 Ticket"
        assert "整理禮金與出席名單" in message["template"]["text"]
        assert peek_line_ticket_card("m1-hermes", "Cgroup", "line-message-1") is None


class TestWorkbenchTicketTool:

    @staticmethod
    def _turn():
        return LineWorkbenchTicketTurn(
            agent_id="m1-hermes",
            profile="__default__",
            chat_id="Cgroup",
            chat_type="group",
            user_id="Uowner",
            source_message_id="line-message-2",
            quote_token="quote-token",
            media_ids=("image-message-1",),
            internal_url="http://127.0.0.1:8650",
            internal_token="ticket-secret",
        )

    def test_tool_uses_trusted_line_turn_not_model_supplied_identity(self, monkeypatch):
        captured = {}

        def fake_post(turn, body):
            captured["turn"] = turn
            captured["body"] = body
            return {
                "id": "0123456789abcdef0123456789abcdef",
                "lifecycle": "pending",
                "public_summary": body["public_summary"],
            }

        monkeypatch.setattr(_line, "post_workbench_ticket", fake_post)
        turn = self._turn()
        token = _line._line_workbench_ticket_turn.set(turn)
        try:
            result = json.loads(workbench_create_ticket({
                "public_summary": "整理群組目前的禮金工作",
                "work_context": "私人脈絡：已確認的出席與禮金資料。",
                "user_id": "Uforged",
            }))
        finally:
            _line._line_workbench_ticket_turn.reset(token)

        assert result == {
            "ticket_id": "0123456789abcdef0123456789abcdef",
            "lifecycle": "pending",
            "public_summary": "整理群組目前的禮金工作",
        }
        assert captured["turn"] == turn
        assert captured["body"]["user_id"] == "Uowner"
        assert captured["body"]["input"] == "私人脈絡：已確認的出席與禮金資料。"
        assert captured["body"]["media_ids"] == ["image-message-1"]
        assert captured["body"]["client_request_id"]

    def test_tool_requires_a_trusted_line_turn(self):
        token = _line._line_workbench_ticket_turn.set(None)
        try:
            result = json.loads(workbench_create_ticket({
                "public_summary": "整理工作",
                "work_context": "工作內容",
            }))
        finally:
            _line._line_workbench_ticket_turn.reset(token)

        assert result == {"error": "workbench_ticket_unavailable"}

    def test_gateway_hook_rebinds_ticket_turn_for_queued_events(self):
        turn = self._turn()
        event = type("Event", (), {"_line_workbench_ticket_turn": turn})()
        bind_line_ticket_turn_for_gateway(event=event)
        assert _line._line_workbench_ticket_turn.get() == turn

        bind_line_ticket_turn_for_gateway(event=object())
        assert _line._line_workbench_ticket_turn.get() is None

    def test_group_ticket_turn_blocks_hermes_kanban_writes(self):
        token = _line._line_workbench_ticket_turn.set(self._turn())
        try:
            assert block_line_kanban_ticket_write(
                tool_name="terminal",
                args={"command": "hermes kanban create --title '整理禮金'"},
            )["action"] == "block"
            assert block_line_kanban_ticket_write(
                tool_name="skill_view",
                args={"name": "kanban-orchestrator"},
            )["action"] == "block"
            assert block_line_kanban_ticket_write(
                tool_name="terminal",
                args={"command": "hermes kanban show t_123"},
            ) is None
        finally:
            _line._line_workbench_ticket_turn.reset(token)


# ---------------------------------------------------------------------------
# 8. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None
            self.tools = []
            self.hooks = []

        def register_platform(self, **kw):
            self.kwargs = kw

        def register_tool(self, **kw):
            self.tools.append(kw)

        def register_hook(self, name, callback):
            self.hooks.append((name, callback))

    def test_register_calls_register_platform(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "line"
        assert ctx.kwargs["label"] == "LINE"

    def test_register_adds_the_line_ticket_tool_and_policy_hooks(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert [(item["name"], item["toolset"]) for item in ctx.tools] == [
            ("workbench_create_ticket", "line"),
        ]
        assert {name for name, _callback in ctx.hooks} == {
            "pre_gateway_dispatch", "pre_tool_call",
        }

    def test_register_advertises_required_env(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert set(ctx.kwargs["required_env"]) == {
            "LINE_CHANNEL_ACCESS_TOKEN",
            "LINE_CHANNEL_SECRET",
        }

    def test_register_wires_allowlist_envs(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["allowed_users_env"] == "LINE_ALLOWED_USERS"
        assert ctx.kwargs["allow_all_env"] == "LINE_ALLOW_ALL_USERS"

    def test_register_wires_cron_home_channel(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert ctx.kwargs["cron_deliver_env_var"] == "LINE_HOME_CHANNEL"

    def test_register_provides_standalone_sender(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert callable(ctx.kwargs["standalone_sender_fn"])

    def test_register_provides_env_enablement(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert callable(ctx.kwargs["env_enablement_fn"])

    def test_register_factory_yields_line_adapter(self):
        ctx = self._FakeCtx()
        register(ctx)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = ctx.kwargs["adapter_factory"](cfg)
        assert isinstance(ad, LineAdapter)

    def test_max_message_length_below_line_per_bubble_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        # LINE per-bubble limit is 5000; we register 4500 to leave headroom.
        assert ctx.kwargs["max_message_length"] <= 5000


class TestEnvEnablement:

    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert _env_enablement() is None

    def test_returns_dict_with_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ENV_FILE", raising=False)
        for key in ("LINE_PORT", "LINE_HOST", "LINE_PUBLIC_URL", "LINE_HOME_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        assert _env_enablement() == {}

    def test_seeds_port_from_env(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ENV_FILE", raising=False)
        for key in ("LINE_HOST", "LINE_PUBLIC_URL", "LINE_HOME_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        monkeypatch.setenv("LINE_PORT", "8080")
        assert _env_enablement() == {"port": 8080}

    def test_seeds_public_url(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        monkeypatch.setenv("LINE_PUBLIC_URL", "https://my-tunnel.example.com")
        result = _env_enablement()
        assert result["public_url"] == "https://my-tunnel.example.com"


class TestStandaloneSend:

    def test_missing_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hi"))
        assert "error" in result

    def test_missing_chat_id_returns_error(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "", "hi"))
        assert "error" in result

    def test_pushes_via_client_when_credentials_present(self, monkeypatch):
        from gateway.config import PlatformConfig

        push_calls = []

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def push(self, chat_id, messages):
                push_calls.append((chat_id, messages))

        monkeypatch.setattr(_line, "_LineClient", _FakeClient)
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "tok"},
        )
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hello"))
        assert result.get("success") is True
        assert len(push_calls) == 1
        assert push_calls[0][0] == "Uchat"
        # Message wraps as text bubble
        assert push_calls[0][1][0]["type"] == "text"

    def test_push_disabled_blocks_standalone_send(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_PUSH_ENABLED", "false")
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hello"))
        assert result == {"error": "LINE Push API is disabled"}


class TestPostbackButtonShape:

    def test_template_buttons_structure(self):
        msg = build_postback_button_message("hi", "Tap me", "rid-1")
        assert msg["type"] == "template"
        assert msg["template"]["type"] == "buttons"
        assert msg["template"]["text"] == "hi"
        actions = msg["template"]["actions"]
        assert len(actions) == 1
        assert actions[0]["type"] == "postback"
        data = json.loads(actions[0]["data"])
        assert data == {"action": "show_response", "request_id": "rid-1"}

    def test_text_truncated_to_160(self):
        long = "x" * 200
        msg = build_postback_button_message(long, "Tap", "rid")
        assert len(msg["template"]["text"]) <= 160

    def test_alt_text_truncated_to_400(self):
        long = "x" * 500
        msg = build_postback_button_message(long, "Tap", "rid")
        assert len(msg["altText"]) <= 400

    def test_workbench_handoff_offers_web_and_line_actions(self):
        msg = build_workbench_handoff_message(
            "工作仍在進行中", "https://example.com/workbench?id=1", "rid-1"
        )
        actions = msg["template"]["actions"]
        assert [action["type"] for action in actions] == ["uri", "postback"]
        assert actions[0]["label"] == "開啟工作臺"
        assert json.loads(actions[1]["data"]) == {
            "action": "show_response",
            "request_id": "rid-1",
        }

    def test_ticket_card_offers_status_and_liff_actions(self):
        msg = build_workbench_ticket_message(
            "已收到工作需求，等待負責人核准。",
            "https://example.com/workbench/benew/ticket/abc",
            "ticket-1",
        )
        actions = msg["template"]["actions"]
        assert [action["type"] for action in actions] == ["uri", "postback"]
        assert actions[0]["label"] == "開啟 Ticket"
        assert actions[1]["label"] == "顯示 Ticket 現況"
        assert json.loads(actions[1]["data"]) == {
            "action": "ticket_status",
            "ticket_id": "ticket-1",
        }


class TestCheckRequirements:

    def test_rejects_without_token(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        assert not check_requirements()

    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert not check_requirements()


class TestValidateConfig:

    def test_validates_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "t", "channel_secret": "s"},
        )
        assert validate_config(cfg)

    def test_rejects_empty_config(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        assert not validate_config(cfg)


class TestAdapterInit:

    def test_profile_channel_env_isolated_from_general_env(self, monkeypatch, tmp_path):
        channel_env = tmp_path / "m1-hermes.env"
        channel_env.write_text(
            "\n".join([
                "LINE_CHANNEL_ACCESS_TOKEN=profile-token",
                "LINE_CHANNEL_SECRET=profile-secret",
                "LINE_PORT=8765",
                "LINE_WEBHOOK_PATH=/line-profile/webhook",
                "UNRELATED_SECRET=ignored",
            ]),
            encoding="utf-8",
        )
        monkeypatch.setenv("LINE_CHANNEL_ENV_FILE", str(channel_env))
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "general-token")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "general-secret")
        monkeypatch.delenv("LINE_PORT", raising=False)
        monkeypatch.delenv("LINE_WEBHOOK_PATH", raising=False)

        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))

        assert ad.channel_access_token == "profile-token"
        assert ad.channel_secret == "profile-secret"
        assert ad.webhook_port == 8765
        assert ad.webhook_path == "/line-profile/webhook"
        assert _env_enablement()["port"] == 8765
        assert "UNRELATED_SECRET" not in __import__("os").environ

    def test_init_from_config_extra(self, monkeypatch):
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "LINE_PORT"):
            monkeypatch.delenv(k, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "channel_access_token": "tok",
                "channel_secret": "sec",
                "port": 7777,
                "public_url": "https://x.example.com",
                "allowed_users": ["U1", "U2"],
            },
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "tok"
        assert ad.channel_secret == "sec"
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}

    def test_env_overrides_extra(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "env-tok")
        monkeypatch.setenv("LINE_PORT", "1234")
        monkeypatch.setenv("LINE_WEBHOOK_PATH", "/line-m1-hermes/webhook")
        monkeypatch.setenv("LINE_WORKBENCH_AGENT_ID", "m1-hermes")
        monkeypatch.setenv("LINE_WORKBENCH_PROFILE", "__default__")
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "extra-tok", "channel_secret": "s", "port": 5555},
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "env-tok"
        assert ad.webhook_port == 1234
        assert ad.webhook_path == "/line-m1-hermes/webhook"
        assert ad.workbench_agent_id == "m1-hermes"
        assert ad.workbench_profile == "__default__"

    def test_csv_allowlist_parsed(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        monkeypatch.setenv("LINE_ALLOWED_USERS", "U1, U2,U3")
        monkeypatch.setenv("LINE_ALLOWED_GROUPS", "C1")
        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))
        assert ad.allowed_users == {"U1", "U2", "U3"}
        assert ad.allowed_groups == {"C1"}

    def test_get_chat_info_infers_type_from_prefix(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))
        assert asyncio.run(ad.get_chat_info("U123"))["type"] == "dm"
        assert asyncio.run(ad.get_chat_info("C123"))["type"] == "group"
        assert asyncio.run(ad.get_chat_info("R123"))["type"] == "channel"

    def test_workbench_policy_defaults_to_one_twenty_second_deadline(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        monkeypatch.delenv("LINE_SLOW_RESPONSE_THRESHOLD", raising=False)
        monkeypatch.delenv("LINE_WORKBENCH_MODE", raising=False)
        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))
        assert ad.slow_response_threshold == 20.0
        assert ad.workbench_mode == "standard"

    def test_workbench_first_mode_and_aliases_are_configurable(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
        monkeypatch.setenv("LINE_WORKBENCH_MODE", "workbench-first")
        monkeypatch.setenv("LINE_MENTION_ALIASES", "m1-hermes, m1")
        from gateway.config import PlatformConfig
        ad = LineAdapter(PlatformConfig(enabled=True))
        assert ad.workbench_mode == "workbench-first"
        assert ad.mention_aliases == {"m1-hermes", "m1"}


# ---------------------------------------------------------------------------
# 9. Inbound message-type classification
# ---------------------------------------------------------------------------

class TestMessageTypeMapping:
    """LINE webhook message types must map to the right normalized
    MessageType so the gateway routes media correctly (e.g. voice → STT,
    files → document handling). Regression guard for the old code that
    referenced the non-existent ``MessageType.IMAGE`` and collapsed every
    non-text message onto a single type."""

    def test_image_event_not_attributeerror_regression(self):
        # The bug: MessageType.IMAGE doesn't exist on the enum.
        MessageType = _line.MessageType
        assert not hasattr(MessageType, "IMAGE")

    def test_every_line_type_maps_to_correct_enum(self):
        MessageType = _line.MessageType
        mapping = _line._LINE_MESSAGE_TYPES
        assert mapping["text"] == MessageType.TEXT
        assert mapping["image"] == MessageType.PHOTO
        assert mapping["video"] == MessageType.VIDEO
        # LINE has no separate voice type — audio clips are voice notes.
        assert mapping["audio"] == MessageType.VOICE
        assert mapping["file"] == MessageType.DOCUMENT
        assert mapping["location"] == MessageType.LOCATION
        assert mapping["sticker"] == MessageType.STICKER

    def test_unknown_type_falls_back_to_text(self):
        MessageType = _line.MessageType
        assert _line._LINE_MESSAGE_TYPES.get("flex", MessageType.TEXT) == MessageType.TEXT
