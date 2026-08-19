"""Regression tests for Codex refresh_token self-heal (cross-store rotation).

Hermes keeps its OWN copy of the Codex OAuth token (per profile + top-level),
separate from the Codex CLI's ``~/.codex/auth.json``. OAuth refresh_tokens are
single-use, so when the Codex CLI (or another Hermes process) rotates the shared
token, the frozen copy's refresh_token goes stale and ``refresh_codex_oauth_pure``
fails with a relogin-required error. ``_refresh_codex_auth_tokens`` must then
recover by re-importing the canonical token from ``~/.codex/auth.json`` instead of
surfacing a hard 401 — but ONLY for relogin-required failures, never for transient
ones (e.g. 429 quota, where the stored token is still valid).
"""

import base64
import json

import pytest

import hermes_cli.auth as auth
from hermes_cli.auth import AuthError, _refresh_codex_auth_tokens, resolve_codex_runtime_credentials

def _jwt(account_id: str) -> str:
    payload = {
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        "exp": 4102444800,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"h.{encoded}.s"


STALE = {"access_token": _jwt("account-a"), "refresh_token": "stale-refresh"}


def _isolate_refresh_transaction(monkeypatch, tmp_path, saved):
    owner_path = tmp_path / "owner-auth.json"
    monkeypatch.setattr(auth, "_provider_auth_owner_path", lambda *a, **k: owner_path)
    monkeypatch.setattr(auth, "_read_owned_codex_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        auth,
        "_commit_owned_codex_tokens",
        lambda _path, tokens, **_kwargs: saved.update(tokens),
    )


def test_self_heals_on_stale_refresh_token(tmp_path, monkeypatch):
    """invalid_grant (relogin-required) → reimport from ~/.codex and persist it."""
    saved = {}
    fresh = {
        "access_token": _jwt("account-a"),
        "refresh_token": "fresh-refresh",
        "last_refresh": "2026-06-12T00:00:00Z",
    }
    _isolate_refresh_transaction(monkeypatch, tmp_path, saved)

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: dict(fresh))

    out = _refresh_codex_auth_tokens(STALE, 20.0)

    assert out["access_token"] == _jwt("account-a")
    assert out["refresh_token"] == "fresh-refresh"
    # the recovered token was persisted to the Hermes auth store
    assert saved["access_token"] == _jwt("account-a")


def test_does_not_self_heal_on_rate_limit(tmp_path, monkeypatch):
    """429 quota keeps relogin_required=False — token still valid, must NOT reimport."""
    import_calls = {"n": 0}
    _isolate_refresh_transaction(monkeypatch, tmp_path, {})

    def _rate_limited(*_a, **_k):
        raise AuthError(
            "quota exhausted",
            provider="openai-codex",
            code="codex_rate_limited",
            relogin_required=False,
        )

    def _import_spy():
        import_calls["n"] += 1
        return {"access_token": "should-not-be-used"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rate_limited)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", _import_spy)
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda *a, **k: None)

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "codex_rate_limited"
    assert import_calls["n"] == 0  # never touched ~/.codex on a transient failure


def test_reraises_when_codex_cli_token_absent(tmp_path, monkeypatch):
    """relogin-required but ~/.codex unavailable/expired → propagate original error."""

    _isolate_refresh_transaction(monkeypatch, tmp_path, {})

    def _reused(*_a, **_k):
        raise AuthError(
            "refresh token reused",
            provider="openai-codex",
            code="refresh_token_reused",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _reused)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: None)
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda *a, **k: None)

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "refresh_token_reused"


def test_happy_path_unchanged(tmp_path, monkeypatch):
    """Normal refresh succeeds → rotated tokens persisted, ~/.codex never consulted."""
    saved = {}
    import_calls = {"n": 0}
    _isolate_refresh_transaction(monkeypatch, tmp_path, saved)

    def _import_spy():
        import_calls["n"] += 1
        return None

    monkeypatch.setattr(
        auth,
        "refresh_codex_oauth_pure",
        lambda *a, **k: {"access_token": "rotated", "refresh_token": "rotated-r"},
    )
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", _import_spy)

    out = _refresh_codex_auth_tokens({"access_token": "a", "refresh_token": "b"}, 20.0)

    assert out["access_token"] == "rotated"
    assert out["refresh_token"] == "rotated-r"
    assert saved["access_token"] == "rotated"
    assert import_calls["n"] == 0  # happy path must not consult ~/.codex


def test_reraises_when_imported_token_lacks_refresh_token(tmp_path, monkeypatch):
    """relogin-required, but ~/.codex returns an access_token with NO refresh_token →
    re-raise rather than persist a half-token that would break the next refresh."""
    saved = {}
    _isolate_refresh_transaction(monkeypatch, tmp_path, saved)

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: {"access_token": "fresh-only"})

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "invalid_grant"
    assert saved == {}  # nothing was persisted


def test_refuses_codex_cli_token_from_different_account(tmp_path, monkeypatch):
    """Terminal refresh errors must not silently switch the Hermes account."""
    saved = {}
    _isolate_refresh_transaction(monkeypatch, tmp_path, saved)

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(
        auth,
        "_import_codex_cli_tokens",
        lambda: {
            "access_token": _jwt("account-b"),
            "refresh_token": "account-b-refresh",
        },
    )

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "invalid_grant"
    assert saved == {}


def test_provider_owner_is_global_store_for_profile_fallback(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    profile_path.parent.mkdir(parents=True)
    root_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps({"version": 1, "providers": {}}))
    root_path.write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "root-a", "refresh_token": "root-r"}
            }
        },
    }))
    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)

    assert auth._provider_auth_owner_path("openai-codex") == root_path


def test_self_heals_missing_singleton_access_token_from_codex_cli(tmp_path, monkeypatch):
    """Exact cron failure path: Hermes auth has refresh_token but missing access_token."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"refresh_token": "stale-refresh"},
                "last_refresh": "2026-06-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    resolved = resolve_codex_runtime_credentials()

    assert resolved["api_key"] == "fresh-access"
    assert resolved["source"] == "hermes-auth-store"
    stored = json.loads((hermes_home / "auth.json").read_text())
    tokens = stored["providers"]["openai-codex"]["tokens"]
    assert tokens["access_token"] == "fresh-access"
    assert tokens["refresh_token"] == "fresh-refresh"


def test_missing_singleton_access_token_reraises_when_codex_cli_half_token(tmp_path, monkeypatch):
    """Missing access_token must not be masked by a malformed Codex CLI import."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"refresh_token": "stale-refresh"},
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "fresh-only"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(AuthError) as ei:
        resolve_codex_runtime_credentials()

    assert ei.value.code == "codex_auth_missing_access_token"
