"""Regression tests for credential-pool OAuth refresh write-through to root.

Companion to ``tests/hermes_cli/test_xai_oauth_writethrough.py``. That file
covers the *non-pool* xAI refresh path (``_save_xai_oauth_tokens``). These
cover the **credential-pool** refresh path
(``CredentialPool._sync_device_code_entry_to_auth_store``): when a profile
that has no own ``providers.<id>`` block refreshes — via the pool — a rotating
OAuth grant it resolved from the global-root fallback, the rotated chain must
be written back to the global root too. Otherwise root keeps a revoked refresh
token and every other profile reading root's stale grant dies with
``refresh_token_reused`` / ``invalid_grant`` once its access token expires
(issue #48415, the Codex/xAI analog of #43589).

The tests drive the real ``_sync_device_code_entry_to_auth_store`` against
real on-disk auth stores (profile + root under ``tmp_path``) rather than
mocking the save boundary, so they exercise the actual atomic write path.
"""

import json

import pytest

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)
from hermes_cli import auth as A


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(
    provider: str,
    *,
    id: str,
    access_token: str,
    refresh_token: str,
    source: str = "device_code",
):
    return PooledCredential(
        provider=provider,
        id=id,
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
    )


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    The pytest seat belt in ``_write_through_provider_state_to_global_root``
    only refuses the *real* user's ``$HOME/.hermes/auth.json``; a tmp_path
    root is allowed, so point HOME away from the tmp root to keep the guard
    from tripping on these fixtures.
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


@pytest.mark.parametrize(
    "provider",
    ["openai-codex", "xai-oauth"],
)
def test_pool_refresh_writes_through_to_root_when_profile_reads_root(
    profile_and_root, provider
):
    """A profile reading root's grant must push rotated tokens back to root."""
    profile_path, root_path = profile_and_root
    # Profile has NO own provider block (reads root via fallback).
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    pool = CredentialPool(provider, [])
    pool._sync_device_code_entry_to_auth_store(
        _entry(provider, id="e1", access_token="new-access", refresh_token="new-refresh")
    )

    # Profile got the rotated chain (existing behavior).
    profile = _read_store(profile_path)
    assert (
        profile["providers"][provider]["tokens"]["refresh_token"] == "new-refresh"
    )

    # AND the global root no longer holds the revoked refresh token (#48415).
    root = _read_store(root_path)
    assert root["providers"][provider]["tokens"]["access_token"] == "new-access"
    assert root["providers"][provider]["tokens"]["refresh_token"] == "new-refresh"


@pytest.mark.parametrize(
    "provider",
    ["openai-codex", "xai-oauth"],
)
def test_pool_refresh_does_not_touch_root_when_profile_shadows(
    profile_and_root, provider
):
    """A profile that genuinely shadows root must NOT clobber the root grant."""
    profile_path, root_path = profile_and_root
    # Profile has its OWN provider block: it shadows root legitimately.
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "profile-old",
                        "refresh_token": "profile-old-refresh",
                    }
                }
            },
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "root-untouched",
                        "refresh_token": "root-untouched-refresh",
                    }
                }
            },
        },
    )

    pool = CredentialPool(provider, [])
    pool._sync_device_code_entry_to_auth_store(
        _entry(
            provider,
            id="e2",
            access_token="profile-new",
            refresh_token="profile-new-refresh",
        )
    )

    profile = _read_store(profile_path)
    assert (
        profile["providers"][provider]["tokens"]["refresh_token"]
        == "profile-new-refresh"
    )

    # Root keeps its own grant — write-through must not run when the profile
    # owns the block.
    root = _read_store(root_path)
    assert (
        root["providers"][provider]["tokens"]["refresh_token"]
        == "root-untouched-refresh"
    )


def test_write_through_helper_is_noop_in_classic_mode(monkeypatch, tmp_path):
    """When profile == root (classic mode), the helper must be a no-op.

    ``_global_auth_file_path`` returns None in classic mode; the profile save
    already wrote to root, so a second write would be redundant (and the
    helper has nothing to target).
    """
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    # Must not raise and must not attempt any write.
    CP._write_through_provider_state_to_global_root(
        "openai-codex", {"tokens": {"access_token": "a", "refresh_token": "r"}}
    )


def test_codex_refresh_locks_and_commits_the_global_owner(
    profile_and_root, monkeypatch
):
    """A profile fallback refresh updates root without creating a profile copy."""
    profile_path, root_path = profile_and_root
    entry = _entry(
        "openai-codex",
        id="root-entry",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
            "credential_pool": {
                "openai-codex": [{
                    "id": "root-entry",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                }]
            },
        },
    )
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        lambda *_a, **_k: {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "last_refresh": "2026-07-16T00:00:00Z",
        },
    )

    refreshed = CredentialPool("openai-codex", [entry])._refresh_entry(
        entry, force=True
    )

    assert refreshed is not None
    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"
    assert root["credential_pool"]["openai-codex"][0]["refresh_token"] == "new-refresh"
    profile = _read_store(profile_path)
    assert profile.get("providers") == {}
    assert "credential_pool" not in profile


def test_independent_manual_codex_refresh_does_not_replace_singleton(
    profile_and_root, monkeypatch
):
    """An independent manual account updates only its exact pool entry."""
    profile_path, root_path = profile_and_root
    entry = _entry(
        "openai-codex",
        id="manual-b",
        access_token="account-b-access",
        refresh_token="account-b-refresh",
        source="manual:device_code",
    )
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "account-a-access",
                        "refresh_token": "account-a-refresh",
                    }
                }
            },
            "credential_pool": {
                "openai-codex": [{
                    "id": "manual-b",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "account-b-access",
                    "refresh_token": "account-b-refresh",
                }]
            },
        },
    )
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        lambda *_a, **_k: {
            "access_token": "account-b-new-access",
            "refresh_token": "account-b-new-refresh",
        },
    )

    refreshed = CredentialPool("openai-codex", [entry])._refresh_entry(
        entry, force=True
    )

    assert refreshed is not None
    root = _read_store(root_path)
    singleton = root["providers"]["openai-codex"]["tokens"]
    manual = root["credential_pool"]["openai-codex"][0]
    assert singleton["access_token"] == "account-a-access"
    assert manual["access_token"] == "account-b-new-access"
