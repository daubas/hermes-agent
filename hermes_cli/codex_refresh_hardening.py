"""Portable Codex OAuth refresh hardening for mixed Hermes installations.

This module intentionally depends only on the standard library and the small
set of auth/credential-pool APIs shared by older Hermes releases.  It can be
installed from the bottom of ``hermes_cli.auth`` and ``agent.credential_pool``
without replacing either version-specific module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


_THREAD_STATE = threading.local()
_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _fingerprint(value: Any) -> str:
    if not value:
        return "none"
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _jwt_payload(token: str) -> Dict[str, Any]:
    if not token:
        return {}
    try:
        segment = token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        value = json.loads(base64.urlsafe_b64decode(segment).decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _account_identity(access_token: str) -> Optional[str]:
    payload = _jwt_payload(access_token)
    auth_claim = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        for key in ("chatgpt_account_id", "account_id"):
            if auth_claim.get(key):
                return str(auth_claim[key])
    for key in ("chatgpt_account_id", "account_id", "sub"):
        if payload.get(key):
            return str(payload[key])
    return None


def _store_paths(namespace: Dict[str, Any]) -> Iterable[Path]:
    local = Path(namespace["_auth_file_path"]())
    yield local
    global_fn = namespace.get("_global_auth_file_path")
    if not callable(global_fn):
        return
    try:
        global_path = global_fn()
    except Exception:
        return
    if global_path is not None and Path(global_path) != local:
        yield Path(global_path)


def _load_store(namespace: Dict[str, Any], path: Path) -> Dict[str, Any]:
    store = namespace["_load_auth_store"](Path(path))
    return store if isinstance(store, dict) else {}


def _save_store(namespace: Dict[str, Any], store: Dict[str, Any], path: Path) -> Path:
    local = Path(namespace["_auth_file_path"]())
    save = namespace["_save_auth_store"]
    if Path(path) == local:
        return Path(save(store))
    return Path(save(store, Path(path)))


def _token_rank(tokens: Dict[str, Any], last_refresh: Any, path: Path) -> Tuple[float, float, float]:
    refresh_rank = 0.0
    if last_refresh:
        try:
            refresh_rank = datetime.fromisoformat(
                str(last_refresh).replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            pass
    payload = _jwt_payload(str(tokens.get("access_token") or ""))
    jwt_rank = float(payload.get("iat") or payload.get("exp") or 0.0)
    try:
        mtime_rank = path.stat().st_mtime
    except OSError:
        mtime_rank = 0.0
    return refresh_rank, jwt_rank, mtime_rank


def _candidate_matches_account(tokens: Dict[str, Any], expected: Optional[str]) -> bool:
    if not expected:
        return True
    candidate = _account_identity(str(tokens.get("access_token") or ""))
    return candidate == expected


def _freshest_tokens(
    namespace: Dict[str, Any],
    supplied: Dict[str, Any],
    *,
    credential_id: Optional[str] = None,
) -> Tuple[Dict[str, str], Path]:
    expected_account = _account_identity(str(supplied.get("access_token") or ""))
    candidates = []
    paths = list(_store_paths(namespace))

    for path in paths:
        store = _load_store(namespace, path)
        if credential_id:
            pool = store.get("credential_pool")
            entries = pool.get("openai-codex") if isinstance(pool, dict) else None
            if isinstance(entries, list):
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("id") or "") != credential_id:
                        continue
                    access = str(item.get("access_token") or "").strip()
                    refresh = str(item.get("refresh_token") or "").strip()
                    if access and refresh and _candidate_matches_account(item, expected_account):
                        tokens = {"access_token": access, "refresh_token": refresh}
                        candidates.append(
                            (_token_rank(tokens, item.get("last_refresh"), path), tokens, path)
                        )
        else:
            providers = store.get("providers")
            state = providers.get("openai-codex") if isinstance(providers, dict) else None
            tokens = state.get("tokens") if isinstance(state, dict) else None
            if isinstance(tokens, dict):
                access = str(tokens.get("access_token") or "").strip()
                refresh = str(tokens.get("refresh_token") or "").strip()
                if access and refresh and _candidate_matches_account(tokens, expected_account):
                    candidates.append(
                        (_token_rank(tokens, state.get("last_refresh"), path), dict(tokens), path)
                    )

    if candidates:
        _, tokens, owner = max(candidates, key=lambda item: item[0])
        return tokens, owner
    return dict(supplied), paths[0]


def _refresh_lock_path(
    namespace: Dict[str, Any], access_token: str, refresh_token: str
) -> Path:
    paths = list(_store_paths(namespace))
    base = paths[-1]
    identity = _account_identity(access_token) or refresh_token or access_token or "missing"
    lock_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return base.parent / ".oauth-refresh-locks" / f"openai-codex-{lock_id}.lock"


@contextmanager
def _refresh_lock(namespace: Dict[str, Any], access_token: str, refresh_token: str):
    lock_path = _refresh_lock_path(namespace, access_token, refresh_token)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(lock_path.parent, 0o700)
    except OSError:
        pass

    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(str(lock_path), threading.Lock())
    with thread_lock:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows fallback
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - Windows fallback
                pass
            os.close(fd)


def _clear_error_state(entry: Dict[str, Any], tokens: Dict[str, str], last_refresh: str) -> None:
    entry["access_token"] = str(tokens.get("access_token") or "")
    entry["refresh_token"] = str(tokens.get("refresh_token") or "")
    entry["last_refresh"] = last_refresh
    for key in (
        "last_status",
        "last_status_at",
        "last_error_code",
        "last_error_reason",
        "last_error_message",
        "last_error_reset_at",
    ):
        entry[key] = None


def _commit_tokens(
    namespace: Dict[str, Any],
    owner: Path,
    tokens: Dict[str, str],
    *,
    credential_id: Optional[str] = None,
    credential_source: Optional[str] = None,
    previous_access_token: Optional[str] = None,
    last_refresh: Optional[str] = None,
) -> None:
    store = _load_store(namespace, owner)
    providers = store.get("providers")
    state = providers.get("openai-codex") if isinstance(providers, dict) else None
    state = dict(state) if isinstance(state, dict) else {}
    previous_tokens = state.get("tokens") if isinstance(state.get("tokens"), dict) else {}
    previous_singleton_access = str(previous_tokens.get("access_token") or "")
    refresh_time = (
        last_refresh
        or str(tokens.get("last_refresh") or "").strip()
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    singleton_alias = credential_id is None or credential_source == "device_code"
    if (
        credential_source == "manual:device_code"
        and previous_access_token
        and previous_singleton_access == previous_access_token
    ):
        singleton_alias = True

    pool = store.setdefault("credential_pool", {})
    if not isinstance(pool, dict):
        pool = {}
        store["credential_pool"] = pool
    entries = pool.get("openai-codex")
    entries = entries if isinstance(entries, list) else []

    if singleton_alias:
        stored_tokens = dict(tokens)
        stored_tokens.pop("last_refresh", None)
        state["tokens"] = stored_tokens
        state["last_refresh"] = refresh_time
        state["auth_mode"] = "chatgpt"
        namespace["_save_provider_state"](store, "openai-codex", state)
        for item in entries:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "")
            is_alias = source == "device_code"
            if source == "manual:device_code" and previous_singleton_access:
                is_alias = str(item.get("access_token") or "") == previous_singleton_access
            if is_alias:
                _clear_error_state(item, stored_tokens, refresh_time)

    if credential_id:
        for item in entries:
            if isinstance(item, dict) and str(item.get("id") or "") == credential_id:
                _clear_error_state(item, tokens, refresh_time)
                break

    _save_store(namespace, store, owner)


def _guarded_cli_recovery(
    namespace: Dict[str, Any],
    reason: str,
    *,
    expected_access_token: Optional[str] = None,
    owner: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    imported = namespace["_import_codex_cli_tokens"]()
    if not (
        imported
        and str(imported.get("access_token") or "").strip()
        and str(imported.get("refresh_token") or "").strip()
    ):
        return None

    expected = _account_identity(str(expected_access_token or ""))
    incoming = _account_identity(str(imported.get("access_token") or ""))
    if expected_access_token and (not expected or not incoming or expected != incoming):
        logger = namespace.get("logger")
        if logger is not None:
            logger.warning(
                "Codex CLI auth recovery refused: account identity mismatch or unavailable "
                "(reason=%s expected_account_fp=%s imported_account_fp=%s).",
                reason,
                _fingerprint(expected),
                _fingerprint(incoming),
            )
        return None

    if owner is None:
        _, owner = _freshest_tokens(namespace, imported)
    _commit_tokens(
        namespace,
        owner,
        dict(imported),
        last_refresh=imported.get("last_refresh"),
    )
    return dict(imported)


def install_auth_hardening(namespace: Dict[str, Any]) -> None:
    """Install bounded, account-safe Codex singleton refresh handling."""
    if namespace.get("_CODEX_REFRESH_HARDENING_INSTALLED"):
        return

    auth_error = namespace["AuthError"]
    logger = namespace.get("logger")

    def guarded_recovery(reason: str, *args: Any, **kwargs: Any):
        expected = kwargs.pop("expected_access_token", None)
        owner = kwargs.pop("auth_path", None)
        expected = expected or getattr(_THREAD_STATE, "expected_access_token", None)
        owner = owner or getattr(_THREAD_STATE, "owner_path", None)
        return _guarded_cli_recovery(
            namespace,
            reason,
            expected_access_token=expected,
            owner=Path(owner) if owner is not None else None,
        )

    def hardened_refresh(tokens: Dict[str, str], timeout_seconds: float) -> Dict[str, str]:
        supplied = dict(tokens)
        access = str(supplied.get("access_token") or "")
        refresh = str(supplied.get("refresh_token") or "")
        with _refresh_lock(namespace, access, refresh):
            current, owner = _freshest_tokens(namespace, supplied)
            try:
                rotated = namespace["refresh_codex_oauth_pure"](
                    str(current.get("access_token") or ""),
                    str(current.get("refresh_token") or ""),
                    timeout_seconds=timeout_seconds,
                )
            except auth_error as exc:
                if not bool(getattr(exc, "relogin_required", False)):
                    raise
                error_code = getattr(exc, "code", None) or "auth_error"
                if logger is not None:
                    logger.warning(
                        "Codex OAuth refresh rejected "
                        "(error_type=%s error_code=%s relogin_required=true "
                        "access_fp=%s refresh_fp=%s owner=%s).",
                        type(exc).__name__,
                        error_code,
                        _fingerprint(current.get("access_token")),
                        _fingerprint(current.get("refresh_token")),
                        owner,
                    )
                imported = _guarded_cli_recovery(
                    namespace,
                    f"refresh_token rejected: {error_code}",
                    expected_access_token=str(current.get("access_token") or ""),
                    owner=owner,
                )
                if imported is None:
                    raise
                return imported

            updated = dict(current)
            updated["access_token"] = str(rotated["access_token"])
            updated["refresh_token"] = str(rotated["refresh_token"])
            _commit_tokens(
                namespace,
                owner,
                updated,
                last_refresh=rotated.get("last_refresh"),
            )
            return updated

    namespace["_recover_codex_tokens_from_cli"] = guarded_recovery
    namespace["_refresh_codex_auth_tokens"] = hardened_refresh
    namespace["_CODEX_REFRESH_HARDENING_INSTALLED"] = True


def install_pool_hardening(namespace: Dict[str, Any]) -> None:
    """Serialize pool refreshes and finalize the real owner store atomically."""
    if namespace.get("_CODEX_POOL_HARDENING_INSTALLED"):
        return

    from hermes_cli import auth as auth_module

    auth_namespace = vars(auth_module)
    install_auth_hardening(auth_namespace)
    credential_pool = namespace["CredentialPool"]
    original_refresh = credential_pool._refresh_entry

    def hardened_pool_refresh(self: Any, entry: Any, *, force: bool):
        if self.provider != "openai-codex" or not entry.refresh_token:
            return original_refresh(self, entry, force=force)

        supplied = {
            "access_token": str(entry.access_token or ""),
            "refresh_token": str(entry.refresh_token or ""),
        }
        with _refresh_lock(
            auth_namespace,
            supplied["access_token"],
            supplied["refresh_token"],
        ):
            current, owner = _freshest_tokens(
                auth_namespace,
                supplied,
                credential_id=str(entry.id),
            )
            refreshed_entry = replace(
                entry,
                access_token=current["access_token"],
                refresh_token=current["refresh_token"],
            )
            _THREAD_STATE.expected_access_token = current["access_token"]
            _THREAD_STATE.owner_path = owner
            try:
                result = original_refresh(self, refreshed_entry, force=force)
            finally:
                _THREAD_STATE.expected_access_token = None
                _THREAD_STATE.owner_path = None

            if result is not None and result.access_token and result.refresh_token:
                _commit_tokens(
                    auth_namespace,
                    owner,
                    {
                        "access_token": str(result.access_token),
                        "refresh_token": str(result.refresh_token),
                    },
                    credential_id=str(result.id),
                    credential_source=str(result.source or ""),
                    previous_access_token=current["access_token"],
                    last_refresh=getattr(result, "last_refresh", None),
                )
            return result

    credential_pool._refresh_entry = hardened_pool_refresh
    namespace["_CODEX_POOL_HARDENING_INSTALLED"] = True
