from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock


_LOCK = Lock()
_STATE: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_state(progress_id: str, message: str) -> dict:
    return {
        "progress_id": progress_id,
        "status": "running",
        "phase": "init",
        "message": message,
        "percent": 1,
        "error": None,
        "updated_at": _now_iso(),
    }


def start_progress(progress_id: str, *, message: str = "开始最新预测流程") -> dict:
    payload = _new_state(progress_id, message)
    with _LOCK:
        _STATE[progress_id] = payload
    return payload


def update_progress(
    progress_id: str,
    *,
    phase: str | None = None,
    message: str | None = None,
    percent: int | None = None,
) -> dict:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            cur = _new_state(progress_id, "开始最新预测流程")
            _STATE[progress_id] = cur

        if phase is not None:
            cur["phase"] = phase
        if message is not None:
            cur["message"] = message
        if percent is not None:
            cur["percent"] = max(0, min(100, int(percent)))
        cur["updated_at"] = _now_iso()
        return dict(cur)


def complete_progress(progress_id: str, *, message: str = "已完成") -> dict:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            cur = _new_state(progress_id, "开始最新预测流程")
            _STATE[progress_id] = cur
        cur["status"] = "completed"
        cur["phase"] = "done"
        cur["message"] = message
        cur["percent"] = 100
        cur["error"] = None
        cur["updated_at"] = _now_iso()
        return dict(cur)


def fail_progress(progress_id: str, *, error: str, phase: str = "error") -> dict:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            cur = _new_state(progress_id, "开始最新预测流程")
            _STATE[progress_id] = cur
        cur["status"] = "failed"
        cur["phase"] = phase
        cur["message"] = "最新预测流程失败"
        cur["error"] = error
        cur["updated_at"] = _now_iso()
        # Keep current percent; if missing set a sensible floor.
        if "percent" not in cur:
            cur["percent"] = 1
        return dict(cur)


def get_progress(progress_id: str) -> dict:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            return {
                "progress_id": progress_id,
                "exists": False,
                "status": "not_found",
                "phase": "unknown",
                "message": "未找到进度记录",
                "percent": 0,
                "error": None,
                "updated_at": _now_iso(),
            }
        payload = dict(cur)
        payload["exists"] = True
        return payload
