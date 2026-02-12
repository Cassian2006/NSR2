from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(name: str) -> str:
    trimmed = name.strip()
    if not trimmed or "/" in trimmed or "\\" in trimmed or ".." in trimmed:
        raise ValueError("invalid run_id")
    return trimmed


@dataclass(frozen=True)
class ActiveRun:
    run_id: str
    path: Path
    summary: dict[str, Any]


class ActiveReviewService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.root = self.settings.outputs_root / "active_learning"
        self.state_root = self.root / "review_state"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _iter_runs(self) -> list[ActiveRun]:
        runs: list[ActiveRun] = []
        if not self.root.exists():
            return runs
        for folder in self.root.iterdir():
            if not folder.is_dir() or not folder.name.startswith("active_"):
                continue
            summary_path = folder / "summary.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
            runs.append(ActiveRun(run_id=folder.name, path=folder, summary=summary))
        runs.sort(key=lambda r: r.path.stat().st_mtime, reverse=True)
        return runs

    def _state_path(self, run_id: str) -> Path:
        run_id = _safe_name(run_id)
        return self.state_root / f"{run_id}.json"

    def _load_state(self, run_id: str) -> dict[str, Any]:
        path = self._state_path(run_id)
        if not path.exists():
            return {"run_id": run_id, "updated_at": "", "decisions": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            decisions = data.get("decisions", {})
            if not isinstance(decisions, dict):
                decisions = {}
            return {
                "run_id": run_id,
                "updated_at": str(data.get("updated_at", "")),
                "decisions": decisions,
            }
        except Exception:
            return {"run_id": run_id, "updated_at": "", "decisions": {}}

    def _save_state(self, run_id: str, decisions: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "run_id": run_id,
            "updated_at": _utc_now_iso(),
            "decisions": decisions,
        }
        path = self._state_path(run_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _resolve_run(self, run_id: str | None = None) -> ActiveRun | None:
        runs = self._iter_runs()
        if not runs:
            return None
        if not run_id:
            return runs[0]
        safe_id = _safe_name(run_id)
        for run in runs:
            if run.run_id == safe_id:
                return run
        return None

    def _mapping_rows(self, run: ActiveRun) -> list[dict[str, str]]:
        mapping_path = run.path / "labelme_active_topk" / "mapping.csv"
        if not mapping_path.exists():
            maybe = run.summary.get("labelme_dir")
            if isinstance(maybe, str) and maybe.strip():
                mapping_path = Path(maybe) / "mapping.csv"
        if not mapping_path.exists():
            return []
        with mapping_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def list_runs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for run in self._iter_runs():
            rows = self._mapping_rows(run)
            state = self._load_state(run.run_id)
            decisions = state.get("decisions", {})
            accepted = 0
            needs_revision = 0
            for item in decisions.values() if isinstance(decisions, dict) else []:
                if not isinstance(item, dict):
                    continue
                decision = str(item.get("decision", ""))
                if decision == "accepted":
                    accepted += 1
                elif decision == "needs_revision":
                    needs_revision += 1
            out.append(
                {
                    "run_id": run.run_id,
                    "created_at": datetime.fromtimestamp(run.path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "candidate_count": int(run.summary.get("candidate_count", 0)),
                    "top_k": int(run.summary.get("top_k", len(rows))),
                    "mapping_count": len(rows),
                    "accepted_count": accepted,
                    "needs_revision_count": needs_revision,
                    "summary_file": str(run.path / "summary.json"),
                }
            )
        return out

    def get_items(self, run_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        run = self._resolve_run(run_id)
        if run is None:
            return {"run_id": "", "items": [], "count": 0}
        rows = self._mapping_rows(run)
        state = self._load_state(run.run_id)
        decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
        explain_dir = run.path / "explanations"

        items: list[dict[str, Any]] = []
        for row in rows[: max(1, int(limit))]:
            filename = str(row.get("filename", ""))
            file_base = Path(filename).stem
            explain_json = row.get("explain_json") or str(explain_dir / f"{file_base}_explain.json")
            explain_png = row.get("explain_png") or str(explain_dir / f"{file_base}_explain.png")
            explanation: dict[str, Any] = {}
            try:
                p = Path(str(explain_json))
                if p.exists():
                    explanation = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                explanation = {}
            ts = str(row.get("timestamp", ""))
            decision = decisions.get(ts, {}) if isinstance(decisions, dict) else {}
            items.append(
                {
                    "rank": int(row.get("rank", len(items) + 1)),
                    "timestamp": ts,
                    "score": float(row.get("score", 0.0)),
                    "uncertainty_score": float(row.get("uncertainty_score", 0.0)),
                    "route_impact_score": float(row.get("route_impact_score", 0.0)),
                    "class_balance_score": float(row.get("class_balance_score", 0.0)),
                    "pred_caution_ratio": float(row.get("pred_caution_ratio", 0.0)),
                    "dominant_factor": str(row.get("dominant_factor", explanation.get("dominant_factor", ""))),
                    "explain_json": str(explain_json),
                    "explain_png": str(explain_png),
                    "explanation": explanation,
                    "decision": decision if isinstance(decision, dict) else {},
                }
            )
        return {"run_id": run.run_id, "items": items, "count": len(rows)}

    def save_decision(self, *, run_id: str, timestamp: str, decision: str, note: str = "") -> dict[str, Any]:
        run = self._resolve_run(run_id)
        if run is None:
            raise FileNotFoundError(f"run not found: {run_id}")
        decision = decision.strip().lower()
        if decision not in {"accepted", "needs_revision"}:
            raise ValueError("decision must be one of: accepted, needs_revision")
        ts = timestamp.strip()
        if not ts:
            raise ValueError("timestamp is required")
        state = self._load_state(run.run_id)
        decisions = state.get("decisions", {})
        if not isinstance(decisions, dict):
            decisions = {}
        decisions[ts] = {
            "decision": decision,
            "note": note.strip(),
            "updated_at": _utc_now_iso(),
        }
        payload = self._save_state(run.run_id, decisions)
        return {
            "ok": True,
            "run_id": run.run_id,
            "timestamp": ts,
            "decision": decision,
            "state_file": str(self._state_path(run.run_id)),
            "updated_at": payload["updated_at"],
        }

