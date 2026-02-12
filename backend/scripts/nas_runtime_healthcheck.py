from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str  # "critical" | "warning"
    detail: str
    data: dict[str, Any]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_base_url(base_url: str) -> str:
    out = base_url.strip().rstrip("/")
    if not out:
        raise ValueError("base_url is empty")
    return out


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _request(
    *,
    url: str,
    timeout_sec: float,
    retries: int,
) -> tuple[int, bytes, dict[str, str]]:
    last_err: Exception | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            req = Request(url=url, method="GET")
            with urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read()
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return int(resp.status), body, headers
        except (HTTPError, URLError, TimeoutError) as exc:
            last_err = exc
            if attempt >= attempts:
                break
            time.sleep(0.25 * attempt)
    raise RuntimeError(f"request failed after {attempts} attempts: {url}; err={last_err}")


def _get_json(
    *,
    base_url: str,
    path: str,
    timeout_sec: float,
    retries: int,
) -> tuple[int, dict[str, Any]]:
    status, body, headers = _request(
        url=f"{base_url}{path}",
        timeout_sec=timeout_sec,
        retries=retries,
    )
    ctype = headers.get("content-type", "")
    if "application/json" not in ctype:
        raise RuntimeError(f"unexpected content-type for {path}: {ctype}")
    return status, json.loads(body.decode("utf-8"))


def _sample_indices(total: int, max_count: int) -> list[int]:
    if total <= 0:
        return []
    if total <= max_count:
        return list(range(total))
    # Prefer recent timestamps while still sampling historical points.
    picks = {0, total - 1}
    step = (total - 1) / float(max_count - 1)
    for i in range(max_count):
        picks.add(int(round(i * step)))
    return sorted(picks)


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = report.get("summary", {})
    lines.append("# NAS Runtime Healthcheck")
    lines.append("")
    lines.append(f"- status: `{summary.get('status', 'unknown')}`")
    lines.append(f"- checked_at: `{summary.get('checked_at', '')}`")
    lines.append(f"- base_url: `{summary.get('base_url', '')}`")
    lines.append(f"- sample_count: `{summary.get('sample_count', 0)}`")
    lines.append(f"- timestamps: `{summary.get('timestamp_count', 0)}`")
    lines.append(f"- selected_timestamp: `{summary.get('selected_timestamp', '-')}`")
    lines.append("")
    lines.append("## Checks")
    for item in report.get("checks", []):
        mark = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"- [{mark}] `{item.get('name')}` ({item.get('severity')})")
        lines.append(f"  - detail: `{item.get('detail', '')}`")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS startup runtime healthcheck for NSR2 service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--required-layers", default="bathy,ais_heatmap,unet_pred,unet_uncertainty")
    parser.add_argument("--optional-layers", default="ice,wave,wind")
    parser.add_argument("--min-sample-count", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--probe-count", type=int, default=12)
    parser.add_argument("--overlay-min-bytes", type=int, default=800)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--strict-optional", action="store_true")
    parser.add_argument(
        "--warn-exit-zero",
        action="store_true",
        help="Return exit code 0 when final status is warn (useful for container healthchecks).",
    )
    args = parser.parse_args()

    base_url = _normalize_base_url(args.base_url)
    required_layers = _split_csv(args.required_layers)
    optional_layers = _split_csv(args.optional_layers)

    checks: list[CheckResult] = []
    selected_ts = ""
    sample_count = 0
    timestamp_count = 0
    status = "pass"

    # 1) Health check
    try:
        code, payload = _get_json(base_url=base_url, path="/healthz", timeout_sec=args.timeout_sec, retries=args.retries)
        ok = code == 200 and str(payload.get("status", "")).lower() == "ok"
        checks.append(CheckResult("healthz", ok, "critical", f"http={code}, status={payload.get('status')}", payload))
    except Exception as exc:
        checks.append(CheckResult("healthz", False, "critical", str(exc), {}))

    # 2) Dataset summary
    datasets_payload: dict[str, Any] = {}
    try:
        code, payload = _get_json(base_url=base_url, path="/v1/datasets", timeout_sec=args.timeout_sec, retries=args.retries)
        datasets_payload = payload.get("dataset", {}) if isinstance(payload, dict) else {}
        sample_count = int(datasets_payload.get("sample_count", 0))
        ok = code == 200 and sample_count >= int(args.min_sample_count)
        checks.append(
            CheckResult(
                "datasets",
                ok,
                "critical",
                f"http={code}, sample_count={sample_count}, min_required={args.min_sample_count}",
                {"dataset": datasets_payload},
            )
        )
    except Exception as exc:
        checks.append(CheckResult("datasets", False, "critical", str(exc), {}))

    # 3) Timestamps
    timestamps: list[str] = []
    try:
        code, payload = _get_json(base_url=base_url, path="/v1/timestamps", timeout_sec=args.timeout_sec, retries=args.retries)
        timestamps = payload.get("timestamps", []) if isinstance(payload, dict) else []
        timestamp_count = len(timestamps)
        ok = code == 200 and timestamp_count > 0
        checks.append(CheckResult("timestamps", ok, "critical", f"http={code}, count={timestamp_count}", {"timestamps_preview": timestamps[:10]}))
    except Exception as exc:
        checks.append(CheckResult("timestamps", False, "critical", str(exc), {}))

    # 4) Layer availability probe
    layer_probe: dict[str, Any] = {}
    if timestamps:
        candidates = [args.timestamp] if args.timestamp.strip() else [timestamps[i] for i in _sample_indices(len(timestamps), max(2, int(args.probe_count)))]
        best_required = -1
        best_optional = -1
        for ts in candidates:
            try:
                query = urlencode({"timestamp": ts})
                code, payload = _get_json(
                    base_url=base_url,
                    path=f"/v1/layers?{query}",
                    timeout_sec=args.timeout_sec,
                    retries=args.retries,
                )
                if code != 200:
                    continue
                layers = payload.get("layers", [])
                avail = {str(item.get("id")): bool(item.get("available")) for item in layers if isinstance(item, dict)}
                required_ok = sum(1 for layer in required_layers if avail.get(layer, False))
                optional_ok = sum(1 for layer in optional_layers if avail.get(layer, False))
                if required_ok > best_required or (required_ok == best_required and optional_ok > best_optional):
                    best_required = required_ok
                    best_optional = optional_ok
                    selected_ts = ts
                    layer_probe = {"timestamp": ts, "availability": avail}
            except Exception:
                continue

        if not selected_ts:
            checks.append(CheckResult("layers_probe", False, "critical", "failed to fetch /v1/layers for all candidates", {}))
        else:
            avail = layer_probe.get("availability", {})
            missing_required = [layer for layer in required_layers if not avail.get(layer, False)]
            missing_optional = [layer for layer in optional_layers if not avail.get(layer, False)]
            ok_required = not missing_required
            severity = "critical"
            detail = f"selected={selected_ts}, missing_required={missing_required}, missing_optional={missing_optional}"
            checks.append(CheckResult("layers_probe", ok_required, severity, detail, layer_probe))

            optional_ok = not missing_optional
            optional_severity = "critical" if args.strict_optional else "warning"
            checks.append(
                CheckResult(
                    "layers_optional",
                    optional_ok,
                    optional_severity,
                    f"missing_optional={missing_optional}",
                    {"missing_optional": missing_optional},
                )
            )
    else:
        checks.append(CheckResult("layers_probe", False, "critical", "skip: no timestamps", {}))

    # 5) Overlay render smoke test
    if selected_ts:
        avail = layer_probe.get("availability", {})
        overlay_layers = [layer for layer in required_layers if avail.get(layer, False)]
        for layer in overlay_layers:
            try:
                query = urlencode(
                    {
                        "timestamp": selected_ts,
                        "bbox": "20,60,180,80",
                        "size": "512,256",
                    }
                )
                status_code, body, headers = _request(
                    url=f"{base_url}/v1/overlay/{layer}.png?{query}",
                    timeout_sec=args.timeout_sec,
                    retries=args.retries,
                )
                ctype = headers.get("content-type", "")
                ok = status_code == 200 and "image/png" in ctype and len(body) >= int(args.overlay_min_bytes)
                checks.append(
                    CheckResult(
                        f"overlay_{layer}",
                        ok,
                        "critical",
                        f"http={status_code}, bytes={len(body)}, content_type={ctype}",
                        {"bytes": len(body), "content_type": ctype},
                    )
                )
            except Exception as exc:
                checks.append(CheckResult(f"overlay_{layer}", False, "critical", str(exc), {}))

    # Aggregate status
    for item in checks:
        if item.severity == "critical" and not item.ok:
            status = "fail"
            break
    if status != "fail":
        has_warning = any((not item.ok) and item.severity == "warning" for item in checks)
        if has_warning:
            status = "warn"

    report = {
        "summary": {
            "status": status,
            "checked_at": _now_utc().isoformat(),
            "base_url": base_url,
            "sample_count": sample_count,
            "timestamp_count": timestamp_count,
            "selected_timestamp": selected_ts,
        },
        "checks": [asdict(item) for item in checks],
    }

    settings = get_settings()
    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_utc().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"nas_runtime_healthcheck_{stamp}.json"
    md_path = out_dir / f"nas_runtime_healthcheck_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")

    print(f"status={status}")
    print(f"json={json_path}")
    print(f"md={md_path}")
    if selected_ts:
        print(f"selected_timestamp={selected_ts}")
    print(f"sample_count={sample_count}")

    if status == "fail":
        raise SystemExit(2)
    if status == "warn":
        if args.warn_exit_zero:
            raise SystemExit(0)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
