from __future__ import annotations

import calendar
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


MONTHS = ["202407", "202408", "202409", "202410"]
EXPECTED_FILES = {"meta.json", "y_corridor.npy", "y_distance.npy", "y_prox.npy"}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def expected_steps(month: str) -> int:
    year = int(month[:4])
    mon = int(month[4:])
    days = calendar.monthrange(year, mon)[1]
    return days * 4


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # minus header
        return max(0, sum(1 for _ in f) - 1)


def month_from_stem(stem: str) -> str:
    # YYYY-MM-DD_HH -> YYYYMM
    return f"{stem[:4]}{stem[5:7]}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    outputs_root = project_root / "outputs" / "qa"
    outputs_root.mkdir(parents=True, exist_ok=True)

    checks: list[CheckResult] = []
    summary: dict[str, object] = {}

    # 1) Core folders
    for rel in [
        "data/raw/ais",
        "data/raw/env_nc/bathy",
        "data/raw/env_nc/ice_conc/2024",
        "data/raw/env_nc/wave_hs/2024",
        "data/raw/env_nc/wind_10m/2024",
        "data/processed/ais_cleaned",
        "data/processed/samples",
        "data/ais_heatmap",
    ]:
        p = project_root / rel
        checks.append(CheckResult(rel, p.exists(), "exists" if p.exists() else "missing"))

    # 2) AIS cleaned coverage
    ais_rows: dict[str, int] = {}
    for m in MONTHS:
        csv_path = data_root / "processed" / "ais_cleaned" / f"{m}_clean.csv"
        ok = csv_path.exists()
        rows = count_csv_rows(csv_path) if ok else 0
        ais_rows[m] = rows
        checks.append(CheckResult(f"ais_cleaned_{m}", ok and rows > 0, f"rows={rows}"))
    summary["ais_cleaned_rows"] = ais_rows

    # 3) Sample coverage and file completeness
    sample_stats: dict[str, dict[str, int]] = {}
    for m in MONTHS:
        folder = data_root / "processed" / "samples" / m
        dirs = sorted([d for d in folder.glob("2024-*-*_??") if d.is_dir()]) if folder.exists() else []
        expected = expected_steps(m)
        complete_dirs = 0
        shapes = set()
        for d in dirs:
            files = {p.name for p in d.glob("*")}
            if EXPECTED_FILES.issubset(files):
                complete_dirs += 1
                arr = np.load(d / "y_corridor.npy", mmap_mode="r")
                shapes.add(tuple(arr.shape))
        ok = len(dirs) >= expected and complete_dirs == len(dirs) and len(shapes) == 1
        sample_stats[m] = {
            "expected_steps": expected,
            "actual_dirs": len(dirs),
            "complete_dirs": complete_dirs,
            "unique_shapes": len(shapes),
        }
        checks.append(
            CheckResult(
                f"samples_{m}",
                ok,
                f"expected={expected}, actual={len(dirs)}, complete={complete_dirs}, shapes={len(shapes)}",
            )
        )
    summary["sample_stats"] = sample_stats

    # 4) Heatmap coverage and quality
    heatmap_root = data_root / "ais_heatmap"
    heatmap_tags = [d for d in heatmap_root.iterdir() if d.is_dir()] if heatmap_root.exists() else []
    heatmap_report: dict[str, dict[str, object]] = {}
    for tag_dir in heatmap_tags:
        files = sorted(tag_dir.glob("*.npy"))
        by_month = {m: 0 for m in MONTHS}
        has_nan = False
        min_v = 1.0
        max_v = 0.0
        shape_set = set()
        for p in files:
            m = month_from_stem(p.stem)
            if m in by_month:
                by_month[m] += 1
            arr = np.load(p, mmap_mode="r")
            shape_set.add(tuple(arr.shape))
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            min_v = min(min_v, arr_min)
            max_v = max(max_v, arr_max)
            if np.isnan(arr).any():
                has_nan = True
        heatmap_report[tag_dir.name] = {
            "count": len(files),
            "by_month": by_month,
            "has_nan": has_nan,
            "min": min_v if files else None,
            "max": max_v if files else None,
            "unique_shapes": len(shape_set),
        }
        checks.append(
            CheckResult(
                f"heatmap_tag_{tag_dir.name}",
                len(files) > 0 and not has_nan and len(shape_set) == 1,
                f"count={len(files)}, nan={has_nan}, shapes={len(shape_set)}, range=[{min_v:.4f},{max_v:.4f}]",
            )
        )
    summary["heatmap"] = heatmap_report

    # 5) Standardized folders from README
    std_folders = ["data/env", "data/bathy", "data/ais_heatmap", "data/unet_pred"]
    for rel in std_folders:
        p = project_root / rel
        checks.append(CheckResult(f"std_{rel}", p.exists(), "exists" if p.exists() else "missing"))

    all_ok = all(c.ok for c in checks)
    summary["all_ok"] = all_ok
    summary["checks"] = [asdict(c) for c in checks]

    json_path = outputs_root / "data_audit.json"
    md_path = outputs_root / "data_audit.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Data Audit", "", f"overall_ok: {all_ok}", ""]
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        lines.append(f"- [{mark}] {c.name}: {c.detail}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"overall_ok={all_ok}")


if __name__ == "__main__":
    main()

