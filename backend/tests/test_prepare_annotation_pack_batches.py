from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_unet_annotation_pack.py"
    spec = importlib.util.spec_from_file_location("prepare_unet_annotation_pack", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_pack_sample(root: Path, ts: str, *, labeled: bool = False) -> None:
    folder = root / ts
    folder.mkdir(parents=True, exist_ok=True)
    h, w = 24, 32
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    Image.fromarray(blank).save(folder / "quicklook_riskhint.png")
    Image.fromarray(blank).save(folder / "quicklook_blocked_overlay.png")
    Image.fromarray(np.full((h, w), 255, dtype=np.uint8)).save(folder / "quicklook_landmask.png")
    caution = np.zeros((h, w), dtype=np.uint8)
    if labeled:
        caution[2:4, 2:4] = 1
    np.save(folder / "caution_mask.npy", caution)
    (folder / "meta.json").write_text("{}", encoding="utf-8")


def test_render_riskhint_returns_expected_shape_and_blocked_dark() -> None:
    mod = _load_script_module()
    h, w = 30, 40
    x = np.zeros((7, h, w), dtype=np.float32)
    x[0] = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    x[2] = np.flipud(x[0])
    x[3] = 0.2
    x[4] = 0.4
    x[6] = np.clip(1.0 - x[0], 0.0, 1.0)
    blocked = np.zeros((h, w), dtype=np.uint8)
    blocked[:2, :] = 1
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[..., 2] = 180

    out = mod._render_riskhint(x, ["ice_conc", "ice_thick", "wave_hs", "wind_u10", "wind_v10", "bathy", "ais_heatmap"], blocked, base)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    # blocked rows should be darkened
    assert int(out[:2, :, :].mean()) < int(out[5:, :, :].mean())


def test_export_batches_has_consistent_mapping_and_file_counts(tmp_path: Path) -> None:
    mod = _load_script_module()
    out_root = tmp_path / "annotation_pack"
    _make_pack_sample(out_root, "2024-07-01_00", labeled=False)
    _make_pack_sample(out_root, "2024-07-01_06", labeled=False)
    _make_pack_sample(out_root, "2024-07-01_12", labeled=True)  # should be skipped when only_unlabeled=True
    _make_pack_sample(out_root, "2024-07-01_18", labeled=False)

    batches_root = tmp_path / "label_batches"
    summary = mod._export_label_batches(
        out_root=out_root,
        batches_root=batches_root,
        batch_size=2,
        resume_batches=False,
        only_unlabeled=True,
        max_batches=0,
    )
    assert int(summary["created_batches"]) == 2
    assert int(summary["exported"]) == 3

    batch_dirs = sorted([p for p in batches_root.iterdir() if p.is_dir() and p.name.startswith("batch_")])
    assert len(batch_dirs) == 2
    for bdir in batch_dirs:
        mapping = bdir / "mapping.csv"
        assert mapping.exists()
        with mapping.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        for row in rows:
            for col in ["riskhint_file", "landmask_file", "blocked_overlay_file"]:
                fp = bdir / row[col]
                assert fp.exists(), f"missing {fp}"

    # resume should not export already-exported timestamps
    summary2 = mod._export_label_batches(
        out_root=out_root,
        batches_root=batches_root,
        batch_size=2,
        resume_batches=True,
        only_unlabeled=True,
        max_batches=0,
    )
    assert int(summary2["created_batches"]) == 0
    assert int(summary2["exported"]) == 0

