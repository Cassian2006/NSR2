from __future__ import annotations

from pathlib import Path

import numpy as np


def run_stub_unet_inference(output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.zeros((128, 128), dtype=np.uint8)
    np.save(output_path, pred)
    return {
        "shape": list(pred.shape),
        "class_hist": {"safe": int((pred == 0).sum()), "caution": int((pred == 1).sum()), "blocked": int((pred == 2).sum())},
    }

