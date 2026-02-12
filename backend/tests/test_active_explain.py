from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.model.active_explain import explain_sample, render_explanation_card


def test_explain_sample_fields_nonnegative_and_normalized() -> None:
    h, w = 32, 48
    x = np.zeros((7, h, w), dtype=np.float32)
    # ice
    x[0] = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    # wave
    x[2] = np.flipud(x[0])
    # wind u/v
    x[3] = 0.2
    x[4] = 0.5
    # ais heatmap
    x[6] = np.clip(1.0 - x[0], 0.0, 1.0)

    blocked = np.zeros((h, w), dtype=np.uint8)
    blocked[:2, :] = 1
    caution_prob = np.clip(0.2 + 0.6 * x[0], 0.0, 1.0)
    entropy = np.clip(0.1 + 0.3 * (1.0 - np.abs(caution_prob - 0.5) * 2.0), 0.0, None)
    suggested = (caution_prob > 0.6).astype(np.uint8)
    channel_names = ["ice_conc", "ice_thick", "wave_hs", "wind_u10", "wind_v10", "bathy", "ais_heatmap"]

    exp = explain_sample(
        x_stack=x,
        blocked_mask=blocked,
        caution_prob=caution_prob,
        entropy=entropy,
        channel_names=channel_names,
        suggested_mask=suggested,
    )

    required_top = {"version", "factors_raw", "factors_norm", "dominant_factor", "stats", "channels_used", "notes"}
    assert required_top.issubset(set(exp.keys()))

    raw = exp["factors_raw"]
    norm = exp["factors_norm"]
    expected_keys = {
        "ice_contribution",
        "wave_contribution",
        "wind_contribution",
        "ais_deviation",
        "historical_misclassification_risk",
    }
    assert expected_keys == set(raw.keys())
    assert expected_keys == set(norm.keys())
    assert all(float(v) >= 0.0 for v in raw.values())
    assert all(float(v) >= 0.0 for v in norm.values())
    assert abs(sum(float(v) for v in norm.values()) - 1.0) < 1e-6
    assert exp["dominant_factor"] in expected_keys


def test_render_explanation_card_outputs_readable_png(tmp_path: Path) -> None:
    exp = {
        "factors_norm": {
            "ice_contribution": 0.25,
            "wave_contribution": 0.20,
            "wind_contribution": 0.15,
            "ais_deviation": 0.30,
            "historical_misclassification_risk": 0.10,
        },
        "dominant_factor": "ais_deviation",
        "stats": {"entropy_mean": 0.33, "entropy_p95": 0.61, "suggested_ratio": 0.07},
    }
    out = tmp_path / "exp.png"
    render_explanation_card(exp, out, title="unit-test")
    assert out.exists()
    img = Image.open(out)
    assert img.size[0] > 100 and img.size[1] > 100

