from __future__ import annotations

from pathlib import Path

import numpy as np

from app.preprocess.labelme_io import labelme_json_to_binary_mask


def test_labelme_json_to_binary_mask_polygon(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        """
{
  "imageHeight": 10,
  "imageWidth": 12,
  "shapes": [
    {"label": "caution", "shape_type": "polygon", "points": [[2,2],[9,2],[9,7],[2,7]]},
    {"label": "other", "shape_type": "polygon", "points": [[0,0],[1,0],[1,1],[0,1]]}
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    m = labelme_json_to_binary_mask(p, target_label="caution")
    assert m.shape == (10, 12)
    assert m.dtype == np.uint8
    assert int(m.sum()) > 0
    assert int(m[0, 0]) == 0


def test_labelme_json_to_binary_mask_rectangle(tmp_path: Path) -> None:
    p = tmp_path / "ann_rect.json"
    p.write_text(
        """
{
  "imageHeight": 8,
  "imageWidth": 8,
  "shapes": [
    {"label": "caution", "shape_type": "rectangle", "points": [[1,1],[4,4]]}
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    m = labelme_json_to_binary_mask(p)
    assert m.shape == (8, 8)
    assert int(m[2, 2]) == 1
