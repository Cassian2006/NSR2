from __future__ import annotations

import pytest

import app.model.infer as infer_mod


def test_ensure_torch_available_raises_clean_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(infer_mod, "torch", None)
    monkeypatch.setattr(infer_mod, "_torch_import_error", ModuleNotFoundError("No module named 'torch'"))
    with pytest.raises(infer_mod.InferenceError, match="PyTorch is required for on-demand inference"):
        infer_mod._ensure_torch_available()
