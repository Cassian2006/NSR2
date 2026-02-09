# Data Directory Policy

This directory is intentionally ignored by git for large local datasets.

Expected local layout:

- `raw/` source downloads
- `processed/` cleaned/intermediate artifacts
- `ais_heatmap/<window>/YYYY-MM-DD_HH.npy` regenerated heatmaps
- `env/`, `bathy/`, `unet_pred/` (optional standardized layout)

To regenerate AIS heatmaps from local cleaned AIS CSV files:

```powershell
cd backend
python scripts/generate_ais_heatmap.py --month 202407 --window-hours 168 --step-hours 6
```
