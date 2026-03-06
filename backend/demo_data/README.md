# Render Demo Data Pack

This folder contains a minimal dataset subset bundled with the repo for smoke tests and demo fallback.

Included timestamps:

- `2024-07-15_00`
- `2024-08-15_00`
- `2024-09-15_00`
- `2024-10-15_00`

Required layout:

- `processed/annotation_pack/<timestamp>/{x_stack.npy,blocked_mask.npy,meta.json}`
- `ais_heatmap/7d/<timestamp>.npy`

Current Dockerfile defaults are production-oriented:

- `NSR_DATA_ROOT=/app/data`
- `NSR_OUTPUTS_ROOT=/app/outputs`
- `NSR_ALLOW_DEMO_FALLBACK=0`

If you want the app to auto-fallback to this bundled demo subset when runtime data roots are empty, set:

- `NSR_ALLOW_DEMO_FALLBACK=1`

If you want to point directly at the demo subset, set explicit overrides such as:

- `NSR_DATA_ROOT=/app/backend/demo_data`
- `NSR_OUTPUTS_ROOT=/app/backend/demo_outputs`
