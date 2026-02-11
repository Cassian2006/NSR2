# Render Demo Data Pack

This folder contains a minimal dataset subset for no-disk Docker deployment.

Included timestamps:

- `2024-07-15_00`
- `2024-08-15_00`
- `2024-09-15_00`
- `2024-10-15_00`

Required layout:

- `processed/annotation_pack/<timestamp>/{x_stack.npy,blocked_mask.npy,meta.json}`
- `ais_heatmap/7d/<timestamp>.npy`

In Dockerfile defaults:

- `NSR_DATA_ROOT=/app/backend/demo_data`
- `NSR_OUTPUTS_ROOT=/app/backend/demo_outputs`

For production, override these environment variables to mounted persistent paths.
