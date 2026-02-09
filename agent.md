# NSR2 Agent Operating Rules

Last updated: 2026-02-09T21:45:22+00:00

## Fixed Principles

1. Before every GitHub push/submit, run a review and `pytest` at least once.
2. GitHub repository: `https://github.com/Cassian2006/NSR2.git`.
3. Keep this `agent.md` updated with every operation, each with a timestamp.
4. After successfully implementing a meaningful part, do review + Git commit.
5. Prefer more tests and robustness checks.
6. `readme.md` is the project strategy/structure source of truth and may be questioned, updated, and improved during development.

## Commit Gate Checklist

- [ ] Code/path changes reviewed
- [ ] `pytest` executed and result recorded
- [ ] Key behavior manually verified when needed
- [ ] Operation log appended in this file

## Operation Log

- 2026-02-09T20:08:06+00:00 | Initialized `agent.md` rules and process.
- 2026-02-09T20:08:06+00:00 | Captured workspace snapshot (`backend`, `frontend`, `data`, `outputs`, `readme.md` present).
- 2026-02-09T20:08:22+00:00 | Initialized git repository in `NSR2`.
- 2026-02-09T20:08:22+00:00 | Added remote `origin` -> `https://github.com/Cassian2006/NSR2.git`.
- 2026-02-09T20:10:27+00:00 | Added backend smoke tests in `backend/tests/test_api_smoke.py`.
- 2026-02-09T20:10:27+00:00 | Pre-submit review: `python -m compileall app` (pass), `npm run build` in `frontend` (pass).
- 2026-02-09T20:10:27+00:00 | Pre-submit test: `python -m pytest -q` in `backend` -> `5 passed`.
- 2026-02-09T20:11:02+00:00 | Staged initial project files for first commit (excluding `data/`).
- 2026-02-09T20:11:35+00:00 | Configured local git identity: `user.name=Cassian2006`, `user.email=Cassian2006@users.noreply.github.com`.
- 2026-02-09T20:11:35+00:00 | Created commit `03e9505` with backend/frontend skeleton and tests.
- 2026-02-09T20:11:35+00:00 | Pushed `main` to `origin` (`https://github.com/Cassian2006/NSR2.git`).
- 2026-02-09T20:11:51+00:00 | Pre-submit test before log commit: `python -m pytest -q` in `backend` -> `5 passed`.
- 2026-02-09T20:21:40+00:00 | Updated `.gitignore` to ignore all local dataset files under `data/**` while keeping `data/README.md` and `data/.gitkeep`.
- 2026-02-09T20:21:40+00:00 | Removed external legacy heatmap dependency in backend; `ais_heatmap` layer now resolves from local `data/ais_heatmap`.
- 2026-02-09T20:21:40+00:00 | Added local AIS heatmap generator: `backend/scripts/generate_ais_heatmap.py` + preprocessing module + tests.
- 2026-02-09T20:21:40+00:00 | Validation: `python -m compileall app scripts` (pass), `python -m pytest -q` (9 passed).
- 2026-02-09T20:21:40+00:00 | Runtime check: generated month heatmaps with `--month 202407 --step-hours 24 --tag smoke24h` -> 31 files.
- 2026-02-09T20:22:20+00:00 | Final pre-commit validation rerun: `python -m compileall app scripts` (pass), `python -m pytest -q` (9 passed).
- 2026-02-09T20:22:53+00:00 | Committed `4c4d9af`: ignore local data + add local AIS heatmap generator and tests.
- 2026-02-09T20:22:53+00:00 | Pushed `4c4d9af` to `origin/main`.
- 2026-02-09T20:23:11+00:00 | Pre-submit test before log sync commit: `python -m pytest -q` in `backend` -> `9 passed`.
- 2026-02-09T20:42:19+00:00 | Added QA scripts: `backend/scripts/visualize_heatmap.py` and `backend/scripts/audit_data_resources.py`.
- 2026-02-09T20:42:19+00:00 | Fixed AIS heatmap generation robustness: sort events by `postime` before rolling-window accumulation.
- 2026-02-09T20:42:19+00:00 | Re-generated local heatmaps: `7d` for `202407-202410`, and refreshed QA previews/reports under `outputs/qa`.
- 2026-02-09T20:42:19+00:00 | Validation: `python -m compileall app scripts` (pass), `python -m pytest -q` -> `10 passed`.
- 2026-02-09T20:42:47+00:00 | Committed and pushed `53617d4`: heatmap QA scripts + unsorted AIS generation fix.
- 2026-02-09T20:56:27+00:00 | Root-cause check: source env vars (ice_conc/ice_thick/wave_hs) in NSRcorridorNA only cover to 2024-10-31_00; 2024-10-31_06/12/18 absent at source intersection.
- 2026-02-09T20:56:27+00:00 | Built patch env tag NSRcorridorNA/data/interim/env_grids/202410_patch_fill and filled 2024-10-31_06/12/18 by persistence copy from 2024-10-31_00 (documented in env_build_report patch_note).
- 2026-02-09T20:56:27+00:00 | Ran src/data/align_windows.py with NSR_OUT_TAG=202410_patch_fill, NSR_WINDOW_HOURS=6; generated aligned windows including 2024-10-31_06/12/18.
- 2026-02-09T20:56:27+00:00 | Ran src/data/rasterize_corridor.py with NSR_DILATE=1, NSR_SIGMA=1.0, NSR_SKELETON=0, NSR_DISTANCE=1, NSR_PROX=1; generated y_corridor/y_distance/y_prox for patch tag.
- 2026-02-09T20:56:27+00:00 | Copied patched sample dirs 2024-10-31_06, 2024-10-31_12, 2024-10-31_18 into NSR2/data/processed/samples/202410.
- 2026-02-09T20:56:27+00:00 | Verification: samples_202410 missing count -> 0/124; generated QA visualization outputs/qa/heatmap_oct31_18_compare.png and reran backend/scripts/audit_data_resources.py.
- 2026-02-09T20:57:05+00:00 | Pre-submit validation: python -m pytest -q in backend -> 10 passed.
- 2026-02-09T20:57:36+00:00 | Committed d45d523: docs: log Oct-31 sample backfill and validation.
- 2026-02-09T20:57:36+00:00 | Pushed d45d523 to origin/main.
- 2026-02-09T20:57:48+00:00 | Pre-submit validation before log-sync push: python -m pytest -q in backend -> 10 passed.
- 2026-02-09T21:18:51+00:00 | Generated NSR2 env grids from raw nc via NSRcorridorNA make_env_grids.py; patched missing 2024-10-31_06/12/18 by persistence copy from 2024-10-31_00; env total=492.
- 2026-02-09T21:18:51+00:00 | Added U-Net data prep module: backend/app/preprocess/unet_dataset.py (stacking, blocked mask, label merge, quicklook helpers).
- 2026-02-09T21:18:51+00:00 | Added scripts: backend/scripts/prepare_unet_annotation_pack.py and backend/scripts/build_unet_labels.py.
- 2026-02-09T21:18:51+00:00 | Built full annotation pack (202407-202410): data/processed/annotation_pack (492 samples).
- 2026-02-09T21:18:51+00:00 | Built training manifest: data/processed/unet_manifest.csv (train=368, val=124 with val_month=202410).
- 2026-02-09T21:18:51+00:00 | Added tests: backend/tests/test_unet_dataset_prep.py; validation: python -m pytest -q -> 14 passed.
- 2026-02-09T21:19:26+00:00 | Pre-submit validation before push: python -m pytest -q in backend -> 14 passed.
- 2026-02-09T21:19:26+00:00 | Committed 4b860b7: feat: prepare unet annotation pack and label manifest.
- 2026-02-09T21:19:26+00:00 | Pushed 4b860b7 to origin/main.
- 2026-02-09T21:19:39+00:00 | Pre-submit validation before log-sync push: python -m pytest -q in backend -> 14 passed.
- 2026-02-09T21:44:33+00:00 | Confirmed Labelme available: labelme --version -> 5.11.2; launched Labelme on annotation sample quicklook with label preset caution.
- 2026-02-09T21:44:33+00:00 | Added labelme conversion helper: backend/app/preprocess/labelme_io.py and backend/scripts/import_labelme_caution.py.
- 2026-02-09T21:44:33+00:00 | Added tests for labelme conversion: backend/tests/test_labelme_io.py; validation: python -m pytest -q -> 16 passed.
- 2026-02-09T21:45:12+00:00 | Pre-submit validation before push: python -m pytest -q in backend -> 16 passed.
- 2026-02-09T21:45:12+00:00 | Committed c06b6a5: feat: add labelme-to-caution import pipeline.
- 2026-02-09T21:45:12+00:00 | Pushed c06b6a5 to origin/main.
- 2026-02-09T21:45:22+00:00 | Pre-submit validation before log-sync push: python -m pytest -q in backend -> 16 passed.
