# NSR2 Agent Operating Rules


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
- 2026-02-09T21:46:46+00:00 | Clarified labeling policy (blocked from bathy/land, caution manual, else safe) and generated balanced shortlist: data/processed/annotation_pack/SHORTLIST_50.txt (50 timestamps).
- 2026-02-09T21:54:42+00:00 | Verified LabelImg and Labelme binaries; launched LabelImg for annotation_pack directory test (bbox workflow).
- 2026-02-09T21:54:42+00:00 | Generated blocked-mask overlay previews for shortlist-50: quicklook_blocked_overlay.png in each timestamp folder.
- 2026-02-09T21:54:42+00:00 | Launched Labelme on overlay image (2024-10-31_18) with label preset caution for polygon-based caution annotation.
- 2026-02-09T21:57:47+00:00 | Created dedicated Labelme set: data/processed/annotation_pack/labelme_blocked_50 with 50 renamed blocked-overlay images (blocked_001.png ... blocked_050.png).
- 2026-02-09T21:57:47+00:00 | Generated mapping file for renamed images: data/processed/annotation_pack/labelme_blocked_50/mapping.csv (filename -> timestamp).
- 2026-02-09T21:57:47+00:00 | Launched Labelme on labelme_blocked_50 folder with preset label caution.
- 2026-02-09T22:01:15+00:00 | Generated and opened land/ocean black-white map: outputs/qa/land_ocean_bw.png (black=land, white=ocean) from data/interim/env_grids/2024-10-31_18/x_bathy.npy.
- 2026-02-09T22:04:13+00:00 | Generated enhanced land-masked label set: data/processed/annotation_pack/labelme_blocked_50_enhanced (50 images + riskhint refs), and launched Labelme on this folder.
- 2026-02-09T22:22:24+00:00 | Ran annotation QC on labelme_blocked_50_enhanced JSONs: outputs/qa/annotation_overlap_report.csv (12 labeled files, overlap mean=0.0139, max=0.0749, no high-overlap cases).
- 2026-02-09T22:32:18+00:00 | Restarted Labelme and reopened annotation folder data/processed/annotation_pack/labelme_blocked_50_enhanced with label preset caution.
- 2026-02-09T22:46:13+00:00 | Added script backend/scripts/import_labelme_from_mapping.py to map renamed Labelme JSONs (blocked_XXX_riskhint.json) back to timestamp caution masks via mapping.csv.
- 2026-02-09T22:46:13+00:00 | Imported current annotations from labelme_blocked_50_enhanced: converted=32 JSONs into data/processed/annotation_pack/<timestamp>/caution_mask.(png|npy).
- 2026-02-09T22:46:13+00:00 | Built labeled-only manifest: data/processed/unet_manifest_labeled.csv with --skip-empty-caution (rows=19; train=13 val=6, val_month=202409).
- 2026-02-09T22:46:13+00:00 | Added quick baseline trainer backend/scripts/train_unet_quick.py (patch-based TinyUNet, weighted CE, train/val summary + checkpoints).
- 2026-02-09T22:46:13+00:00 | Fixed NaN instability in trainer by sanitizing non-finite inputs before stats/normalization.
- 2026-02-09T22:46:13+00:00 | Baseline run complete: outputs/train_runs/unet_quick_20260209_224525/summary.json (best epoch=7, val_loss=0.5703, val_miou=0.5713, val_iou_caution=0.0923, val_iou_blocked=0.8563).
- 2026-02-09T22:46:13+00:00 | Validation: python -m pytest -q -> 16 passed.
- 2026-02-09T22:46:41+00:00 | Pre-submit validation before push: python -m pytest -q in backend -> 16 passed.
- 2026-02-09T22:46:41+00:00 | Committed 24e8dcf: feat: import labelme mapping and run quick unet baseline.
- 2026-02-09T22:46:41+00:00 | Pushed 24e8dcf to origin/main.
- 2026-02-09T22:46:54+00:00 | Pre-submit validation before log-sync push: python -m pytest -q in backend -> 16 passed.
- 2026-02-09T22:50:01+00:00 | Verified current labelme JSON quality: 32 *_riskhint.json all contain positive caution polygons (label=caution).
- 2026-02-09T22:50:01+00:00 | Rebuilt labeled manifest from annotation_pack with --skip-empty-caution; result updated to rows=32 (train=26, val=6).
- 2026-02-09T22:50:01+00:00 | Reran quick baseline with 32 labeled samples: outputs/train_runs/unet_quick_20260209_224924/summary.json (best val_loss=0.5048, val_miou=0.6021, val_iou_caution=0.2399, val_iou_blocked=0.8748).
- 2026-02-09T22:58:20+00:00 | Added reusable TinyUNet module: backend/app/model/tiny_unet.py and test backend/tests/test_tiny_unet.py.
- 2026-02-09T22:58:20+00:00 | Upgraded trainer backend/scripts/train_unet_quick.py with stronger augmentation options (rot90/gamma/noise) and caution-focused patch sampling.
- 2026-02-09T22:58:20+00:00 | Added active learning script backend/scripts/active_learning_suggest.py; ranked 460 unlabeled candidates and exported top-20 set at outputs/active_learning/active_20260209_225559/labelme_active_topk.
- 2026-02-09T22:58:20+00:00 | Updated import script backend/scripts/import_labelme_from_mapping.py to support configurable filename template for different label rounds.
- 2026-02-09T22:58:20+00:00 | Training comparison runs completed: unet_quick_20260209_225653 and unet_quick_20260209_225723 (augmentation configs tested).
- 2026-02-09T22:58:20+00:00 | Validation: python -m pytest -q -> 17 passed.
- 2026-02-09T22:58:47+00:00 | Pre-submit validation before push: python -m pytest -q in backend -> 17 passed.
- 2026-02-09T22:58:47+00:00 | Committed 2692c35: feat: add active learning suggestion flow and stronger quick training aug.
- 2026-02-09T22:58:47+00:00 | Pushed 2692c35 to origin/main.
- 2026-02-09T22:59:04+00:00 | Pre-submit validation before log-sync push: python -m pytest -q in backend -> 17 passed.
- 2026-02-09T23:10:07+00:00 | Generated review overlays for active top-20 at outputs/active_learning/active_20260209_225559/labelme_active_topk/review_overlay20 (blocked=black, AI suggest=cyan+white edge), with mapping_review.csv.
- 2026-02-09T23:21:01+00:00 | Added backend/scripts/finalize_active_review.py to finalize active-review labels by merging human review JSON with AI suggested caution masks.
- 2026-02-09T23:21:01+00:00 | Finalized active top-20 review into annotation_pack: merged=19, suggest_only=1.
- 2026-02-09T23:21:01+00:00 | Rebuilt labeled manifest: data/processed/unet_manifest_labeled.csv -> rows=52 (train=39, val=13 with val_month=202408).
- 2026-02-09T23:21:01+00:00 | Retrained quick U-Net on 52 labels: outputs/train_runs/unet_quick_20260209_231959/summary.json (best val_loss=0.2922, val_miou=0.6743, val_iou_caution=0.3012, val_iou_blocked=0.9394; best caution epoch=4 -> 0.3290).
- 2026-02-09T23:21:01+00:00 | Validation: python -m pytest -q -> 17 passed.
- 2026-02-09T23:21:23+00:00 | Pre-submit validation before push: python -m pytest -q in backend -> 17 passed.
- 2026-02-09T23:21:23+00:00 | Committed bef9055: feat: finalize active review labels and retrain on 52 samples.
- 2026-02-09T23:21:23+00:00 | Pushed bef9055 to origin/main.
- 2026-02-09T23:27:35+00:00 | Started new active-learning round: generated active_20260209_232624 top-20 and review overlays at outputs/active_learning/active_20260209_232624/labelme_active_topk/review_overlay20; restarted Labelme on review folder.
- 2026-02-10T12:13:32+00:00 | Finalized active round active_20260209_232624 review labels into annotation_pack: merged=11, suggest_only=9.
- 2026-02-10T12:13:32+00:00 | Rebuilt labeled manifest after round merge: data/processed/unet_manifest_labeled.csv -> rows=72 (train=57, val=15, val_month=202408).
- 2026-02-10T12:13:32+00:00 | Retrained quick U-Net on 72 labels: outputs/train_runs/unet_quick_20260210_121250/summary.json (best val_loss=0.3840, val_miou=0.6461, val_iou_caution=0.3801, val_iou_blocked=0.8822).
- 2026-02-10T12:26:50+00:00 | Started next active-learning round: active_20260210_122544 (top-20), generated review overlays at outputs/active_learning/active_20260210_122544/labelme_active_topk/review_overlay20, and restarted Labelme on this folder.
- 2026-02-10T13:08:08+00:00 | Finalized active round active_20260210_122544 review labels into annotation_pack: merged=15, suggest_only=5.
- 2026-02-10T13:08:08+00:00 | Rebuilt labeled manifest after merge: data/processed/unet_manifest_labeled.csv -> rows=92 (train=71, val=21, val_month=202408).
- 2026-02-10T13:08:08+00:00 | Retrained quick U-Net on 92 labels: outputs/train_runs/unet_quick_20260210_130711/summary.json (best val_loss=0.3167, val_miou=0.7220, val_iou_caution=0.4313, val_iou_blocked=0.9325; peak caution epoch=9 -> 0.4637).
- 2026-02-10T13:11:33+00:00 | Started new active-learning round: active_20260210_131016 (top-20), generated review overlays at outputs/active_learning/active_20260210_131016/labelme_active_topk/review_overlay20, and restarted Labelme on review folder.
- 2026-02-10T13:16:27+00:00 | Finalized active round active_20260210_131016 review labels into annotation_pack: merged=9, suggest_only=11.
- 2026-02-10T13:16:27+00:00 | Rebuilt labeled manifest after merge: data/processed/unet_manifest_labeled.csv -> rows=112 (train=80, val=32, val_month=202408).
- 2026-02-10T13:16:27+00:00 | Retrained quick U-Net on 112 labels: outputs/train_runs/unet_quick_20260210_131527/summary.json (best val_loss=0.3352, val_miou=0.6629, val_iou_caution=0.4300, val_iou_blocked=0.8681; peak caution epoch=3 -> 0.4816).
- 2026-02-10T13:27:46.9066241Z | Tuned backend/scripts/active_learning_suggest.py for conservative AI suggestion defaults: pred_threshold=0.60, max_suggest_ratio=0.06, plus neighborhood smoothing controls.
- 2026-02-10T13:27:46.9066241Z | Added tests backend/tests/test_active_learning_suggest.py covering ratio-capped thresholding and binary mask smoothing behavior.
- 2026-02-10T13:27:46.9066241Z | Validation: python -m pytest -q in backend -> 19 passed.
- 2026-02-10T13:27:46.9066241Z | Ran strict active-learning export: python backend/scripts/active_learning_suggest.py --top-k 20 --pred-threshold 0.65 --max-suggest-ratio 0.04 --smooth-min-neighbors 2 --smooth-iters 1 -> outputs/active_learning/active_20260210_132408/labelme_active_topk.
- 2026-02-10T13:27:46.9066241Z | QA on top-20 suggested masks: mean sea ratio=0.0466 (min=0.0455, max=0.0480), confirming lower AI auto-annotation coverage for manual²¹±ê workflow.
- 2026-02-10T13:30:26.0264903Z | Prepared isolated labeling folder for current active round: outputs/active_learning/active_20260210_132408/labelme_active_topk/only20_raw (20 raw images + mapping.csv + README.txt).
- 2026-02-10T13:30:26.0264903Z | Launched Labelme and Explorer on outputs/active_learning/active_20260210_132408/labelme_active_topk/only20_raw for manual caution²¹±ê workflow.
- 2026-02-10T13:32:11.0224680Z | Fixed black active-learning base images: backend/scripts/active_learning_suggest.py now loads quicklook_blocked_overlay.png, then quicklook.png, then quicklook_riskhint.png before zero fallback.
- 2026-02-10T13:32:11.0224680Z | Repaired current batch visuals in-place for outputs/active_learning/active_20260210_132408/labelme_active_topk and only20_raw by regenerating active_001..020 from annotation_pack quicklook images.
- 2026-02-10T13:32:11.0224680Z | Validation: python -m pytest -q in backend -> 19 passed. Relaunched Labelme on outputs/active_learning/active_20260210_132408/labelme_active_topk/only20_raw.
- 2026-02-10T13:38:25.4139201Z | Generated overlay labeling set with landmask+AI suggestion: outputs/active_learning/active_20260210_132408/labelme_active_topk/only20_landmask_ai (blocked/land=black, AI caution=cyan with white boundary).
- 2026-02-10T13:38:25.4139201Z | Launched Explorer and Labelme on outputs/active_learning/active_20260210_132408/labelme_active_topk/only20_landmask_ai for manual²¹±ê.
- 2026-02-10T14:40:01.6346997Z | Finalized user-completed active round from outputs/active_learning/active_20260210_132408/labelme_active_topk/only20_landmask_ai: merged=20, human_only=0, suggest_only=0.
- 2026-02-10T14:40:01.6346997Z | Rebuilt labeled manifest: data/processed/unet_manifest_labeled.csv -> rows=132 (train=90, val=42, val_month=202408, skip-empty-caution=true).
- 2026-02-10T14:40:01.6346997Z | Trained quick U-Net baseline (default 4 epochs): outputs/train_runs/unet_quick_20260210_143810/summary.json.
- 2026-02-10T14:40:01.6346997Z | Trained extended quick U-Net (12 epochs): outputs/train_runs/unet_quick_20260210_143847/summary.json (best loss epoch=10, val_loss=0.3896; peak caution epoch=11, val_iou_caution=0.3683).
- 2026-02-10T14:40:16.2746577Z | Validation: python -m pytest -q in backend -> 19 passed.
- 2026-02-10T14:53:33.0373774Z | Added backend/scripts/filter_hard_samples.py to score label difficulty (CE + caution IoU) and export filtered manifest by dropping top-K hard train samples.
- 2026-02-10T14:53:33.0373774Z | Ran hard-sample audit using outputs/train_runs/unet_quick_20260210_131527/best.pt: exported data/processed/unet_manifest_filtered_hard10.csv and report data/processed/unet_manifest_filtered_hard10.report.json (dropped 10 train rows, 0 val rows).
- 2026-02-10T14:53:33.0373774Z | Ran augmentation+cycle training experiments: unet_cycle_full_v1, unet_cycle_filtered_v1, unet_cycle_filtered_v2, unet_cycle_full_v2, unet_cycle_full_v3 for comparison.
- 2026-02-10T14:53:33.0373774Z | Best balanced run so far from this sweep: outputs/train_runs/unet_cycle_full_v1/summary.json (best val_loss=0.2587, val_miou=0.7058, val_iou_caution=0.4198, val_iou_blocked=0.9206).
- 2026-02-10T14:53:33.0373774Z | Validation: python -m pytest -q in backend -> 19 passed.
- 2026-02-10T15:00:57.8603599Z | Started new optimization labeling round from outputs/train_runs/unet_cycle_full_v1/summary.json using active_learning_suggest (top_k=20, pred_threshold=0.68, max_suggest_ratio=0.035, smooth=2x1).
- 2026-02-10T15:00:57.8603599Z | Generated round outputs at outputs/active_learning/active_20260210_145951/labelme_active_topk with suggest ratio mean=0.0411 (min=0.0392, max=0.0432).
- 2026-02-10T15:00:57.8603599Z | Prepared labeling folders: only20_raw and only20_landmask_ai (blocked=black, AI suggestion=cyan+white edge), and launched Labelme on only20_landmask_ai.
- 2026-02-10T15:10:30.9966684Z | Received user completion notice for active round active_20260210_145951; detected 15/20 review JSONs in only20_landmask_ai.
- 2026-02-10T15:10:30.9966684Z | Finalized partial review merge for active_20260210_145951 with --merge-with-suggest: merged=15, suggest_only=5, human_only=0.
- 2026-02-10T15:10:30.9966684Z | Rebuilt manifest after merge: data/processed/unet_manifest_labeled.csv -> rows=152 (train=109, val=43, val_month=202408).
- 2026-02-10T15:10:30.9966684Z | Trained run outputs/train_runs/unet_cycle_full_v1_r2 and outputs/train_runs/unet_cycle_full_v1_r2_focus for post-merge evaluation.
- 2026-02-10T15:14:17.5201507Z | User selected option 2: prepared missing-5 rapid review set for active_20260210_145951 at labelme_active_topk/review_missing5_landmask_ai and review_missing5_raw, with mapping_missing5.csv.
- 2026-02-10T15:14:17.5201507Z | Missing ranks/timestamps: 8(2024-07-22_06), 11(2024-07-20_12), 17(2024-07-22_00), 19(2024-07-25_12), 20(2024-07-30_06). Launched Labelme on review_missing5_landmask_ai.
- 2026-02-10T15:23:09.8742288Z | Received user completion for missing-5 review; copied newly provided review JSONs into active_20260210_145951/only20_landmask_ai (4 new: active_008/017/019/020).
- 2026-02-10T15:23:09.8742288Z | Re-finalized active_20260210_145951 with full mapping: merged=19, suggest_only=1 (remaining missing review: active_011 / 2024-07-20_12).
- 2026-02-10T15:23:09.8742288Z | Rebuilt manifest post-merge: data/processed/unet_manifest_labeled.csv -> rows=152 (train=109, val=43).
- 2026-02-10T15:23:09.8742288Z | Retrain comparison runs completed: outputs/train_runs/unet_cycle_full_v1_r3 and outputs/train_runs/unet_cycle_full_v1_r3_focus; both underperform prior bests (v1 and v1_r2_focus).
