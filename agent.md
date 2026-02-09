# NSR2 Agent Operating Rules

Last updated: 2026-02-09T20:11:51+00:00

## Fixed Principles

1. Before every GitHub push/submit, run a review and `pytest` at least once.
2. GitHub repository: `https://github.com/Cassian2006/NSR2.git` (currently no submissions yet).
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
