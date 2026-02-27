from __future__ import annotations

from scripts.generate_p3_casebook import pick_case_windows


def test_pick_case_windows_evenly_spreads() -> None:
    ts = [f"2024-07-01_{h:02d}" for h in range(0, 24, 2)]  # 12 steps
    windows = pick_case_windows(ts, window=4, case_count=3)
    assert len(windows) == 3
    assert all(len(win) == 4 for win in windows)
    assert windows[0][0] == ts[0]
    assert windows[-1][-1] == ts[-1]


def test_pick_case_windows_small_dataset() -> None:
    ts = ["2024-07-01_00", "2024-07-01_06", "2024-07-01_12"]
    windows = pick_case_windows(ts, window=4, case_count=3)
    assert windows == []
    windows2 = pick_case_windows(ts, window=2, case_count=2)
    assert len(windows2) >= 1
