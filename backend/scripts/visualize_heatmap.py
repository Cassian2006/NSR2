from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize generated AIS heatmaps")
    p.add_argument("--tag", default="7d", help="Subfolder under data/ais_heatmap")
    p.add_argument("--out", default="", help="Output PNG path")
    return p.parse_args()


def pick_examples(files: list[Path]) -> list[Path]:
    if not files:
        return []
    if len(files) == 1:
        return [files[0]]
    if len(files) == 2:
        return [files[0], files[1]]
    idx = sorted({0, len(files) // 2, len(files) - 1})
    return [files[i] for i in idx]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    heatmap_dir = project_root / "data" / "ais_heatmap" / args.tag
    if not heatmap_dir.exists():
        raise FileNotFoundError(f"Heatmap tag directory not found: {heatmap_dir}")

    files = sorted(heatmap_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy heatmaps found in: {heatmap_dir}")

    examples = pick_examples(files)
    arrays = [np.load(p) for p in examples]
    all_vals = np.concatenate([a.ravel() for a in arrays]).astype(np.float32)

    ncols = len(arrays)
    fig = plt.figure(figsize=(5 * ncols, 8), dpi=150)
    gs = fig.add_gridspec(2, ncols, height_ratios=[3, 1])

    for i, (path, arr) in enumerate(zip(examples, arrays, strict=True)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(arr, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(path.stem)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    hist_ax = fig.add_subplot(gs[1, :])
    hist_ax.hist(all_vals, bins=80, color="#1f77b4", alpha=0.9)
    hist_ax.set_title("Value distribution (sampled panels)")
    hist_ax.set_xlabel("Normalized heat value")
    hist_ax.set_ylabel("Count")

    fig.suptitle(f"AIS Heatmap Preview | tag={args.tag} | files={len(files)}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = project_root / "outputs" / "qa" / f"heatmap_preview_{args.tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

