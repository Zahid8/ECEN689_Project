#!/usr/bin/env python3
"""Plot raw MOTSynth CrowdCluster export vs optional centroid CSV from main.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_crowdcluster_txt(path: Path) -> pd.DataFrame:
    """Load space-delimited frame id x y file."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["frame", "id", "x", "y"],
        engine="python",
    )
    return df


def load_centroid_csv(path: Path) -> pd.DataFrame:
    """Load CrowdCluster centroid CSV (cluster id inserted at column 1)."""
    raw = pd.read_csv(path, header=None)
    ncols = raw.shape[1]
    if ncols < 4:
        raise ValueError(f"Expected at least 4 columns in {path}, got {ncols}")
    out = raw.iloc[:, :4].copy()
    out.columns = ["frame", "cluster", "x", "y"]
    return out


def plot_raw_vs_clustered(
    txt_path: Path,
    csv_path: Optional[Path],
    out_png: Path,
    frame_min: Optional[int],
    frame_max: Optional[int],
    max_ids: int,
) -> None:
    """Save side-by-side figure: raw trajectories and centroid trajectories."""
    raw_df = load_crowdcluster_txt(txt_path)
    if frame_min is not None:
        raw_df = raw_df[raw_df["frame"] >= frame_min]
    if frame_max is not None:
        raw_df = raw_df[raw_df["frame"] <= frame_max]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    ids = raw_df["id"].unique()
    if len(ids) > max_ids:
        rng = np.random.default_rng(0)
        ids = rng.choice(ids, size=max_ids, replace=False)
        raw_df = raw_df[raw_df["id"].isin(ids)]

    ax0 = axes[0]
    for ped_id, g in raw_df.groupby("id"):
        g = g.sort_values("frame")
        ax0.plot(g["x"], g["y"], alpha=0.35, linewidth=0.8)
    ax0.set_title("Raw agent trajectories (bbox centers)")
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px)")
    ax0.invert_yaxis()

    ax1 = axes[1]
    if csv_path is not None and csv_path.is_file():
        cdf = load_centroid_csv(csv_path)
        if frame_min is not None:
            cdf = cdf[cdf["frame"] >= frame_min]
        if frame_max is not None:
            cdf = cdf[cdf["frame"] <= frame_max]
        clusters = cdf["cluster"].to_numpy()
        sc = ax1.scatter(
            cdf["x"],
            cdf["y"],
            c=clusters,
            cmap="tab20",
            s=6,
            alpha=0.85,
        )
        plt.colorbar(sc, ax=ax1, label="cluster id")
        ax1.set_title("CrowdCluster centroid CSV (frame vs xy)")
    else:
        ax1.text(
            0.5,
            0.5,
            "No centroid CSV provided or file missing.\nRun CrowdCluster main.py\nwith motsynth_settings.py",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Clustered (placeholder)")
    ax1.set_xlabel("x (px)")
    ax1.invert_yaxis()

    fig.suptitle(f"MOTSynth CrowdCluster viz: {txt_path.name}", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crowdcluster-txt",
        type=Path,
        required=True,
        help="Exported CrowdCluster-format .txt (frame id x y).",
    )
    parser.add_argument(
        "--centroid-csv",
        type=Path,
        default=None,
        help="Optional CSV from CrowdCluster main.py (filename setting).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument("--frame-min", type=int, default=None)
    parser.add_argument("--frame-max", type=int, default=None)
    parser.add_argument(
        "--max-ids",
        type=int,
        default=80,
        help="Max pedestrian ids to draw for raw panel (subsample if larger).",
    )
    args = parser.parse_args()

    plot_raw_vs_clustered(
        args.crowdcluster_txt.expanduser().resolve(),
        args.centroid_csv.expanduser().resolve() if args.centroid_csv else None,
        args.out_png.expanduser().resolve(),
        args.frame_min,
        args.frame_max,
        args.max_ids,
    )
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
