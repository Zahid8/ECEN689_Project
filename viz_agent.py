import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from preprocess_centroids import build_centroid_tracks_from_clusters, run_dynamic_clustering_scene
from utils.data import make_motsynth_df, resolve_motsynth_root
from utils.run_logging import finalize_run_logging, start_run_logging


@dataclass
class PlotConfig:
    direction_thresh_deg: float = 50.0
    distance_thresh_px: float = 120.0
    lof_contamination: float = 0.2
    lof_neighbor_ratio: float = 0.8
    reeval_interval: int = 10
    temporary_recluster_min_size: int = 10
    cluster_empty_tolerance: int = 3
    centroid_update_interval: int = 1


def _prepare_matplotlib(use_agg: bool = True):
    try:
        import matplotlib

        if use_agg:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for viz_agent.py. Install it with: pip install matplotlib"
        ) from exc
    return plt


def _load_raw_bbox_centers(scene_id: str, data_dir: str, start: int, finish: int) -> Tuple[pd.DataFrame, str]:
    scene = f"{int(scene_id):03d}" if str(scene_id).isdigit() else str(scene_id)
    motsynth_root = resolve_motsynth_root(data_dir)
    gt_path = os.path.join(motsynth_root, "mot_annotations", scene, "gt", "gt.txt")
    raw_df = make_motsynth_df(gt_path)[["frame", "id", "bb_center_x", "bb_center_y"]].copy()
    raw_df = raw_df.rename(columns={"bb_center_x": "x", "bb_center_y": "y"})
    raw_df["frame"] = raw_df["frame"].astype(int)
    raw_df["id"] = raw_df["id"].astype(int)
    raw_df = raw_df[(raw_df["frame"] >= start) & (raw_df["frame"] < finish)].copy()
    raw_df = raw_df.sort_values(["frame", "id"]).reset_index(drop=True)
    return raw_df, gt_path


def _build_scene_tensors(raw_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    if raw_df.empty:
        return (
            np.zeros((0, 0, 1, 3), dtype=np.float32),
            np.zeros((0, 0, 1), dtype=np.float32),
            [],
            [],
        )

    frames = sorted(raw_df["frame"].astype(int).unique().tolist())
    ids = sorted(raw_df["id"].astype(int).unique().tolist())
    id2i = {pid: i for i, pid in enumerate(ids)}
    frame2t = {fr: t for t, fr in enumerate(frames)}

    traj = np.zeros((len(ids), len(frames), 1, 3), dtype=np.float32)
    mask = np.zeros((len(ids), len(frames), 1), dtype=np.float32)

    for row in raw_df.itertuples(index=False):
        i = id2i[int(row.id)]
        t = frame2t[int(row.frame)]
        traj[i, t, 0, 0] = float(row.x)
        traj[i, t, 0, 1] = float(row.y)
        traj[i, t, 0, 2] = 0.0
        mask[i, t, 0] = 1.0

    return traj, mask, ids, frames


def _run_clustering(
    traj: np.ndarray,
    mask: np.ndarray,
    frames: List[int],
    cfg: PlotConfig,
) -> Tuple[pd.DataFrame, Dict[int, dict]]:
    if len(frames) == 0 or traj.shape[0] == 0:
        return pd.DataFrame(columns=["frame", "cluster_id", "x", "y", "cluster_size"]), {}

    clusters = run_dynamic_clustering_scene(
        scene_traj=traj,
        scene_mask=mask,
        frames=frames,
        direction_thresh_deg=cfg.direction_thresh_deg,
        distance_thresh_px=cfg.distance_thresh_px,
        lof_contamination=cfg.lof_contamination,
        lof_neighbor_ratio=cfg.lof_neighbor_ratio,
        reeval_interval=cfg.reeval_interval,
        temporary_recluster_min_size=cfg.temporary_recluster_min_size,
        cluster_empty_tolerance=cfg.cluster_empty_tolerance,
        centroid_update_interval=cfg.centroid_update_interval,
    )
    centroid_tracks, centroid_masks, centroid_ids, centroid_meta = build_centroid_tracks_from_clusters(clusters, frames)

    rows: List[List[float]] = []
    for i, cid in enumerate(centroid_ids):
        valid = np.where(centroid_masks[i] > 0)[0]
        for t in valid:
            rows.append(
                [
                    int(frames[t]),
                    int(cid),
                    float(centroid_tracks[i, t, 0]),
                    float(centroid_tracks[i, t, 1]),
                    int(centroid_meta[int(cid)].get("cluster_size", 0)),
                ]
            )

    cluster_df = pd.DataFrame(rows, columns=["frame", "cluster_id", "x", "y", "cluster_size"])
    return cluster_df, centroid_meta


def _plot_raw_vs_cluster(
    raw_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    scene_id: str,
    start: int,
    finish: int,
    out_path: str,
    show_plot: bool,
) -> None:
    plt = _prepare_matplotlib(use_agg=not show_plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_raw, ax_cluster = axes

    # teammate style: one color per id
    raw_ids = sorted(raw_df["id"].astype(int).unique().tolist())
    raw_cmap = plt.cm.get_cmap("hsv", max(1, len(raw_ids)))
    for i, pid in enumerate(raw_ids):
        sub = raw_df[raw_df["id"] == pid].sort_values("frame")
        if len(sub) < 2:
            continue
        ax_raw.plot(sub["x"], sub["y"], linewidth=1.0, alpha=0.9, color=raw_cmap(i))

    ax_raw.set_title(f"Raw trajectories scene {scene_id}, #pedestrians={len(raw_ids)}")
    ax_raw.set_xlabel("x")
    ax_raw.set_ylabel("y")
    ax_raw.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # teammate style: one color per cluster
    if not cluster_df.empty:
        cl_ids = sorted(cluster_df["cluster_id"].astype(int).unique().tolist())
        cl_cmap = plt.cm.get_cmap("hsv", max(1, len(cl_ids)))
        for i, cid in enumerate(cl_ids):
            sub = cluster_df[cluster_df["cluster_id"] == cid].sort_values("frame")
            if len(sub) < 2:
                continue
            ax_cluster.plot(sub["x"], sub["y"], linewidth=1.4, alpha=0.95, color=cl_cmap(i))
        ax_cluster.set_title(f"Cluster trajectories scene {scene_id}, #clusters={len(cl_ids)}")
    else:
        ax_cluster.set_title("Cluster trajectories (empty)")

    ax_cluster.set_xlabel("x")
    ax_cluster.set_ylabel("y")
    ax_cluster.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.suptitle(f"Frames [{start}, {finish})", fontsize=12)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=180)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_plot_pipeline(
    scene_id: str,
    data_dir: str,
    start: int,
    finish: int,
    cfg: PlotConfig,
    output_dir: str,
    save_image: bool = True,
    show_plot: bool = False,
) -> Dict:
    scene = f"{int(scene_id):03d}" if str(scene_id).isdigit() else str(scene_id)
    raw_df, gt_path = _load_raw_bbox_centers(scene, data_dir, start, finish)
    traj, mask, raw_ids, frames = _build_scene_tensors(raw_df)

    cluster_df, centroid_meta = _run_clustering(traj, mask, frames, cfg)

    os.makedirs(output_dir, exist_ok=True)
    raw_csv = os.path.join(output_dir, f"{scene}_raw_tracks.csv")
    cl_csv = os.path.join(output_dir, f"{scene}_cluster_tracks.csv")
    meta_json = os.path.join(output_dir, f"{scene}_cluster_metadata.json")
    raw_df.to_csv(raw_csv, index=False)
    cluster_df.to_csv(cl_csv, index=False)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in centroid_meta.items()}, f, indent=2)

    image_path = os.path.join(output_dir, f"{scene}_raw_vs_cluster.png") if save_image else ""
    _plot_raw_vs_cluster(
        raw_df=raw_df,
        cluster_df=cluster_df,
        scene_id=scene,
        start=start,
        finish=finish,
        out_path=image_path,
        show_plot=show_plot,
    )

    summary = {
        "scene_id": scene,
        "gt_path": gt_path,
        "start_frame": int(start),
        "finish_frame": int(finish),
        "num_frames_used": int(len(frames)),
        "raw_ped_count": int(len(raw_ids)),
        "cluster_count": int(cluster_df["cluster_id"].nunique()) if not cluster_df.empty else 0,
        "raw_csv": raw_csv,
        "cluster_csv": cl_csv,
        "cluster_metadata": meta_json,
        "image": image_path if image_path else None,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="CrowdCluster-style raw vs cluster plotting for one MOTSynth scene.")
    parser.add_argument("--scene_id", type=str, default="000", help="Scene id, e.g. 000")
    parser.add_argument("--data_dir", type=str, default="data", help="Data root (resolver supports dataset fallback).")
    parser.add_argument("--start", type=int, default=75, help="Inclusive start frame.")
    parser.add_argument("--finish", type=int, default=950, help="Exclusive finish frame.")
    parser.add_argument("--direction_thresh_deg", type=float, default=50.0)
    parser.add_argument("--distance_thresh_px", type=float, default=120.0)
    parser.add_argument("--lof_contamination", type=float, default=0.2)
    parser.add_argument("--lof_neighbor_ratio", type=float, default=0.8)
    parser.add_argument("--reeval_interval", type=int, default=10)
    parser.add_argument("--temporary_recluster_min_size", type=int, default=10)
    parser.add_argument("--cluster_empty_tolerance", type=int, default=3)
    parser.add_argument("--centroid_update_interval", type=int, default=1)
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--no_save_image", action="store_true")
    parser.add_argument("--output_dir", type=str, default="", help="Default: outputs/visualizations/raw_vs_cluster_<timestamp>")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--disable_file_logging", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join("outputs", "visualizations", f"raw_vs_cluster_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    log_state = None
    if not args.disable_file_logging:
        log_state, log_path = start_run_logging(args.log_dir, script_name="viz_agent")
        print(f"[run-log] Capturing stdout/stderr to {log_path}")

    try:
        cfg = PlotConfig(
            direction_thresh_deg=args.direction_thresh_deg,
            distance_thresh_px=args.distance_thresh_px,
            lof_contamination=args.lof_contamination,
            lof_neighbor_ratio=args.lof_neighbor_ratio,
            reeval_interval=args.reeval_interval,
            temporary_recluster_min_size=args.temporary_recluster_min_size,
            cluster_empty_tolerance=args.cluster_empty_tolerance,
            centroid_update_interval=args.centroid_update_interval,
        )
        summary = run_plot_pipeline(
            scene_id=args.scene_id,
            data_dir=args.data_dir,
            start=args.start,
            finish=args.finish,
            cfg=cfg,
            output_dir=out_dir,
            save_image=not args.no_save_image,
            show_plot=args.show_plot,
        )
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"Summary: {summary_path}")
    finally:
        finalize_run_logging(log_state)


if __name__ == "__main__":
    main()
