import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from preprocess_centroids import (
    build_centroid_tracks_from_clusters,
    run_dynamic_clustering_scene,
)
from utils.data import resolve_motsynth_root
from utils.plotting import get_distinct_colors, prepare_matplotlib
from utils.run_logging import finalize_run_logging, start_run_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw vs centroid trajectories by MOTSynth annotation scene id "
            "(e.g., 000, 001), not processed sample indices."
        )
    )
    parser.add_argument(
        "--scenes",
        type=str,
        required=True,
        help="Comma-separated scene ids, e.g. 000,001,033",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data root used by MOTSynth resolver (fallbacks include dataset/).",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Subsample frame interval for visualization/clustering.",
    )
    parser.add_argument(
        "--min_track_len",
        type=int,
        default=1,
        help="Keep only agent ids with at least this many points in selected frames.",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=0,
        help=(
            "Target number of raw agents to visualize/export per scene. "
            "0 means use all eligible agents."
        ),
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=0,
        help=(
            "Target number of centroid trajectories to visualize per scene. "
            "0 means use all generated centroids."
        ),
    )
    parser.add_argument("--direction_thresh_deg", type=float, default=50.0)
    parser.add_argument("--distance_thresh_px", type=float, default=120.0)
    parser.add_argument("--lof_contamination", type=float, default=0.2)
    parser.add_argument("--lof_neighbor_ratio", type=float, default=0.8)
    parser.add_argument("--reeval_interval", type=int, default=10)
    parser.add_argument("--temporary_recluster_min_size", type=int, default=10)
    parser.add_argument("--cluster_empty_tolerance", type=int, default=3)
    parser.add_argument("--centroid_update_interval", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Default: outputs/visualizations/scene_annotations_<timestamp>",
    )
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--disable_file_logging", action="store_true")
    return parser.parse_args()


def _prepare_matplotlib():
    return prepare_matplotlib(use_agg=True)


def parse_scene_ids(scenes_arg: str) -> List[str]:
    scene_ids = [s.strip() for s in scenes_arg.split(",") if s.strip()]
    return [f"{int(s):03d}" if s.isdigit() else s for s in scene_ids]


def load_scene_df(motsynth_root: str, scene_id: str) -> pd.DataFrame:
    gt_path = os.path.join(motsynth_root, "mot_annotations", scene_id, "gt", "gt.txt")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Missing scene annotation file: {gt_path}")

    cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    df = pd.read_csv(gt_path, header=None, names=cols)
    df["cx"] = df["bb_left"] + (df["bb_width"] / 2.0)
    df["cy"] = df["bb_top"] + (df["bb_height"] / 2.0)
    return df[["frame", "id", "cx", "cy"]]


def build_scene_tensors(
    scene_df: pd.DataFrame,
    frame_step: int,
    min_track_len: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    all_frames = sorted(scene_df["frame"].unique().tolist())
    selected_frames = all_frames[:: max(1, frame_step)]
    selected = scene_df[scene_df["frame"].isin(selected_frames)].copy()

    id_counts = selected.groupby("id")["frame"].nunique()
    kept_ids = sorted(id_counts[id_counts >= min_track_len].index.astype(int).tolist())
    selected = selected[selected["id"].isin(kept_ids)].copy()
    selected_frames = sorted(selected["frame"].unique().tolist())

    id_to_idx = {pid: i for i, pid in enumerate(kept_ids)}
    frame_to_idx = {fr: t for t, fr in enumerate(selected_frames)}

    traj = np.zeros((len(kept_ids), len(selected_frames), 1, 3), dtype=np.float32)
    mask = np.zeros((len(kept_ids), len(selected_frames), 1), dtype=np.float32)

    for row in selected.itertuples(index=False):
        i = id_to_idx[int(row.id)]
        t = frame_to_idx[int(row.frame)]
        traj[i, t, 0, 0] = float(row.cx)
        traj[i, t, 0, 1] = float(row.cy)
        traj[i, t, 0, 2] = 0.0
        mask[i, t, 0] = 1.0

    return traj, mask, kept_ids, selected_frames


def scene_points_from_tracks(tracks: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if len(tracks) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = tracks[:, :, :2][masks > 0]
    return np.asarray(pts, dtype=np.float32) if len(pts) else np.zeros((0, 2), dtype=np.float32)


def select_agent_subset(
    traj: np.ndarray,
    mask: np.ndarray,
    raw_ids: Sequence[int],
    n_agents: int,
):
    if n_agents <= 0 or n_agents >= len(raw_ids):
        return traj, mask, list(raw_ids)

    ranking = []
    for i, pid in enumerate(raw_ids):
        valid_len = int((mask[i, :, 0] > 0).sum())
        ranking.append((valid_len, i, int(pid)))

    ranking.sort(reverse=True)
    keep_indices = sorted([r[1] for r in ranking[:n_agents]])
    sub_traj = traj[keep_indices]
    sub_mask = mask[keep_indices]
    sub_ids = [raw_ids[i] for i in keep_indices]
    return sub_traj, sub_mask, sub_ids


def save_scene_plot(
    scene_id: str,
    raw_df: pd.DataFrame,
    raw_ids: Sequence[int],
    raw_tracks: np.ndarray,
    raw_masks: np.ndarray,
    centroid_tracks: np.ndarray,
    centroid_masks: np.ndarray,
    centroid_ids: Sequence[int],
    out_path: str,
):
    plt = _prepare_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    raw_colors = get_distinct_colors(len(raw_ids))
    for i, pid in enumerate(raw_ids):
        d = raw_df[raw_df["id"] == pid].sort_values("frame")
        if len(d) < 2:
            continue
        axes[0].plot(
            d["cx"].to_numpy(),
            d["cy"].to_numpy(),
            linewidth=1.0,
            alpha=0.9,
            color=raw_colors[i],
        )

    cen_colors = get_distinct_colors(len(centroid_ids))
    for i, _cid in enumerate(centroid_ids):
        valid = np.where(centroid_masks[i] > 0)[0]
        if len(valid) < 2:
            continue
        xy = centroid_tracks[i, valid]
        axes[1].plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=1.2,
            alpha=0.95,
            color=cen_colors[i],
        )

    raw_pts = scene_points_from_tracks(raw_tracks[:, :, 0, :], raw_masks[:, :, 0])
    cen_pts = scene_points_from_tracks(centroid_tracks, centroid_masks)
    all_pts = np.concatenate([raw_pts, cen_pts], axis=0) if len(cen_pts) else raw_pts
    if len(all_pts):
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        padx = (xmax - xmin) * 0.03 + 1.0
        pady = (ymax - ymin) * 0.03 + 1.0
        for ax in axes:
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymax + pady, ymin - pady)  # invert y-axis once

    axes[0].set_title(f"Scene {scene_id} Raw Trajectories | n_agents={len(raw_ids)}")
    axes[1].set_title(
        f"Scene {scene_id} Dynamic Centroid Trajectories | n_clusters={len(centroid_ids)}"
    )
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"MOTSynth scene {scene_id}: raw vs centroid", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scene_csvs(
    scene_id: str,
    selected_df: pd.DataFrame,
    centroid_tracks: np.ndarray,
    centroid_masks: np.ndarray,
    centroid_ids: Sequence[int],
    centroid_meta: Dict[int, dict],
    frames: Sequence[int],
    out_dir: str,
):
    raw_csv = os.path.join(out_dir, f"{scene_id}_raw_tracks.csv")
    selected_df.sort_values(["id", "frame"]).to_csv(raw_csv, index=False)

    rows = []
    for i, cluster_id in enumerate(centroid_ids):
        valid = np.where(centroid_masks[i] > 0)[0]
        for t in valid:
            rows.append(
                (
                    int(frames[t]),
                    int(cluster_id),
                    float(centroid_tracks[i, t, 0]),
                    float(centroid_tracks[i, t, 1]),
                    int(centroid_meta[cluster_id].get("cluster_size", 0)),
                )
            )
    centroid_df = pd.DataFrame(
        rows,
        columns=["frame", "cluster_id", "cx", "cy", "cluster_size"],
    )
    centroid_csv = os.path.join(out_dir, f"{scene_id}_centroid_tracks.csv")
    centroid_df.to_csv(centroid_csv, index=False)

    metadata_json = os.path.join(out_dir, f"{scene_id}_centroid_metadata.json")
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in centroid_meta.items()}, f, indent=2)

    return raw_csv, centroid_csv, metadata_json


def select_centroid_subset(
    centroid_tracks: np.ndarray,
    centroid_masks: np.ndarray,
    centroid_ids: Sequence[int],
    centroid_meta: Dict[int, dict],
    n_clusters: int,
):
    if n_clusters <= 0 or n_clusters >= len(centroid_ids):
        return centroid_tracks, centroid_masks, list(centroid_ids), dict(centroid_meta)

    ranking = []
    for i, cluster_id in enumerate(centroid_ids):
        valid_len = int((centroid_masks[i] > 0).sum())
        cluster_size = int(centroid_meta.get(cluster_id, {}).get("cluster_size", 1))
        score = valid_len * max(1, cluster_size)
        ranking.append((score, valid_len, cluster_size, i))

    ranking.sort(reverse=True)
    keep_indices = sorted([r[3] for r in ranking[:n_clusters]])

    sub_tracks = centroid_tracks[keep_indices]
    sub_masks = centroid_masks[keep_indices]
    sub_ids = [centroid_ids[i] for i in keep_indices]
    sub_meta = {cid: centroid_meta[cid] for cid in sub_ids if cid in centroid_meta}
    return sub_tracks, sub_masks, sub_ids, sub_meta


def main():
    args = parse_args()
    scene_ids = parse_scene_ids(args.scenes)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join("outputs", "visualizations", f"scene_annotations_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    log_state = None
    if not args.disable_file_logging:
        log_state, log_path = start_run_logging(args.log_dir, script_name="viz_scene")
        print(f"[run-log] Capturing stdout/stderr to {log_path}")

    try:
        motsynth_root = resolve_motsynth_root(args.data_dir)
        print(f"MOTSynth root resolved to: {motsynth_root}")
        print(f"Scenes: {scene_ids}")

        manifest = {}

        for scene_id in scene_ids:
            print(f"\n[Scene {scene_id}] loading annotations...")
            scene_df = load_scene_df(motsynth_root, scene_id)

            traj, mask, raw_ids, frames = build_scene_tensors(
                scene_df=scene_df,
                frame_step=args.frame_step,
                min_track_len=args.min_track_len,
            )
            if len(raw_ids) == 0:
                print(f"[Scene {scene_id}] no trajectories after filtering; skipping.")
                continue
            raw_agents_total = len(raw_ids)
            traj_plot, mask_plot, raw_ids_plot = select_agent_subset(
                traj=traj,
                mask=mask,
                raw_ids=raw_ids,
                n_agents=args.n_agents,
            )

            selected_df = scene_df[
                scene_df["frame"].isin(frames) & scene_df["id"].isin(raw_ids_plot)
            ].copy()

            clusters = run_dynamic_clustering_scene(
                scene_traj=traj_plot,
                scene_mask=mask_plot,
                frames=frames,
                direction_thresh_deg=args.direction_thresh_deg,
                distance_thresh_px=args.distance_thresh_px,
                lof_contamination=args.lof_contamination,
                lof_neighbor_ratio=args.lof_neighbor_ratio,
                reeval_interval=args.reeval_interval,
                temporary_recluster_min_size=args.temporary_recluster_min_size,
                cluster_empty_tolerance=args.cluster_empty_tolerance,
                centroid_update_interval=args.centroid_update_interval,
            )

            centroid_tracks, centroid_masks, centroid_ids, centroid_meta = (
                build_centroid_tracks_from_clusters(clusters, frames)
            )
            (
                centroid_tracks_plot,
                centroid_masks_plot,
                centroid_ids_plot,
                centroid_meta_plot,
            ) = select_centroid_subset(
                centroid_tracks=centroid_tracks,
                centroid_masks=centroid_masks,
                centroid_ids=centroid_ids,
                centroid_meta=centroid_meta,
                n_clusters=args.n_clusters,
            )

            img_path = os.path.join(out_dir, f"{scene_id}_raw_vs_centroid.png")
            save_scene_plot(
                scene_id=scene_id,
                raw_df=selected_df,
                raw_ids=raw_ids_plot,
                raw_tracks=traj_plot,
                raw_masks=mask_plot,
                centroid_tracks=centroid_tracks_plot,
                centroid_masks=centroid_masks_plot,
                centroid_ids=centroid_ids_plot,
                out_path=img_path,
            )

            raw_csv, centroid_csv, metadata_json = save_scene_csvs(
                scene_id=scene_id,
                selected_df=selected_df,
                centroid_tracks=centroid_tracks_plot,
                centroid_masks=centroid_masks_plot,
                centroid_ids=centroid_ids_plot,
                centroid_meta=centroid_meta_plot,
                frames=frames,
                out_dir=out_dir,
            )

            manifest[scene_id] = {
                "raw_agent_count_total": int(raw_agents_total),
                "raw_agent_count_plotted": int(len(raw_ids_plot)),
                "centroid_cluster_count_total": int(len(centroid_ids)),
                "centroid_cluster_count_plotted": int(len(centroid_ids_plot)),
                "frame_min": int(min(frames)),
                "frame_max": int(max(frames)),
                "num_frames": int(len(frames)),
                "files": {
                    "image": img_path,
                    "raw_csv": raw_csv,
                    "centroid_csv": centroid_csv,
                    "centroid_metadata": metadata_json,
                },
            }
            print(
                f"[Scene {scene_id}] raw_agents_total={raw_agents_total} | "
                f"raw_agents_plotted={len(raw_ids_plot)} | "
                f"centroid_clusters_total={len(centroid_ids)} | "
                f"centroid_clusters_plotted={len(centroid_ids_plot)} | saved={img_path}"
            )

        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "motsynth_root": motsynth_root,
                    "args": vars(args),
                    "scenes": manifest,
                },
                f,
                indent=2,
            )
        print(f"\nDone. Output directory: {out_dir}")
        print(f"Manifest: {manifest_path}")

    finally:
        finalize_run_logging(log_state)


if __name__ == "__main__":
    main()
