"""Core raw-to-cluster pipeline migrated from CrowdCluster main flow.

This module keeps the computational clustering behavior while removing plotting
and animation side effects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from utils.data import make_mot_standard_gt_df


@dataclass
class Raw2ClusterConfig:
    """Configuration for the migrated raw-to-cluster pipeline."""

    n_initial_cluster: int = 8
    tdist: float = 110.0
    tdirect: float = 50.0
    eval_interval: int = 10
    output_csv: Optional[Path] = None
    eval_frame_interval: int = 1


def calculate_direction(delta_x: float, delta_y: float) -> float:
    """Return heading in degrees in range [0, 360)."""
    angle_radians = math.atan2(delta_y, delta_x)
    return (math.degrees(angle_radians) + 360.0) % 360.0


def smallest_angular_distance(angle1: float, angle2: float) -> float:
    """Return minimum angular distance between two headings."""
    delta = abs(angle1 - angle2) % 360.0
    if delta > 180.0:
        delta = 360.0 - delta
    return delta


def _safe_float(value: Any) -> float:
    """Convert value to float and replace NaN with 0."""
    out = float(value)
    if math.isnan(out):
        return 0.0
    return out


def load_tracks(input_path: Path) -> pd.DataFrame:
    """Load trajectory data into frame/id/x/y columns.

    Supports:
    - space-separated `frame id x y` files (at least 4 columns)
    - MOT-style `gt.txt` with bbox (at least 6 columns): uses bbox center x/y
      via ``make_mot_standard_gt_df``, matching ``utils/data.py`` MOTSynth prep.
    """
    raw = pd.read_csv(input_path, sep=r"\s+|,", engine="python", header=None)
    if raw.shape[1] >= 6:
        center_df = make_mot_standard_gt_df(str(input_path))
        frame = center_df["frame"].astype(np.int64)
        ped_id = center_df["id"].astype(np.int64)
        x = center_df["bb_center_x"].astype(np.float64)
        y = center_df["bb_center_y"].astype(np.float64)
    elif raw.shape[1] >= 4:
        frame = raw.iloc[:, 0].astype(np.int64)
        ped_id = raw.iloc[:, 1].astype(np.int64)
        x = raw.iloc[:, 2].astype(np.float64)
        y = raw.iloc[:, 3].astype(np.float64)
    else:
        raise ValueError(f"Unsupported input format with {raw.shape[1]} columns: {input_path}")

    df = pd.DataFrame({"frame": frame, "id": ped_id, "x": x, "y": y})
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def prepare_data(df: pd.DataFrame) -> np.ndarray:
    """Create per-detection array: frame, id, x, y, va, vx, vy, cluster_id."""
    data = df.copy()
    data["va"] = 0.0
    data["vx"] = 0.0
    data["vy"] = 0.0
    data["cluster_id"] = np.nan
    arr = data[["frame", "id", "x", "y", "va", "vx", "vy", "cluster_id"]].to_numpy(
        dtype=np.float64
    )

    prev_id: Optional[int] = None
    xcenters: List[float] = []
    ycenters: List[float] = []
    for i in range(len(arr)):
        ped_id = int(arr[i, 1])
        if prev_id is None or ped_id != prev_id:
            xcenters = []
            ycenters = []
        xcenters.append(float(arr[i, 2]))
        ycenters.append(float(arr[i, 3]))
        if len(xcenters) > 1:
            vx = _safe_float(xcenters[-2] - xcenters[-1])
            vy = _safe_float(ycenters[-2] - ycenters[-1])
            arr[i, 4] = calculate_direction(vx, vy)
            arr[i, 5] = vx
            arr[i, 6] = vy
        prev_id = ped_id
    return arr


def get_data_by_frame(data: np.ndarray, frame_no: int) -> np.ndarray:
    """Select rows for a single frame."""
    return data[data[:, 0] == frame_no].copy()


def _cluster_initial_frame(frame_data: np.ndarray, cfg: Raw2ClusterConfig) -> Dict[int, int]:
    """Build initial person-id to cluster-id map from the start frame."""
    if len(frame_data) == 0:
        return {}

    direction = frame_data[:, 4].reshape(-1, 1)
    n_clusters = min(cfg.n_initial_cluster, len(direction))
    if n_clusters <= 1:
        base_labels = np.zeros(len(direction), dtype=np.int64)
    else:
        base_model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="complete", metric="manhattan"
        )
        base_labels = base_model.fit_predict(direction)

    next_cluster_id = 0
    id_to_cluster: Dict[int, int] = {}
    for base_id in np.unique(base_labels):
        idx = np.where(base_labels == base_id)[0]
        members = frame_data[idx]
        if len(members) <= 1:
            pid = int(members[0, 1])
            id_to_cluster[pid] = next_cluster_id
            next_cluster_id += 1
            continue
        sub_model = AgglomerativeClustering(
            distance_threshold=cfg.tdist,
            n_clusters=None,
            linkage="complete",
            metric="manhattan",
        )
        sub_labels = sub_model.fit_predict(members[:, [2, 3]])
        for sub_id in np.unique(sub_labels):
            sub_idx = np.where(sub_labels == sub_id)[0]
            for member in members[sub_idx]:
                id_to_cluster[int(member[1])] = next_cluster_id
            next_cluster_id += 1
    return id_to_cluster


def _group_by_cluster(frame_data: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Group frame rows by cluster id."""
    valid = frame_data[~np.isnan(frame_data[:, 7])]
    if len(valid) == 0:
        return []
    result: List[Tuple[int, np.ndarray]] = []
    for cluster_id in np.unique(valid[:, 7].astype(np.int64)):
        members = valid[valid[:, 7].astype(np.int64) == cluster_id]
        result.append((int(cluster_id), members))
    return result


def _compute_centroids(frame_data: np.ndarray) -> List[List[float]]:
    """Compute centroid records for each cluster in one frame."""
    centroids: List[List[float]] = []
    grouped = _group_by_cluster(frame_data)
    for cluster_id, members in grouped:
        n = len(members)
        x = float(np.mean(members[:, 2]))
        y = float(np.mean(members[:, 3]))
        va = float(np.mean(members[:, 4]))
        vx = float(np.mean(members[:, 5]))
        vy = float(np.mean(members[:, 6]))
        points = members[:, [2, 3]]
        d = float(np.max(np.linalg.norm(points - np.array([x, y]), axis=1))) if n > 1 else 0.0
        centroids.append([float(members[0, 0]), float(cluster_id), x, y, va, vx, vy, d, float(n)])
    return centroids


def _assign_unknowns_to_centroids(
    frame_data: np.ndarray,
    centroids: Sequence[Sequence[float]],
    next_cluster_id: int,
    cfg: Raw2ClusterConfig,
) -> int:
    """Assign unknown cluster rows by nearest centroid with thresholds."""
    for idx in range(len(frame_data)):
        if not (math.isnan(frame_data[idx, 7]) or int(frame_data[idx, 7]) == -1):
            continue
        best_label: Optional[int] = None
        best_dir = float("inf")
        best_dist = float("inf")
        for centro in centroids:
            dir_delta = smallest_angular_distance(float(frame_data[idx, 4]), float(centro[4]))
            if dir_delta > cfg.tdirect:
                continue
            dist_delta = distance.euclidean(frame_data[idx, [2, 3]], [centro[2], centro[3]])
            if dist_delta <= cfg.tdist and (dir_delta < best_dir or dist_delta < best_dist):
                best_label = int(centro[1])
                best_dir = dir_delta
                best_dist = dist_delta
        if best_label is None:
            frame_data[idx, 7] = float(next_cluster_id)
            next_cluster_id += 1
        else:
            frame_data[idx, 7] = float(best_label)
    return next_cluster_id


def _distance_eval(frame_data: np.ndarray, centroids: Sequence[Sequence[float]]) -> List[List[float]]:
    """Compute per-cluster angular deviation from centroid direction."""
    eval_rows: List[List[float]] = []
    grouped = _group_by_cluster(frame_data)
    centroid_by_id = {int(c[1]): c for c in centroids}
    for cluster_id, members in grouped:
        centro = centroid_by_id.get(cluster_id)
        if centro is None or len(members) == 0:
            continue
        center_dir = float(centro[4])
        angular = [
            smallest_angular_distance(center_dir, float(member_dir))
            for member_dir in members[:, 4].tolist()
        ]
        eval_rows.append([float(cluster_id), float(np.mean(angular)), float(len(members))])
    return eval_rows


def arrange_per_cluster(centroids_per_frame: Sequence[Sequence[Sequence[float]]]) -> List[List[Any]]:
    """Convert frame-major centroid list into cluster-major layout."""
    grouped: Dict[int, List[List[float]]] = {}
    for frame_centroids in centroids_per_frame:
        for row in frame_centroids:
            cluster_id = int(row[1])
            grouped.setdefault(cluster_id, []).append(
                [row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]
            )
    return [[cluster_id, rows] for cluster_id, rows in sorted(grouped.items(), key=lambda x: x[0])]


def mask_centroids(centroids_per_frame: Sequence[Sequence[Sequence[float]]]) -> List[List[Any]]:
    """Fill missing centroid frames by linear interpolation per cluster."""
    arranged = arrange_per_cluster(centroids_per_frame)
    for cluster_entry in arranged:
        frames = cluster_entry[1]
        filled: List[List[float]] = []
        for i, row in enumerate(frames):
            if i == 0:
                filled.append(row)
                continue
            prev = filled[-1]
            gap = int(row[0] - prev[0])
            if gap > 1:
                for step in range(1, gap):
                    ratio = step / float(gap)
                    interp = [
                        prev[0] + step,
                        prev[1] + ratio * (row[1] - prev[1]),
                        prev[2] + ratio * (row[2] - prev[2]),
                        prev[3] + ratio * (row[3] - prev[3]),
                        prev[4] + ratio * (row[4] - prev[4]),
                        prev[5] + ratio * (row[5] - prev[5]),
                        prev[6] + ratio * (row[6] - prev[6]),
                        0.0,
                    ]
                    filled.append(interp)
            filled.append(row)
        cluster_entry[1] = filled
    return arranged


def arrange_per_frame(
    centroids_by_cluster: Sequence[Sequence[Any]],
    start: int,
    finish: int,
) -> List[List[List[float]]]:
    """Convert cluster-major centroids into frame-major centroids."""
    frame_result: List[List[List[float]]] = []
    for frame_no in range(start, finish):
        frame_rows: List[List[float]] = []
        for cluster_id, rows in centroids_by_cluster:
            for row in rows:
                if int(row[0]) == frame_no:
                    frame_rows.append(
                        [
                            float(row[0]),
                            float(cluster_id),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                            float(row[6]),
                            float(row[7]),
                        ]
                    )
                    break
        frame_result.append(frame_rows)
    return frame_result


def evaluation_per_cluster(
    centroids_by_frame: Sequence[Sequence[Sequence[float]]],
    distance_evaluation: Sequence[Tuple[int, Sequence[Sequence[float]]]],
) -> List[List[Any]]:
    """Return per-cluster distance evaluations across frames."""
    cluster_ids: set[int] = set()
    for frame_centroids in centroids_by_frame:
        for row in frame_centroids:
            cluster_ids.add(int(row[1]))
    output: List[List[Any]] = []
    for cluster_id in sorted(cluster_ids):
        values: List[List[Any]] = []
        for frame_no, eval_rows in distance_evaluation:
            for eval_row in eval_rows:
                if int(eval_row[0]) == cluster_id:
                    values.append([frame_no, eval_row])
                    break
        output.append([cluster_id, values])
    return output


def _compute_cluster_stats(eval_by_cluster: Sequence[Sequence[Any]]) -> Dict[str, float]:
    """Compute aggregate statistics from per-cluster evaluations."""
    avg_rows: List[Tuple[int, float, float]] = []
    for cluster_id, values in eval_by_cluster:
        if len(values) == 0:
            avg_rows.append((int(cluster_id), 0.0, 0.0))
            continue
        avg_distance = float(np.mean([float(v[1][1]) for v in values]))
        avg_members = float(np.mean([float(v[1][2]) for v in values]))
        avg_rows.append((int(cluster_id), avg_distance, avg_members))

    if len(avg_rows) == 0:
        return {"total_clusters": 0.0, "avg_distance": 0.0, "avg_distance_multi_member": 0.0}

    avg_distance = float(np.mean([x[1] for x in avg_rows]))
    multi = [x[1] for x in avg_rows if x[2] > 1.0]
    avg_distance_multi = float(np.mean(multi)) if len(multi) > 0 else 0.0
    return {
        "total_clusters": float(len(avg_rows)),
        "avg_distance": avg_distance,
        "avg_distance_multi_member": avg_distance_multi,
    }


def run_raw2cluster_pipeline(
    input_path: str | Path,
    start: int,
    finish: int,
    config: Optional[Raw2ClusterConfig] = None,
) -> Dict[str, Any]:
    """Run migrated raw-to-cluster pipeline.

    Args:
        input_path: Path to input raw tracks.
        start: Inclusive start frame.
        finish: Exclusive finish frame.
        config: Optional pipeline config.

    Returns:
        Dictionary with centroids, evaluations, per-frame members, and summary.
    """
    cfg = config or Raw2ClusterConfig()
    tracks_df = load_tracks(Path(input_path))
    data = prepare_data(tracks_df)
    initial_data = get_data_by_frame(data, start)
    id_to_cluster = _cluster_initial_frame(initial_data, cfg)
    next_cluster_id = (max(id_to_cluster.values()) + 1) if len(id_to_cluster) > 0 else 0

    centroids_by_frame: List[List[List[float]]] = []
    distance_evals: List[Tuple[int, List[List[float]]]] = []
    member_centroids: List[List[Tuple[int, np.ndarray]]] = []

    prev_centroids: List[List[float]] = []
    for frame_no in range(start, finish):
        frame_data = get_data_by_frame(data, frame_no)
        if len(frame_data) == 0:
            centroids_by_frame.append([])
            distance_evals.append((frame_no, []))
            member_centroids.append([])
            continue

        for row_idx in range(len(frame_data)):
            ped_id = int(frame_data[row_idx, 1])
            if ped_id in id_to_cluster:
                frame_data[row_idx, 7] = float(id_to_cluster[ped_id])
            else:
                frame_data[row_idx, 7] = -1.0

        if frame_no == start:
            centroids = _compute_centroids(frame_data)
        else:
            next_cluster_id = _assign_unknowns_to_centroids(frame_data, prev_centroids, next_cluster_id, cfg)
            centroids = _compute_centroids(frame_data)

        grouped = _group_by_cluster(frame_data)
        if frame_no == start or (frame_no - start) % cfg.eval_frame_interval == 0:
            frame_eval = _distance_eval(frame_data, centroids)
        else:
            frame_eval = []
        distance_evals.append((frame_no, frame_eval))
        centroids_by_frame.append(centroids)
        member_centroids.append(grouped)
        prev_centroids = centroids

        for row_idx in range(len(frame_data)):
            id_to_cluster[int(frame_data[row_idx, 1])] = int(frame_data[row_idx, 7])

    masked = mask_centroids(centroids_by_frame)
    arranged_frames = arrange_per_frame(masked, start, finish)
    per_cluster_eval = evaluation_per_cluster(arranged_frames, distance_evals)
    cluster_stats = _compute_cluster_stats(per_cluster_eval)

    result: Dict[str, Any] = {
        "centroids": arranged_frames,
        "distance_eval": per_cluster_eval,
        "member_centroids": member_centroids,
        "cluster_stats": cluster_stats,
    }

    if cfg.output_csv is not None:
        rows: List[List[float]] = []
        for frame_rows in arranged_frames:
            rows.extend(frame_rows)
        out_df = pd.DataFrame(
            rows, columns=["frame", "cluster_id", "x", "y", "va", "vx", "vy", "radius", "count"]
        )
        cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(cfg.output_csv, index=False)

    return result


def visualize_raw_vs_cluster(
    input_path: str | Path,
    pipeline_result: Dict[str, Any],
    start: int,
    finish: int,
    out_path: str | Path,
) -> Path:
    """Create side-by-side raw trajectories vs cluster trajectories figure.

    Args:
        input_path: Input trajectories file path.
        pipeline_result: Output dict from ``run_raw2cluster_pipeline``.
        start: Inclusive start frame for plotting.
        finish: Exclusive finish frame for plotting.
        out_path: Figure output path.

    Returns:
        Saved figure path.
    """
    out = Path(out_path)
    raw_df = load_tracks(Path(input_path))
    raw_df = raw_df[(raw_df["frame"] >= start) & (raw_df["frame"] < finish)]

    centroids = pipeline_result["centroids"]
    cluster_rows: List[List[float]] = []
    for frame_rows in centroids:
        cluster_rows.extend(frame_rows)
    cluster_df = pd.DataFrame(
        cluster_rows,
        columns=["frame", "cluster_id", "x", "y", "va", "vx", "vy", "radius", "count"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_raw, ax_cluster = axes
    scene_name = _infer_scene_name(input_path)

    for ped_id, sub_df in raw_df.groupby("id"):
        ax_raw.plot(sub_df["x"], sub_df["y"], linewidth=1.0, alpha=0.9)
        # ax_raw.scatter(sub_df["x"], sub_df["y"], s=10, alpha=0.5)
    ax_raw.set_title(f"Raw trajectories for scene {scene_name}, #pedestrians={len(raw_df['id'].unique())}")
    ax_raw.set_xlabel("x")
    ax_raw.set_ylabel("y")
    ax_raw.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if not cluster_df.empty:
        for cluster_id, sub_df in cluster_df.groupby("cluster_id"):
            ax_cluster.plot(sub_df["x"], sub_df["y"], linewidth=1.4, alpha=0.95)
            # ax_cluster.scatter(sub_df["x"], sub_df["y"], s=10, alpha=0.5)
        ax_cluster.set_title(f"Cluster trajectories for scene {scene_name}, #clusters={len(cluster_df['cluster_id'].unique())}")
    else:
        ax_cluster.set_title("Cluster trajectories (empty)")
    ax_cluster.set_xlabel("x")
    ax_cluster.set_ylabel("y")
    ax_cluster.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def process_motsynth_batch(
    data_root: str | Path,
    output_root: str | Path,
    start: int,
    finish: int,
    config: Optional[Raw2ClusterConfig] = None,
    make_comparison_figure: bool = True,
    scene_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Process all MOTSynth scenes under ``mot_annotations``.

    Args:
        data_root: Root data path, expected to contain ``motsynth/mot_annotations``.
        output_root: Output folder for csv and optional figures.
        start: Inclusive start frame.
        finish: Exclusive finish frame.
        config: Optional pipeline config overrides.
        make_comparison_figure: Whether to save raw vs cluster comparison image.
        scene_ids: Optional scene filters (e.g., ["000", "001"]). None means all.

    Returns:
        List of per-scene summary dicts.
    """
    cfg = config or Raw2ClusterConfig()
    root = Path(data_root)
    out_root = Path(output_root)
    gt_files = sorted((root / "motsynth" / "mot_annotations").glob("*/gt/gt.txt"))
    scene_filter = set(scene_ids) if scene_ids is not None else None
    results: List[Dict[str, Any]] = []

    for gt_file in gt_files:
        scene = gt_file.parents[1].name
        if scene_filter is not None and scene not in scene_filter:
            continue
        scene_csv = out_root / "csv" / f"{scene}.csv"
        run_cfg = Raw2ClusterConfig(
            n_initial_cluster=cfg.n_initial_cluster,
            tdist=cfg.tdist,
            tdirect=cfg.tdirect,
            eval_interval=cfg.eval_interval,
            output_csv=scene_csv,
            eval_frame_interval=cfg.eval_frame_interval,
        )
        result = run_raw2cluster_pipeline(gt_file, start=start, finish=finish, config=run_cfg)
        if make_comparison_figure:
            fig_path = out_root / "figures" / f"{scene}_raw_vs_cluster.png"
            visualize_raw_vs_cluster(
                input_path=gt_file,
                pipeline_result=result,
                start=start,
                finish=finish,
                out_path=fig_path,
            )
        results.append(
            {
                "scene": scene,
                "input": str(gt_file),
                "output_csv": str(scene_csv),
                "cluster_stats": result["cluster_stats"],
            }
        )
    return results


def _build_config_from_args(args: argparse.Namespace) -> Raw2ClusterConfig:
    """Build pipeline config from CLI args."""
    return Raw2ClusterConfig(
        n_initial_cluster=args.n_initial_cluster,
        tdist=args.tdist,
        tdirect=args.tdirect,
        eval_interval=args.eval_interval,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        eval_frame_interval=args.eval_frame_interval,
    )


def _infer_scene_name(input_path: str | Path) -> str:
    """Infer scene id/name from a MOTSynth-like gt path."""
    path_obj = Path(input_path)
    if path_obj.name == "gt.txt" and len(path_obj.parents) >= 3:
        return path_obj.parents[1].name
    return path_obj.stem


def _default_single_scene_outputs(input_path: str | Path) -> Tuple[Path, Path]:
    """Build default csv and figure output paths for one scene."""
    scene_name = _infer_scene_name(input_path)
    project_root = Path(__file__).resolve().parent
    data_root = project_root.parent / "data" / "motsynth_cluster"
    figure_root = project_root.parent / "figures"
    csv_path = data_root / f"{scene_name}.csv"
    figure_path = figure_root / f"{scene_name}_raw_vs_cluster.png"
    return csv_path, figure_path


def main() -> None:
    """CLI entrypoint for single-scene or batch processing."""
    parser = argparse.ArgumentParser(description="Run raw-to-cluster trajectory processing.")
    parser.add_argument("--input", type=str, default=None, help="Single input file path.")
    parser.add_argument("--start", type=int, default=75, help="Inclusive start frame.")
    parser.add_argument("--finish", type=int, default=950, help="Exclusive finish frame.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Single-scene csv output. Default: ../data/motsynth_cluster/<scene>.csv",
    )
    parser.add_argument(
        "--comparison-figure",
        type=str,
        default=None,
        help="Single-scene raw-vs-cluster figure output. Default: ../figures/<scene>_raw_vs_cluster.png",
    )
    parser.add_argument(
        "--batch-data-root",
        type=str,
        default=None,
        help="Batch mode root with motsynth/mot_annotations.",
    )
    parser.add_argument(
        "--batch-output-root",
        type=str,
        default=None,
        help="Batch output root (csv/ and figures/).",
    )
    parser.add_argument(
        "--batch-scenes",
        type=str,
        default=None,
        help="Optional comma-separated scenes for batch mode, e.g. 000,001.",
    )
    parser.add_argument("--tdist", type=float, default=110.0, help="Spatial threshold.")
    parser.add_argument("--tdirect", type=float, default=50.0, help="Direction threshold.")
    parser.add_argument(
        "--n-initial-cluster",
        type=int,
        default=8,
        help="Initial agglomerative clusters on start frame.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Compatibility placeholder from original pipeline config.",
    )
    parser.add_argument(
        "--eval-frame-interval",
        type=int,
        default=1,
        help="How often to compute distance eval rows.",
    )
    args = parser.parse_args()
    cfg = _build_config_from_args(args)

    if args.batch_data_root:
        if not args.batch_output_root:
            raise ValueError("--batch-output-root is required when --batch-data-root is set")
        scene_ids = None
        if args.batch_scenes:
            scene_ids = [scene.strip() for scene in args.batch_scenes.split(",") if scene.strip()]
        summaries = process_motsynth_batch(
            data_root=args.batch_data_root,
            output_root=args.batch_output_root,
            start=args.start,
            finish=args.finish,
            config=cfg,
            make_comparison_figure=True,
            scene_ids=scene_ids,
        )
        print(f"Processed scenes: {len(summaries)}")
        return

    if not args.input:
        raise ValueError("Provide --input for single-scene mode")

    default_csv_path, default_figure_path = _default_single_scene_outputs(args.input)
    args.output_csv = args.output_csv or str(default_csv_path)
    args.comparison_figure = args.comparison_figure or str(default_figure_path)
    cfg = _build_config_from_args(args)

    result = run_raw2cluster_pipeline(args.input, start=args.start, finish=args.finish, config=cfg)
    if args.comparison_figure:
        visualize_raw_vs_cluster(
            input_path=args.input,
            pipeline_result=result,
            start=args.start,
            finish=args.finish,
            out_path=args.comparison_figure,
        )
    print(f"Total clusters: {result['cluster_stats']['total_clusters']}")


if __name__ == "__main__":
    main()
