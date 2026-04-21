from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from raw2cluster import (
    Raw2ClusterConfig,
    extract_ped_cluster_labels,
    run_raw2cluster_pipeline_from_df,
)


@dataclass
class DynamicClusterBuildResult:
    """Output container for clustered-pool dataset construction."""

    trajs: List[torch.Tensor]
    masks: List[torch.Tensor]
    filename_list: List[str]
    frames_list: List[List[int]]
    pedestrians_list: List[int]
    filename2idxs_dict: Dict[str, List[int]]
    idx2filename_dict: Dict[int, str]
    pool_indices_by_fold: List[List[int]]
    valid_indices_by_fold: List[List[int]]
    cluster_meta_by_fold: List[Dict[int, Dict[str, Any]]]


def load_dc_config(dc_config_path: str | Path) -> Dict[str, Any]:
    """Load dynamic-clustering YAML config from file."""
    with open(dc_config_path, mode="rt", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj)
    if loaded is None:
        return {}
    return loaded


def filter_valid_indices_with_pool(
    valid_indices_by_fold: Sequence[Sequence[int]],
    pool_indices_by_fold: Sequence[Sequence[int]],
    filename2idxs_dict: Dict[str, List[int]],
    idx2filename_dict: Dict[int, str],
    min_prompt_num: int,
) -> List[List[int]]:
    """Filter valid samples that do not have enough in-file pool prompts."""
    filtered: List[List[int]] = []
    for fold_idx, valid_indices in enumerate(valid_indices_by_fold):
        pool_indices_set = set(pool_indices_by_fold[fold_idx])
        kept: List[int] = []
        for idx in valid_indices:
            same_file_indices = filename2idxs_dict[idx2filename_dict[idx]]
            prompt_count = sum(
                1 for candidate_idx in same_file_indices if candidate_idx in pool_indices_set and candidate_idx != idx
            )
            if prompt_count >= min_prompt_num:
                kept.append(idx)
        filtered.append(kept)
    return filtered


def build_trajs_dc_aligned_per_fold(
    trajs: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    filename_list: Sequence[str],
    pool_indices_by_fold: Sequence[Sequence[int]],
    dc_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, List[torch.Tensor]]], List[Dict[int, Dict[str, Any]]]]:
    """Build per-fold trajectory and mask lists aligned with global indices.

    For each fold, non-pool indices keep a clone of the raw trajectory and
    mask. Each pool sample is replaced by the centroid trajectory (and its
    mask) of its DC cluster in pixel coordinates.

    Args:
        trajs: Original per-sample trajectories ``(N, T, 1, 3)``.
        masks: Masks matching ``trajs``.
        filename_list: Scene name per global index.
        pool_indices_by_fold: Pool index lists (one per cross-validation fold).
        dc_cfg: Dynamic clustering parameters (see ``_cluster_pool_for_file``).

    Returns:
        A tuple of:
        - One dict per fold with keys ``"trajs"`` and ``"masks"``: lists of
          the same length as ``trajs`` with matching tensor shapes.
        - Fold-wise cluster metadata with original members and cluster weight.
    """
    num_samples = len(trajs)
    trajs_dc_by_fold: List[Dict[str, List[torch.Tensor]]] = []
    cluster_meta_by_fold: List[Dict[int, Dict[str, Any]]] = []

    for pool_indices in pool_indices_by_fold:
        aligned_traj: List[torch.Tensor] = [trajs[i].clone() for i in range(num_samples)]
        aligned_mask: List[torch.Tensor] = [masks[i].clone() for i in range(num_samples)]
        pool_by_file: Dict[str, List[int]] = defaultdict(list)
        fold_meta: Dict[int, Dict[str, Any]] = {}
        next_meta_id = 0
        for sample_idx in pool_indices:
            pool_by_file[filename_list[sample_idx]].append(sample_idx)

        for file_name, file_pool_indices in pool_by_file.items():
            clustered_trajs, clustered_masks, cluster_members = _cluster_pool_for_file(
                trajs=trajs,
                masks=masks,
                file_pool_indices=file_pool_indices,
                dc_cfg=dc_cfg,
            )
            if len(clustered_trajs) == 0:
                continue
            for local_idx, member_indices in cluster_members.items():
                centroid_t = clustered_trajs[local_idx]
                centroid_m = clustered_masks[local_idx]
                for global_idx in member_indices:
                    aligned_traj[global_idx] = centroid_t.clone()
                    aligned_mask[global_idx] = centroid_m.clone()
                fold_meta[next_meta_id] = {
                    "file_name": file_name,
                    "member_original_indices": sorted(member_indices),
                    "cluster_weight": int(len(member_indices)),
                }
                next_meta_id += 1

        trajs_dc_by_fold.append({"trajs": aligned_traj, "masks": aligned_mask})
        cluster_meta_by_fold.append(fold_meta)

    return trajs_dc_by_fold, cluster_meta_by_fold


def build_clustered_pool_dataset(
    trajs: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    filename_list: Sequence[str],
    frames_list: Sequence[Sequence[int]],
    pedestrians_list: Sequence[int],
    pool_indices_by_fold: Sequence[Sequence[int]],
    valid_indices_by_fold: Sequence[Sequence[int]],
    dc_cfg: Dict[str, Any],
    min_prompt_num: int,
) -> DynamicClusterBuildResult:
    """Build a clustered-pool dataset while keeping validation trajectories raw."""
    new_trajs: List[torch.Tensor] = [traj.clone() for traj in trajs]
    new_masks: List[torch.Tensor] = [mask.clone() for mask in masks]
    new_filename_list: List[str] = list(filename_list)
    new_frames_list: List[List[int]] = [list(frame_seq) for frame_seq in frames_list]
    new_pedestrians_list: List[int] = list(pedestrians_list)
    next_index = len(new_trajs)

    fold_pool_indices: List[List[int]] = []
    fold_cluster_meta: List[Dict[int, Dict[str, Any]]] = []

    for fold_idx, pool_indices in enumerate(pool_indices_by_fold):
        pool_by_file: Dict[str, List[int]] = defaultdict(list)
        for sample_idx in pool_indices:
            pool_by_file[filename_list[sample_idx]].append(sample_idx)

        current_fold_pool: List[int] = []
        current_meta: Dict[int, Dict[str, Any]] = {}

        for file_name, file_pool_indices in pool_by_file.items():
            clustered_trajs, clustered_masks, cluster_members = _cluster_pool_for_file(
                trajs=trajs,
                masks=masks,
                file_pool_indices=file_pool_indices,
                dc_cfg=dc_cfg,
            )

            for cluster_local_idx, (cluster_traj, cluster_mask) in enumerate(
                zip(clustered_trajs, clustered_masks)
            ):
                member_indices = cluster_members.get(cluster_local_idx, [])
                cluster_global_idx = next_index
                next_index += 1

                new_trajs.append(cluster_traj)
                new_masks.append(cluster_mask)
                new_filename_list.append(file_name)
                new_frames_list.append(list(range(cluster_traj.shape[1])))
                # Negative ids avoid collision with original pedestrian ids.
                new_pedestrians_list.append(-1_000_000 - cluster_global_idx)

                current_fold_pool.append(cluster_global_idx)
                current_meta[cluster_global_idx] = {
                    "file_name": file_name,
                    "member_original_indices": member_indices,
                    "cluster_weight": int(len(member_indices)),
                }

        fold_pool_indices.append(current_fold_pool)
        fold_cluster_meta.append(current_meta)

    filename2idxs_dict: Dict[str, List[int]] = defaultdict(list)
    idx2filename_dict: Dict[int, str] = {}
    for idx, file_name in enumerate(new_filename_list):
        filename2idxs_dict[file_name].append(idx)
        idx2filename_dict[idx] = file_name

    filtered_valid_indices = filter_valid_indices_with_pool(
        valid_indices_by_fold=valid_indices_by_fold,
        pool_indices_by_fold=fold_pool_indices,
        filename2idxs_dict=filename2idxs_dict,
        idx2filename_dict=idx2filename_dict,
        min_prompt_num=min_prompt_num,
    )

    return DynamicClusterBuildResult(
        trajs=new_trajs,
        masks=new_masks,
        filename_list=new_filename_list,
        frames_list=new_frames_list,
        pedestrians_list=new_pedestrians_list,
        filename2idxs_dict=dict(filename2idxs_dict),
        idx2filename_dict=idx2filename_dict,
        pool_indices_by_fold=fold_pool_indices,
        valid_indices_by_fold=filtered_valid_indices,
        cluster_meta_by_fold=fold_cluster_meta,
    )


def build_cluster_pool_per_fold(
    trajs: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    filename_list: Sequence[str],
    pool_indices_by_fold: Sequence[Sequence[int]],
    dc_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build unique centroid pool datasets for each fold.

    Returns one bundle per fold containing centroid trajectories/masks and
    mappings between raw indices and cluster indices.
    """
    fold_bundles: List[Dict[str, Any]] = []
    for pool_indices in pool_indices_by_fold:
        pool_by_file: Dict[str, List[int]] = defaultdict(list)
        for sample_idx in pool_indices:
            pool_by_file[filename_list[sample_idx]].append(sample_idx)

        cluster_trajs: List[torch.Tensor] = []
        cluster_masks: List[torch.Tensor] = []
        cluster_filename2idxs_dict: Dict[str, List[int]] = defaultdict(list)
        cluster_idx2filename_dict: Dict[int, str] = {}
        cluster_meta: Dict[int, Dict[str, Any]] = {}
        raw_idx_to_cluster_idx: Dict[int, int] = {}
        next_cluster_idx = 0
        scene_stats: List[Dict[str, Any]] = []

        for file_name, file_pool_indices in pool_by_file.items():
            clustered_trajs, clustered_masks, cluster_members = _cluster_pool_for_file(
                trajs=trajs,
                masks=masks,
                file_pool_indices=file_pool_indices,
                dc_cfg=dc_cfg,
            )
            for local_idx, centroid_traj in enumerate(clustered_trajs):
                centroid_mask = clustered_masks[local_idx]
                member_indices = sorted(cluster_members.get(local_idx, []))
                cluster_idx = next_cluster_idx
                next_cluster_idx += 1

                cluster_trajs.append(centroid_traj)
                cluster_masks.append(centroid_mask)
                cluster_filename2idxs_dict[file_name].append(cluster_idx)
                cluster_idx2filename_dict[cluster_idx] = file_name
                cluster_meta[cluster_idx] = {
                    "file_name": file_name,
                    "member_original_indices": member_indices,
                    "cluster_weight": int(len(member_indices)),
                }
                for raw_idx in member_indices:
                    raw_idx_to_cluster_idx[int(raw_idx)] = cluster_idx

            raw_count = int(len(file_pool_indices))
            cluster_count = int(len(clustered_trajs))
            reduction_pct = 0.0
            if raw_count > 0:
                reduction_pct = 100.0 * float(raw_count - cluster_count) / float(raw_count)
            scene_stats.append(
                {
                    "file_name": file_name,
                    "raw_count": raw_count,
                    "cluster_count": cluster_count,
                    "reduction_pct": reduction_pct,
                }
            )

        fold_bundles.append(
            {
                "trajs": cluster_trajs,
                "masks": cluster_masks,
                "filename2idxs_dict": dict(cluster_filename2idxs_dict),
                "idx2filename_dict": cluster_idx2filename_dict,
                "cluster_meta": cluster_meta,
                "raw_idx_to_cluster_idx": raw_idx_to_cluster_idx,
                "scene_stats": scene_stats,
                "raw_pool_count": int(len(pool_indices)),
                "cluster_pool_count": int(len(cluster_trajs)),
            }
        )

    return fold_bundles


def _cluster_pool_for_file(
    trajs: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    file_pool_indices: Sequence[int],
    dc_cfg: Dict[str, Any],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[int, List[int]]]:
    """Cluster pool trajectories in one scene file and return centroid samples."""
    if len(file_pool_indices) == 0:
        return [], [], {}

    first_idx = file_pool_indices[0]
    time_steps = int(trajs[first_idx].shape[1])

    mot_rows: List[Dict[str, float]] = []
    ped_lookup: Dict[int, int] = {}
    for local_ped_id, sample_idx in enumerate(file_pool_indices):
        ped_lookup[local_ped_id] = int(sample_idx)
        traj_tensor = trajs[sample_idx]
        mask_tensor = masks[sample_idx]
        for t in range(time_steps):
            if bool(mask_tensor[0, t, 0]):
                mot_rows.append(
                    {
                        "frame": int(t),
                        "id": int(local_ped_id),
                        "x": float(traj_tensor[0, t, 0, 0].item()),
                        "y": float(traj_tensor[0, t, 0, 1].item()),
                    }
                )

    if len(mot_rows) == 0:
        return [], [], {}

    tracks_df = pd.DataFrame(mot_rows)
    tuned_tdist, tuned_tdirect = _get_scene_adaptive_thresholds(
        tracks_df=tracks_df,
        base_tdist=float(dc_cfg.get("tdist", 110.0)),
        base_tdirect=float(dc_cfg.get("tdirect", 50.0)),
        dc_cfg=dc_cfg,
    )
    config = Raw2ClusterConfig(
        n_initial_cluster=int(dc_cfg.get("n_initial_cluster", 8)),
        tdist=tuned_tdist,
        tdirect=tuned_tdirect,
        eval_interval=int(dc_cfg.get("eval_interval", 10)),
        eval_frame_interval=int(dc_cfg.get("eval_frame_interval", 1)),
    )

    pipeline_result = run_raw2cluster_pipeline_from_df(
        tracks_df=tracks_df,
        start=0,
        finish=time_steps,
        config=config,
    )

    cluster_to_members = _build_cluster_members(
        pipeline_result=pipeline_result,
        start=0,
        finish=time_steps,
        ped_lookup=ped_lookup,
    )

    centroids = pipeline_result["centroids"]
    cluster_series: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for frame_rows in centroids:
        for row in frame_rows:
            frame_no = int(row[0])
            cluster_id = int(row[1])
            x_val = float(row[2])
            y_val = float(row[3])
            cluster_series[cluster_id][frame_no] = (x_val, y_val)

    cluster_ids_sorted = sorted(cluster_series.keys())
    cluster_id_to_local = {cluster_id: local_idx for local_idx, cluster_id in enumerate(cluster_ids_sorted)}

    clustered_trajs: List[torch.Tensor] = []
    clustered_masks: List[torch.Tensor] = []
    clustered_members: Dict[int, List[int]] = {}
    for cluster_id in cluster_ids_sorted:
        traj_template = trajs[first_idx].clone()
        mask_template = masks[first_idx].clone()
        traj_template.zero_()
        mask_template.zero_()

        for frame_no, pos in cluster_series[cluster_id].items():
            if 0 <= frame_no < time_steps:
                traj_template[0, frame_no, 0, 0] = pos[0]
                traj_template[0, frame_no, 0, 1] = pos[1]
                traj_template[0, frame_no, 0, 2] = 0.0
                mask_template[0, frame_no, 0] = True

        local_cluster_idx = cluster_id_to_local[cluster_id]
        clustered_trajs.append(traj_template)
        clustered_masks.append(mask_template)
        clustered_members[local_cluster_idx] = sorted(cluster_to_members.get(cluster_id, []))

    covered_indices = set()
    for member_indices in clustered_members.values():
        covered_indices.update(member_indices)

    uncovered_indices = sorted(set(file_pool_indices) - covered_indices)
    for uncovered_idx in uncovered_indices:
        singleton_traj = trajs[uncovered_idx].clone()
        singleton_mask = masks[uncovered_idx].clone()
        singleton_local_idx = len(clustered_trajs)
        clustered_trajs.append(singleton_traj)
        clustered_masks.append(singleton_mask)
        clustered_members[singleton_local_idx] = [int(uncovered_idx)]

    return clustered_trajs, clustered_masks, clustered_members


def _get_scene_adaptive_thresholds(
    tracks_df: pd.DataFrame,
    base_tdist: float,
    base_tdirect: float,
    dc_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """Return scene-adaptive tdist/tdirect if enabled in config."""
    use_adaptive = bool(dc_cfg.get("adaptive_scene_thresholds", False))
    if not use_adaptive:
        return base_tdist, base_tdirect

    dist_percentile = float(dc_cfg.get("adaptive_dist_percentile", 50.0))
    dist_scale = float(dc_cfg.get("adaptive_dist_scale", 1.0))
    min_tdist = float(dc_cfg.get("adaptive_tdist_min", 20.0))
    max_tdist = float(dc_cfg.get("adaptive_tdist_max", 250.0))

    speed_percentile = float(dc_cfg.get("adaptive_speed_percentile", 50.0))
    speed_reference = float(dc_cfg.get("adaptive_speed_reference", 4.0))
    direction_scale = float(dc_cfg.get("adaptive_direction_scale", 1.0))
    min_tdirect = float(dc_cfg.get("adaptive_tdirect_min", 15.0))
    max_tdirect = float(dc_cfg.get("adaptive_tdirect_max", 90.0))

    frame_distances: List[float] = []
    for _, frame_df in tracks_df.groupby("frame"):
        coords = frame_df[["x", "y"]].to_numpy(dtype=np.float64)
        if coords.shape[0] < 2:
            continue
        deltas = coords[:, None, :] - coords[None, :, :]
        pairwise = np.sqrt(np.sum(deltas * deltas, axis=2))
        upper = pairwise[np.triu_indices(coords.shape[0], k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size > 0:
            frame_distances.extend(upper.tolist())

    if len(frame_distances) > 0:
        dist_stat = float(np.percentile(frame_distances, dist_percentile))
        tuned_tdist = float(np.clip(dist_stat * dist_scale, min_tdist, max_tdist))
    else:
        tuned_tdist = base_tdist

    speed_values: List[float] = []
    for _, ped_df in tracks_df.sort_values(["id", "frame"]).groupby("id"):
        points = ped_df[["x", "y"]].to_numpy(dtype=np.float64)
        if points.shape[0] < 2:
            continue
        diff = points[1:] - points[:-1]
        speeds = np.sqrt(np.sum(diff * diff, axis=1))
        speeds = speeds[np.isfinite(speeds)]
        if speeds.size > 0:
            speed_values.extend(speeds.tolist())

    if len(speed_values) > 0:
        speed_stat = float(np.percentile(speed_values, speed_percentile))
        speed_ratio = speed_stat / max(speed_reference, 1e-6)
        tuned_tdirect = float(
            np.clip(base_tdirect * speed_ratio * direction_scale, min_tdirect, max_tdirect)
        )
    else:
        tuned_tdirect = base_tdirect

    return tuned_tdist, tuned_tdirect


def _build_cluster_members(
    pipeline_result: Dict[str, Any],
    start: int,
    finish: int,
    ped_lookup: Dict[int, int],
) -> Dict[int, List[int]]:
    """Infer original sample memberships per cluster from per-frame labels."""
    labels = extract_ped_cluster_labels(pipeline_result, start=start, finish=finish)
    ped_votes: Dict[int, Counter[int]] = defaultdict(Counter)
    for frame_labels in labels.values():
        for local_ped_id, cluster_id in frame_labels.items():
            ped_votes[int(local_ped_id)][int(cluster_id)] += 1

    cluster_to_members: Dict[int, List[int]] = defaultdict(list)
    for local_ped_id, counter in ped_votes.items():
        if len(counter) == 0:
            continue
        winner_cluster_id = counter.most_common(1)[0][0]
        cluster_to_members[winner_cluster_id].append(ped_lookup[local_ped_id])

    return dict(cluster_to_members)
