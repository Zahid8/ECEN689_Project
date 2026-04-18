import argparse
import json
import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from preprocess import (
    compute_stats_traj,
    compute_sim_matrix,
    compute_trajectory_similarity,
    infer_r_stride,
    load_data as load_raw_data,
    load_processed_data as load_processed_split,
    pool_valid_split,
    save_data,
)
from utils.run_logging import finalize_run_logging, start_run_logging

warnings.simplefilter("ignore")

EPS = 1e-6
CENTROID_ID_OFFSET = 1_000_000


@dataclass
class PedestrianState:
    ped_id: int
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    cluster_id: Optional[int]


@dataclass
class ClusterRuntime:
    cluster_id: int
    created_frame_idx: int
    members: set = field(default_factory=set)
    centroid_by_frame: Dict[int, np.ndarray] = field(default_factory=dict)
    direction_by_frame: Dict[int, float] = field(default_factory=dict)
    member_history: Dict[int, List[int]] = field(default_factory=dict)
    size_history: Dict[int, int] = field(default_factory=dict)
    last_nonempty_frame_idx: int = 0
    active: bool = True


def smallest_angular_distance(theta_a: float, theta_b: float) -> float:
    """Paper uses smallest angular distance for direction similarity."""
    diff = (theta_a - theta_b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(diff))


def _latest_value(history: Dict[int, np.ndarray], frame_idx: int):
    keys = [k for k in history.keys() if k <= frame_idx]
    if not keys:
        return None
    return history[max(keys)]


def _latest_angle(history: Dict[int, float], frame_idx: int) -> float:
    keys = [k for k in history.keys() if k <= frame_idx]
    if not keys:
        return 0.0
    return history[max(keys)]


def _pairwise_distances(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def compute_lof_scores(features: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_samples = features.shape[0]
    if n_samples <= 2:
        return np.ones(n_samples, dtype=np.float32)

    n_neighbors = max(2, min(n_neighbors, n_samples - 1))
    dists = _pairwise_distances(features)
    sorted_idx = np.argsort(dists, axis=1)

    k_dist = np.zeros(n_samples, dtype=np.float32)
    neighborhoods: List[np.ndarray] = []

    for i in range(n_samples):
        neighbor_order = sorted_idx[i][1 : n_neighbors + 1]
        if len(neighbor_order) == 0:
            neighborhoods.append(np.array([], dtype=np.int64))
            k_dist[i] = 0.0
            continue
        k_dist[i] = float(dists[i, neighbor_order[-1]])
        neighbors = np.where((dists[i] <= k_dist[i] + EPS) & (np.arange(n_samples) != i))[0]
        if len(neighbors) == 0:
            neighbors = neighbor_order
        neighborhoods.append(neighbors)

    lrd = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        neighbors = neighborhoods[i]
        if len(neighbors) == 0:
            lrd[i] = 0.0
            continue
        reach_dists = []
        for o in neighbors:
            reach_dists.append(max(k_dist[o], float(dists[i, o])))
        mean_reach = float(np.mean(reach_dists))
        lrd[i] = 1.0 / (mean_reach + EPS)

    lof = np.ones(n_samples, dtype=np.float32)
    for i in range(n_samples):
        neighbors = neighborhoods[i]
        if len(neighbors) == 0 or lrd[i] <= EPS:
            lof[i] = 1.0
            continue
        lof[i] = float(np.mean(lrd[neighbors] / (lrd[i] + EPS)))

    return lof


def threshold_agglomerative(
    ped_ids: Sequence[int],
    distance_fn,
    threshold: float,
) -> List[List[int]]:
    """Single-link thresholded agglomerative clustering via union-find components."""
    ped_ids = list(ped_ids)
    n = len(ped_ids)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if distance_fn(ped_ids[i], ped_ids[j]) <= threshold:
                union(i, j)

    groups = defaultdict(list)
    for i, ped_id in enumerate(ped_ids):
        groups[find(i)].append(ped_id)
    return list(groups.values())


def agglomerative_cluster_by_direction(
    ped_states: Dict[int, PedestrianState],
    direction_thresh_deg: float,
) -> List[List[int]]:
    ped_ids = list(ped_states.keys())

    def direction_distance(a: int, b: int) -> float:
        return smallest_angular_distance(ped_states[a].theta, ped_states[b].theta)

    return threshold_agglomerative(ped_ids, direction_distance, direction_thresh_deg)


def agglomerative_cluster_by_location(
    ped_states: Dict[int, PedestrianState],
    distance_thresh_px: float,
) -> List[List[int]]:
    ped_ids = list(ped_states.keys())

    def location_distance(a: int, b: int) -> float:
        ax, ay = ped_states[a].x, ped_states[a].y
        bx, by = ped_states[b].x, ped_states[b].y
        return float(math.hypot(ax - bx, ay - by))

    return threshold_agglomerative(ped_ids, location_distance, distance_thresh_px)


def nested_initial_clustering(
    active_states: Dict[int, PedestrianState],
    direction_thresh_deg: float,
    distance_thresh_px: float,
) -> List[List[int]]:
    """Paper: direction clustering first, then location clustering inside each direction cluster."""
    clusters: List[List[int]] = []
    direction_clusters = agglomerative_cluster_by_direction(
        active_states,
        direction_thresh_deg,
    )
    for direction_cluster in direction_clusters:
        nested_states = {pid: active_states[pid] for pid in direction_cluster}
        location_clusters = agglomerative_cluster_by_location(
            nested_states,
            distance_thresh_px,
        )
        clusters.extend(location_clusters)
    return clusters


def build_cluster_feature_matrix(
    cluster_members: Sequence[int],
    states: Dict[int, PedestrianState],
) -> Tuple[List[int], np.ndarray]:
    features = []
    member_ids = []
    for member_id in cluster_members:
        st = states.get(member_id)
        if st is None:
            continue
        member_ids.append(member_id)
        features.append(
            [
                st.x,
                st.y,
                math.cos(st.theta),
                math.sin(st.theta),
            ]
        )

    if not features:
        return [], np.empty((0, 4), dtype=np.float32)

    feats = np.asarray(features, dtype=np.float32)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std < EPS] = 1.0
    feats = (feats - mean) / std
    return member_ids, feats


def evaluate_cluster_members_with_lof(
    cluster_members: Sequence[int],
    states: Dict[int, PedestrianState],
    contamination: float,
    n_neighbors_ratio: float,
) -> List[int]:
    if len(cluster_members) < 3:
        return []

    member_ids, feats = build_cluster_feature_matrix(cluster_members, states)
    if len(member_ids) < 3:
        return []

    n_neighbors = int(math.ceil(n_neighbors_ratio * len(member_ids)))
    n_neighbors = max(2, min(n_neighbors, len(member_ids) - 1))

    lof_scores = compute_lof_scores(feats, n_neighbors=n_neighbors)
    n_outliers = int(math.ceil(contamination * len(member_ids)))
    n_outliers = max(1, min(n_outliers, len(member_ids) - 1))

    outlier_order = np.argsort(lof_scores)[::-1][:n_outliers]
    outlier_ids = [member_ids[idx] for idx in outlier_order if lof_scores[idx] > 1.0]
    return outlier_ids


def find_nearest_compatible_cluster(
    outlier_state: PedestrianState,
    active_clusters: Dict[int, ClusterRuntime],
    frame_idx: int,
    direction_thresh_deg: float,
    distance_thresh_px: float,
    skip_cluster_id: Optional[int] = None,
) -> Optional[int]:
    best_cluster_id = None
    best_distance = float("inf")

    for cluster_id, cluster in active_clusters.items():
        if not cluster.active:
            continue
        if skip_cluster_id is not None and cluster_id == skip_cluster_id:
            continue

        cluster_centroid = _latest_value(cluster.centroid_by_frame, frame_idx)
        if cluster_centroid is None:
            continue

        cluster_theta = _latest_angle(cluster.direction_by_frame, frame_idx)
        direction_distance = smallest_angular_distance(outlier_state.theta, cluster_theta)
        location_distance = float(
            math.hypot(outlier_state.x - cluster_centroid[0], outlier_state.y - cluster_centroid[1])
        )

        if direction_distance <= direction_thresh_deg and location_distance <= distance_thresh_px:
            if location_distance < best_distance:
                best_distance = location_distance
                best_cluster_id = cluster_id

    return best_cluster_id


def compute_motion_state(
    track_xy: np.ndarray,
    track_mask: np.ndarray,
    ped_id: int,
    frame_idx: int,
    assignments: Dict[int, Optional[int]],
) -> Optional[PedestrianState]:
    if not track_mask[ped_id, frame_idx]:
        return None

    x, y = track_xy[ped_id, frame_idx]

    prev_idx = frame_idx - 1
    while prev_idx >= 0 and not track_mask[ped_id, prev_idx]:
        prev_idx -= 1

    if prev_idx < 0:
        vx, vy = 0.0, 0.0
        theta = 0.0
    else:
        prev_x, prev_y = track_xy[ped_id, prev_idx]
        vx, vy = x - prev_x, y - prev_y
        if abs(vx) < EPS and abs(vy) < EPS:
            theta = 0.0
        else:
            theta = float(math.atan2(vy, vx))

    return PedestrianState(
        ped_id=ped_id,
        x=float(x),
        y=float(y),
        vx=float(vx),
        vy=float(vy),
        theta=theta,
        cluster_id=assignments.get(ped_id),
    )


def get_active_pedestrians(
    track_xy: np.ndarray,
    track_mask: np.ndarray,
    frame_idx: int,
    assignments: Dict[int, Optional[int]],
) -> Dict[int, PedestrianState]:
    states: Dict[int, PedestrianState] = {}
    for ped_id in range(track_xy.shape[0]):
        state = compute_motion_state(track_xy, track_mask, ped_id, frame_idx, assignments)
        if state is not None:
            states[ped_id] = state
    return states


def create_cluster(
    active_clusters: Dict[int, ClusterRuntime],
    active_states: Dict[int, PedestrianState],
    assignments: Dict[int, Optional[int]],
    temporary_pool: Dict[int, PedestrianState],
    cluster_id: int,
    member_ids: Sequence[int],
    frame_idx: int,
) -> None:
    runtime = ClusterRuntime(
        cluster_id=cluster_id,
        created_frame_idx=frame_idx,
        members=set(member_ids),
        last_nonempty_frame_idx=frame_idx,
        active=True,
    )

    member_positions = np.asarray(
        [[active_states[m].x, active_states[m].y] for m in member_ids],
        dtype=np.float32,
    )
    centroid = member_positions.mean(axis=0)
    runtime.centroid_by_frame[frame_idx] = centroid
    runtime.direction_by_frame[frame_idx] = 0.0
    runtime.member_history[frame_idx] = sorted(member_ids)
    runtime.size_history[frame_idx] = len(member_ids)

    active_clusters[cluster_id] = runtime
    for member_id in member_ids:
        assignments[member_id] = cluster_id
        temporary_pool.pop(member_id, None)


def update_cluster_assignments(
    frame_idx: int,
    active_states: Dict[int, PedestrianState],
    active_clusters: Dict[int, ClusterRuntime],
    assignments: Dict[int, Optional[int]],
    temporary_pool: Dict[int, PedestrianState],
    direction_thresh_deg: float,
    distance_thresh_px: float,
    lof_contamination: float,
    lof_neighbor_ratio: float,
    temporary_recluster_min_size: int,
    next_cluster_id: int,
) -> int:
    outlier_records: List[Tuple[int, int]] = []

    for cluster_id, cluster in active_clusters.items():
        members_now = [pid for pid in cluster.members if pid in active_states]
        outlier_ids = evaluate_cluster_members_with_lof(
            members_now,
            active_states,
            contamination=lof_contamination,
            n_neighbors_ratio=lof_neighbor_ratio,
        )
        for outlier_id in outlier_ids:
            outlier_records.append((outlier_id, cluster_id))

    for outlier_id, source_cluster_id in outlier_records:
        if source_cluster_id in active_clusters:
            active_clusters[source_cluster_id].members.discard(outlier_id)
        assignments[outlier_id] = None

    for outlier_id, source_cluster_id in outlier_records:
        state = active_states.get(outlier_id)
        if state is None:
            continue
        nearest_cluster_id = find_nearest_compatible_cluster(
            outlier_state=state,
            active_clusters=active_clusters,
            frame_idx=frame_idx,
            direction_thresh_deg=direction_thresh_deg,
            distance_thresh_px=distance_thresh_px,
            skip_cluster_id=source_cluster_id,
        )
        if nearest_cluster_id is not None:
            active_clusters[nearest_cluster_id].members.add(outlier_id)
            assignments[outlier_id] = nearest_cluster_id
            temporary_pool.pop(outlier_id, None)
        else:
            temporary_pool[outlier_id] = state

    for ped_id, state in active_states.items():
        if assignments.get(ped_id) is not None:
            continue
        nearest_cluster_id = find_nearest_compatible_cluster(
            outlier_state=state,
            active_clusters=active_clusters,
            frame_idx=frame_idx,
            direction_thresh_deg=direction_thresh_deg,
            distance_thresh_px=distance_thresh_px,
        )
        if nearest_cluster_id is not None:
            active_clusters[nearest_cluster_id].members.add(ped_id)
            assignments[ped_id] = nearest_cluster_id
            temporary_pool.pop(ped_id, None)
        else:
            temporary_pool[ped_id] = state

    stale_temp_ids = [
        ped_id
        for ped_id in temporary_pool.keys()
        if ped_id not in active_states or assignments.get(ped_id) is not None
    ]
    for ped_id in stale_temp_ids:
        temporary_pool.pop(ped_id, None)

    # Paper text contains both "5+" and ">10" descriptions; we default to 10 from Algorithm 1.
    if len(temporary_pool) >= temporary_recluster_min_size:
        temp_states = {ped_id: active_states[ped_id] for ped_id in temporary_pool.keys()}
        new_groups = nested_initial_clustering(
            temp_states,
            direction_thresh_deg=direction_thresh_deg,
            distance_thresh_px=distance_thresh_px,
        )
        for member_ids in new_groups:
            create_cluster(
                active_clusters=active_clusters,
                active_states=active_states,
                assignments=assignments,
                temporary_pool=temporary_pool,
                cluster_id=next_cluster_id,
                member_ids=member_ids,
                frame_idx=frame_idx,
            )
            next_cluster_id += 1

    return next_cluster_id


def initialize_centroid(cluster_members: Sequence[int], active_states: Dict[int, PedestrianState]) -> np.ndarray:
    positions = np.asarray(
        [[active_states[mid].x, active_states[mid].y] for mid in cluster_members],
        dtype=np.float32,
    )
    return positions.mean(axis=0)


def update_centroid_with_delta(
    prev_centroid_xy: np.ndarray,
    cluster_members_curr: Sequence[int],
    track_xy: np.ndarray,
    track_mask: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    deltas = []
    for member_id in cluster_members_curr:
        if frame_idx <= 0:
            continue
        if not (track_mask[member_id, frame_idx] and track_mask[member_id, frame_idx - 1]):
            continue
        prev_xy = track_xy[member_id, frame_idx - 1]
        curr_xy = track_xy[member_id, frame_idx]
        deltas.append(curr_xy - prev_xy)

    if not deltas:
        return prev_centroid_xy.copy()

    avg_delta = np.asarray(deltas, dtype=np.float32).mean(axis=0)
    return prev_centroid_xy + avg_delta


def run_dynamic_clustering_scene(
    scene_traj: np.ndarray,
    scene_mask: np.ndarray,
    frames: Sequence[int],
    direction_thresh_deg: float,
    distance_thresh_px: float,
    lof_contamination: float,
    lof_neighbor_ratio: float,
    reeval_interval: int,
    temporary_recluster_min_size: int,
    cluster_empty_tolerance: int,
    centroid_update_interval: int,
):
    # Retained for backward CLI/config compatibility. Centroid delta update is
    # applied per frame based on Eq. (3)-(4); membership re-evaluation cadence
    # is controlled by `reeval_interval`.
    _ = centroid_update_interval

    num_pedestrians, seq_len, _, _ = scene_traj.shape
    track_xy = scene_traj[:, :, 0, :2].astype(np.float32)
    track_mask = scene_mask[:, :, 0].astype(bool)

    assignments: Dict[int, Optional[int]] = {pid: None for pid in range(num_pedestrians)}
    active_clusters: Dict[int, ClusterRuntime] = {}
    archived_clusters: Dict[int, ClusterRuntime] = {}
    temporary_pool: Dict[int, PedestrianState] = {}
    next_cluster_id = 0
    initialized = False

    for frame_idx in range(seq_len):
        active_states = get_active_pedestrians(track_xy, track_mask, frame_idx, assignments)

        if frame_idx == 0:
            continue

        active_ped_ids = set(active_states.keys())

        for cluster in active_clusters.values():
            cluster.members = {pid for pid in cluster.members if pid in active_ped_ids}

        for ped_id in range(num_pedestrians):
            if ped_id not in active_ped_ids:
                assignments[ped_id] = None

        if not initialized:
            if active_states:
                initial_groups = nested_initial_clustering(
                    active_states,
                    direction_thresh_deg=direction_thresh_deg,
                    distance_thresh_px=distance_thresh_px,
                )
                for member_ids in initial_groups:
                    create_cluster(
                        active_clusters=active_clusters,
                        active_states=active_states,
                        assignments=assignments,
                        temporary_pool=temporary_pool,
                        cluster_id=next_cluster_id,
                        member_ids=member_ids,
                        frame_idx=frame_idx,
                    )
                    next_cluster_id += 1
                initialized = True
        else:
            for ped_id, state in active_states.items():
                if assignments.get(ped_id) is not None:
                    continue
                nearest_cluster_id = find_nearest_compatible_cluster(
                    outlier_state=state,
                    active_clusters=active_clusters,
                    frame_idx=frame_idx,
                    direction_thresh_deg=direction_thresh_deg,
                    distance_thresh_px=distance_thresh_px,
                )
                if nearest_cluster_id is not None:
                    active_clusters[nearest_cluster_id].members.add(ped_id)
                    assignments[ped_id] = nearest_cluster_id
                    temporary_pool.pop(ped_id, None)
                else:
                    temporary_pool[ped_id] = state

            if frame_idx % reeval_interval == 0 and active_clusters:
                next_cluster_id = update_cluster_assignments(
                    frame_idx=frame_idx,
                    active_states=active_states,
                    active_clusters=active_clusters,
                    assignments=assignments,
                    temporary_pool=temporary_pool,
                    direction_thresh_deg=direction_thresh_deg,
                    distance_thresh_px=distance_thresh_px,
                    lof_contamination=lof_contamination,
                    lof_neighbor_ratio=lof_neighbor_ratio,
                    temporary_recluster_min_size=temporary_recluster_min_size,
                    next_cluster_id=next_cluster_id,
                )

        for cluster_id, cluster in list(active_clusters.items()):
            members_now = sorted([pid for pid in cluster.members if pid in active_states])
            cluster.member_history[frame_idx] = members_now
            cluster.size_history[frame_idx] = len(members_now)
            if members_now:
                cluster.last_nonempty_frame_idx = frame_idx

            if frame_idx not in cluster.centroid_by_frame:
                prev_centroid = _latest_value(cluster.centroid_by_frame, frame_idx - 1)
                if prev_centroid is None:
                    if members_now:
                        cluster.centroid_by_frame[frame_idx] = initialize_centroid(members_now, active_states)
                    continue

                # Paper Eq. (3)-(4) defines centroid update from t-1 -> t displacement.
                # We therefore propagate centroid every frame via average member delta
                # (while membership itself is re-evaluated every `reeval_interval`).
                # This avoids staircase-like trajectories from holding positions for
                # multiple frames and better matches Figure 2 behavior.
                if members_now:
                    updated_centroid = update_centroid_with_delta(
                        prev_centroid_xy=prev_centroid,
                        cluster_members_curr=members_now,
                        track_xy=track_xy,
                        track_mask=track_mask,
                        frame_idx=frame_idx,
                    )
                    cluster.centroid_by_frame[frame_idx] = updated_centroid
                else:
                    cluster.centroid_by_frame[frame_idx] = prev_centroid.copy()

            prev_centroid = _latest_value(cluster.centroid_by_frame, frame_idx - 1)
            curr_centroid = cluster.centroid_by_frame[frame_idx]
            if prev_centroid is None:
                cluster.direction_by_frame[frame_idx] = 0.0
            else:
                direction_vec = prev_centroid - curr_centroid  # paper Eq: V_i_t = C^p_{t-1} - C^p_t
                if np.linalg.norm(direction_vec) < EPS:
                    cluster.direction_by_frame[frame_idx] = _latest_angle(
                        cluster.direction_by_frame,
                        frame_idx - 1,
                    )
                else:
                    cluster.direction_by_frame[frame_idx] = float(
                        math.atan2(direction_vec[1], direction_vec[0])
                    )

            if (frame_idx - cluster.last_nonempty_frame_idx) > cluster_empty_tolerance:
                cluster.active = False
                archived_clusters[cluster_id] = cluster
                del active_clusters[cluster_id]
                for ped_id, assigned_cluster in assignments.items():
                    if assigned_cluster == cluster_id:
                        assignments[ped_id] = None

    all_clusters = {**archived_clusters, **active_clusters}

    return all_clusters


def build_centroid_tracks_from_clusters(
    clusters: Dict[int, ClusterRuntime],
    frames: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, dict]]:
    seq_len = len(frames)
    ordered_cluster_ids = sorted(
        clusters.keys(), key=lambda cid: (clusters[cid].created_frame_idx, cid)
    )

    centroid_tracks = []
    centroid_masks = []
    metadata = {}

    for cluster_id in ordered_cluster_ids:
        cluster = clusters[cluster_id]
        trajectory = np.zeros((seq_len, 2), dtype=np.float32)
        mask = np.zeros(seq_len, dtype=np.float32)

        for frame_idx in range(seq_len):
            centroid = cluster.centroid_by_frame.get(frame_idx)
            if centroid is None:
                continue
            trajectory[frame_idx] = centroid
            # Only expose centroid points for frames where the cluster has members.
            # This prevents plotting/using stale carried-forward positions as active
            # trajectories when a cluster is temporarily empty.
            if cluster.size_history.get(frame_idx, 0) > 0:
                mask[frame_idx] = 1.0

        valid_idxs = np.where(mask > 0)[0]
        if len(valid_idxs) < 2:
            continue

        sizes = [cluster.size_history.get(i, 0) for i in valid_idxs]
        nonzero_sizes = [s for s in sizes if s > 0]
        member_ids = sorted(
            {
                member_id
                for frame_members in cluster.member_history.values()
                for member_id in frame_members
            }
        )

        size_history = {
            str(frames[i]): int(cluster.size_history.get(i, 0))
            for i in valid_idxs
        }

        metadata[cluster_id] = {
            "cluster_size": int(round(np.mean(nonzero_sizes))) if nonzero_sizes else 0,
            "start_frame": int(frames[int(valid_idxs[0])]),
            "end_frame": int(frames[int(valid_idxs[-1])]),
            "member_ids": [int(mid) for mid in member_ids],
            "cluster_size_history": size_history,
        }

        centroid_tracks.append(trajectory)
        centroid_masks.append(mask)

    if not centroid_tracks:
        return (
            np.zeros((0, seq_len, 2), dtype=np.float32),
            np.zeros((0, seq_len), dtype=np.float32),
            [],
            {},
        )

    kept_cluster_ids = [
        cid
        for cid in ordered_cluster_ids
        if cid in metadata
    ]

    return (
        np.asarray(centroid_tracks, dtype=np.float32),
        np.asarray(centroid_masks, dtype=np.float32),
        kept_cluster_ids,
        metadata,
    )


def convert_scene_to_centroid_samples(
    scene_traj: np.ndarray,
    scene_mask: np.ndarray,
    filename: str,
    frames: Sequence[int],
    source_sample_index: int,
    global_centroid_id_counter: int,
    direction_thresh_deg: float,
    distance_thresh_px: float,
    lof_contamination: float,
    lof_neighbor_ratio: float,
    reeval_interval: int,
    temporary_recluster_min_size: int,
    cluster_empty_tolerance: int,
    centroid_update_interval: int,
):
    clusters = run_dynamic_clustering_scene(
        scene_traj=scene_traj,
        scene_mask=scene_mask,
        frames=frames,
        direction_thresh_deg=direction_thresh_deg,
        distance_thresh_px=distance_thresh_px,
        lof_contamination=lof_contamination,
        lof_neighbor_ratio=lof_neighbor_ratio,
        reeval_interval=reeval_interval,
        temporary_recluster_min_size=temporary_recluster_min_size,
        cluster_empty_tolerance=cluster_empty_tolerance,
        centroid_update_interval=centroid_update_interval,
    )

    centroid_tracks, centroid_masks, local_cluster_ids, local_metadata = (
        build_centroid_tracks_from_clusters(clusters, frames)
    )

    if len(local_cluster_ids) == 0:
        return [], [], [], [], {}, global_centroid_id_counter

    num_centroids, seq_len, _ = centroid_tracks.shape

    local_to_global = {}
    for i, local_cluster_id in enumerate(local_cluster_ids):
        local_to_global[local_cluster_id] = global_centroid_id_counter + i

    centroid_samples = []
    centroid_filename_list = []
    centroid_frames_list = []
    centroid_pedestrians_list = []
    centroid_metadata = {}

    for local_cluster_id in local_cluster_ids:
        global_centroid_id = local_to_global[local_cluster_id]
        metadata = dict(local_metadata[local_cluster_id])
        metadata["source_scene"] = filename
        metadata["source_sample_index"] = int(source_sample_index)
        metadata["source_local_cluster_id"] = int(local_cluster_id)
        centroid_metadata[str(global_centroid_id)] = metadata

    for primary_idx, local_cluster_id in enumerate(local_cluster_ids):
        order = [primary_idx] + [j for j in range(num_centroids) if j != primary_idx]

        ordered_tracks = centroid_tracks[order]
        ordered_masks = centroid_masks[order]

        traj_arr = np.zeros((num_centroids, seq_len, 1, 3), dtype=np.float32)
        traj_arr[:, :, 0, :2] = ordered_tracks
        mask_arr = ordered_masks[:, :, None].astype(np.float32)

        centroid_samples.append((traj_arr, mask_arr))
        centroid_filename_list.append(filename)
        centroid_frames_list.append(list(frames))
        centroid_pedestrians_list.append(local_to_global[local_cluster_id])

    next_counter = global_centroid_id_counter + len(local_cluster_ids)

    return (
        centroid_samples,
        centroid_filename_list,
        centroid_frames_list,
        centroid_pedestrians_list,
        centroid_metadata,
        next_counter,
    )


def write_centroid_scene(scene_output_path: str, centroid_tracks, metadata) -> None:
    os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
    with open(scene_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_centroid_tracks": int(len(centroid_tracks)),
                "metadata": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def process_split(
    split: str,
    name: str,
    data_dir: str,
    save_root: str,
    valid_ratio: float,
    min_prompt_num: int,
    direction_thresh_deg: float,
    distance_thresh_px: float,
    lof_contamination: float,
    lof_neighbor_ratio: float,
    reeval_interval: int,
    temporary_recluster_min_size: int,
    cluster_empty_tolerance: int,
    centroid_update_interval: int,
    output_name_suffix: str,
) -> str:
    r, stride = infer_r_stride(name)
    (
        raw_trajs,
        raw_masks,
        filename_list,
        frames_list,
        pedestrians_list,
        _,
        _,
        _,
    ) = load_raw_data(split=split, name=name, data_dir=data_dir, r=r, stride=stride)

    centroid_joint_and_mask = []
    centroid_filename_list = []
    centroid_frames_list = []
    centroid_pedestrians_list = []
    centroid_metadata = {}
    centroid_metadata_by_scene = defaultdict(dict)

    global_centroid_id_counter = CENTROID_ID_OFFSET

    for sample_idx, (traj_tensor, mask_tensor, filename, frames) in enumerate(
        tqdm(
            zip(raw_trajs, raw_masks, filename_list, frames_list),
            total=len(raw_trajs),
            desc=f"[{split}] dynamic centroid clustering",
        )
    ):
        scene_traj = traj_tensor.cpu().numpy()
        scene_mask = mask_tensor.cpu().numpy()

        (
            scene_samples,
            scene_filenames,
            scene_frames,
            scene_pedestrians,
            scene_metadata,
            global_centroid_id_counter,
        ) = convert_scene_to_centroid_samples(
            scene_traj=scene_traj,
            scene_mask=scene_mask,
            filename=filename,
            frames=frames,
            source_sample_index=sample_idx,
            global_centroid_id_counter=global_centroid_id_counter,
            direction_thresh_deg=direction_thresh_deg,
            distance_thresh_px=distance_thresh_px,
            lof_contamination=lof_contamination,
            lof_neighbor_ratio=lof_neighbor_ratio,
            reeval_interval=reeval_interval,
            temporary_recluster_min_size=temporary_recluster_min_size,
            cluster_empty_tolerance=cluster_empty_tolerance,
            centroid_update_interval=centroid_update_interval,
        )

        centroid_joint_and_mask.extend(scene_samples)
        centroid_filename_list.extend(scene_filenames)
        centroid_frames_list.extend(scene_frames)
        centroid_pedestrians_list.extend(scene_pedestrians)

        centroid_metadata.update(scene_metadata)
        for centroid_track_id, metadata in scene_metadata.items():
            centroid_metadata_by_scene[filename][centroid_track_id] = metadata

    if not centroid_joint_and_mask:
        raise RuntimeError(
            f"No centroid trajectories generated for split={split}. "
            "Check preprocessing thresholds or input data integrity."
        )

    centroid_trajs = [
        np.asarray(sample[0], dtype=np.float32) for sample in centroid_joint_and_mask
    ]
    centroid_masks = [
        np.asarray(sample[1], dtype=np.float32) for sample in centroid_joint_and_mask
    ]

    import torch

    centroid_trajs = [torch.from_numpy(arr) for arr in centroid_trajs]
    centroid_masks = [torch.from_numpy(arr) for arr in centroid_masks]

    filename2idxs_dict = defaultdict(list)
    idx2filename_dict = {}
    for idx, filename in enumerate(centroid_filename_list):
        filename2idxs_dict[filename].append(idx)
        idx2filename_dict[idx] = filename

    pool_indices_by_fold, valid_indices_by_fold = pool_valid_split(
        centroid_filename_list,
        centroid_frames_list,
        centroid_pedestrians_list,
        filename2idxs_dict,
        idx2filename_dict,
        valid_ratio=valid_ratio,
        min_prompt_num=min_prompt_num,
    )

    num_people = [traj.shape[0] for traj in centroid_trajs]
    cluster_sizes = [meta["cluster_size"] for meta in centroid_metadata.values()]
    single_member_clusters = sum(1 for s in cluster_sizes if s == 1)
    centroid_track_lengths = [
        len(meta.get("cluster_size_history", {})) for meta in centroid_metadata.values()
    ]

    config = {
        "num_trajs": int(len(centroid_trajs)),
        "avg_num_people": int(np.mean(num_people)),
        "max_num_people": int(np.max(num_people)),
        "min_num_people": int(np.min(num_people)),
        "r": int(r),
        "stride": int(stride),
        "name": f"{name}{output_name_suffix}",
        "source_name": name,
        "split": split,
        "preprocess_type": "centroid_dynamic_clustering",
        "paper_defaults": {
            "distance_thresh_px": distance_thresh_px,
            "direction_thresh_deg": direction_thresh_deg,
            "lof_contamination": lof_contamination,
            "lof_neighbor_ratio": lof_neighbor_ratio,
            "reeval_interval": reeval_interval,
            "temporary_recluster_min_size": temporary_recluster_min_size,
            "cluster_empty_tolerance": cluster_empty_tolerance,
            "centroid_update_interval": centroid_update_interval,
        },
        "stats": compute_stats_traj(centroid_trajs),
        "sanity": {
            "raw_input_track_windows": int(len(raw_trajs)),
            "centroid_track_windows": int(len(centroid_trajs)),
            "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "single_member_clusters": int(single_member_clusters),
            "avg_centroid_track_length": float(np.mean(centroid_track_lengths))
            if centroid_track_lengths
            else 0.0,
        },
    }

    print(
        f"[{split}] raw windows={len(raw_trajs)} | centroid windows={len(centroid_trajs)} | "
        f"avg_cluster_size={config['sanity']['avg_cluster_size']:.3f} | "
        f"single_member_clusters={single_member_clusters} | "
        f"avg_centroid_track_length={config['sanity']['avg_centroid_track_length']:.3f}"
    )

    save_name = f"{name}{output_name_suffix}"
    save_data(
        save_name=save_name,
        split=split,
        trajs=centroid_trajs,
        masks=centroid_masks,
        valid_indices_by_fold=valid_indices_by_fold,
        pool_indices_by_fold=pool_indices_by_fold,
        filename2idxs_dict=filename2idxs_dict,
        idx2filename_dict=idx2filename_dict,
        config=config,
        filename_list=centroid_filename_list,
        frames_list=centroid_frames_list,
        pedestrians_list=centroid_pedestrians_list,
        save_root=save_root,
    )

    save_dir = os.path.join(save_root, save_name)
    with open(
        os.path.join(save_dir, f"{split}_centroid_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(centroid_metadata, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(save_dir, f"{split}_centroid_metadata_by_scene.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(centroid_metadata_by_scene, f, ensure_ascii=False, indent=2)

    write_centroid_scene(
        os.path.join(save_dir, f"{split}_centroid_scene_summary.json"),
        centroid_metadata,
        {
            "raw_input_track_windows": len(raw_trajs),
            "centroid_track_windows": len(centroid_trajs),
            "avg_cluster_size": config["sanity"]["avg_cluster_size"],
            "single_member_clusters": single_member_clusters,
            "avg_centroid_track_length": config["sanity"]["avg_centroid_track_length"],
        },
    )

    return save_name


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic-clustering centroid preprocessing for TrajICL"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocess", "sim_matrix", "traj_sim", "all"],
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_root", type=str, default="outputs/processed_data")
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--similarity_scopes", type=str, default="hist,seq")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--min_prompt_num", type=int, default=16)
    parser.add_argument("--hist_len", type=int, default=9)

    parser.add_argument("--output_name_suffix", type=str, default="_centroid")

    # Paper defaults
    parser.add_argument("--distance_thresh_px", type=float, default=120.0)
    parser.add_argument("--direction_thresh_deg", type=float, default=50.0)
    parser.add_argument("--lof_contamination", type=float, default=0.2)
    parser.add_argument("--lof_neighbor_ratio", type=float, default=0.8)
    parser.add_argument("--reeval_interval", type=int, default=10)
    parser.add_argument("--cluster_empty_tolerance", type=int, default=3)

    # Algorithm 1 lists >10 while text also mentions 5+; default to Algorithm 1 value (10).
    parser.add_argument("--temporary_recluster_min_size", type=int, default=10)
    parser.add_argument(
        "--centroid_update_interval",
        type=int,
        default=1,
        help=(
            "Deprecated compatibility flag. Centroid delta update is computed "
            "per frame; cluster membership re-evaluation uses --reeval_interval."
        ),
    )

    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument("--vel_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max_similar", type=int, default=16)
    parser.add_argument("--no_parallel", action="store_true")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--load_precomputed", action="store_true")
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--disable_file_logging", action="store_true")

    args = parser.parse_args()
    log_state = None
    if not args.disable_file_logging:
        log_state, log_path = start_run_logging(
            log_dir=args.log_dir,
            script_name="preprocess_centroids",
        )
        print(f"[run-log] Capturing stdout/stderr to {log_path}")

    try:
        splits = [s.strip() for s in args.splits.split(",")]
        similarity_scopes = [s.strip() for s in args.similarity_scopes.split(",")]
        save_name = f"{args.name}{args.output_name_suffix}"

        if args.stage in ["preprocess", "all"]:
            print("===== Stage 1: centroid preprocess =====")
            for split in splits:
                print(f"[CentroidPreprocess] name={args.name}, split={split}")
                save_name = process_split(
                    split=split,
                    name=args.name,
                    data_dir=args.data_dir,
                    save_root=args.save_root,
                    valid_ratio=args.valid_ratio,
                    min_prompt_num=args.min_prompt_num,
                    direction_thresh_deg=args.direction_thresh_deg,
                    distance_thresh_px=args.distance_thresh_px,
                    lof_contamination=args.lof_contamination,
                    lof_neighbor_ratio=args.lof_neighbor_ratio,
                    reeval_interval=args.reeval_interval,
                    temporary_recluster_min_size=args.temporary_recluster_min_size,
                    cluster_empty_tolerance=args.cluster_empty_tolerance,
                    centroid_update_interval=args.centroid_update_interval,
                    output_name_suffix=args.output_name_suffix,
                )

        if args.stage in ["sim_matrix", "all"]:
            print("===== Stage 2: compute sim_matrix =====")
            for split in splits:
                (
                    trajs,
                    masks,
                    filename2idxs_dict,
                    idx2filename_dict,
                    config,
                    filename_list,
                    frames_list,
                    pedestrians_list,
                    valid_indices_by_fold,
                    pool_indices_by_fold,
                    _,
                ) = load_processed_split(save_name, split, args.save_root)

                save_dir = os.path.join(args.save_root, save_name)
                for similarity_scope in similarity_scopes:
                    print(f"  -> similarity_scope={similarity_scope}")
                    compute_sim_matrix(
                        trajs,
                        filename2idxs_dict,
                        hist_len=args.hist_len,
                        save_dir=save_dir,
                        split=split,
                        load_precomputed=args.load_precomputed,
                        similarity_scope=similarity_scope,
                    )

        if args.stage in ["traj_sim", "all"]:
            print("===== Stage 3: compute trajectory similarity dicts =====")
            for split in splits:
                for similarity_scope in similarity_scopes:
                    (
                        trajs,
                        masks,
                        filename2idxs_dict,
                        idx2filename_dict,
                        config,
                        filename_list,
                        frames_list,
                        pedestrians_list,
                        valid_indices_by_fold,
                        pool_indices_by_fold,
                        sim_matrix_dicts,
                    ) = load_processed_split(
                        save_name,
                        split,
                        args.save_root,
                        similarity_scope=similarity_scope,
                    )

                    similar_traj_dicts = []
                    for i, (valid_indices, pool_indices) in enumerate(
                        zip(valid_indices_by_fold, pool_indices_by_fold)
                    ):
                        print(
                            f"  Fold {i}: pool={len(pool_indices)}, valid={len(valid_indices)}"
                        )
                        similar_traj_dict, _ = compute_trajectory_similarity(
                            filename2idxs_dict,
                            dist_weight=args.dist_weight,
                            vel_weight=args.vel_weight,
                            threshold=args.threshold,
                            max_similar=args.max_similar,
                            pool_indices=pool_indices,
                            sim_matrix_dicts=sim_matrix_dicts,
                            use_parallel=not args.no_parallel,
                            max_workers=args.max_workers,
                        )
                        similar_traj_dicts.append(similar_traj_dict)

                    out_path = os.path.join(
                        args.save_root,
                        save_name,
                        f"{split}_similar_traj_dicts_{similarity_scope}.pickle",
                    )
                    with open(out_path, mode="wb") as f:
                        import pickle

                        pickle.dump(similar_traj_dicts, f)
                    print(f"  -> saved similar_traj_dicts to {out_path}")
    finally:
        finalize_run_logging(log_state)


if __name__ == "__main__":
    main()
