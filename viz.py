import argparse
import json
import math
import os
import pickle
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.plotting import prepare_matplotlib
from utils.run_logging import finalize_run_logging, start_run_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize raw vs centroid TrajICL datasets and save presentation-ready plots."
    )
    parser.add_argument("--raw_dir", type=str, default="outputs/processed_data/motsynth")
    parser.add_argument(
        "--centroid_dir",
        type=str,
        default="outputs/processed_data/motsynth_centroid",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--hist_len", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_agents_per_plot", type=int, default=40)
    parser.add_argument(
        "--pair_coordinate_mode",
        type=str,
        default="both",
        choices=["absolute", "normalized", "both"],
        help=(
            "Coordinate mode for paired raw-vs-centroid figures: "
            "'absolute' keeps scene coordinates, 'normalized' subtracts each panel's "
            "primary origin, 'both' exports both variants."
        ),
    )
    parser.add_argument("--max_heatmap_points", type=int, default=500000)
    parser.add_argument(
        "--stats_max_samples",
        type=int,
        default=5000,
        help="Cap number of samples used for global stats plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Default: outputs/visualizations/raw_vs_centroid_<timestamp>",
    )
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--disable_file_logging", action="store_true")
    return parser.parse_args()


def _prepare_matplotlib():
    try:
        plt = prepare_matplotlib(use_agg=True)
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for viz.py. Install it in your environment, "
            "for example: pip install matplotlib"
        ) from exc
    return plt, PdfPages


def load_split(directory: str, split: str):
    traj_path = os.path.join(directory, f"{split}_trajs.pt")
    mask_path = os.path.join(directory, f"{split}_masks.pt")
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Missing mask file: {mask_path}")
    trajs = torch.load(traj_path)
    masks = torch.load(mask_path)
    return trajs, masks


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_centroid_match_data(centroid_dir: str, split: str):
    metadata_path = os.path.join(centroid_dir, f"{split}_centroid_metadata.json")
    ped_list_path = os.path.join(centroid_dir, f"{split}_pedestrians_list.pickle")
    if not os.path.isfile(metadata_path) or not os.path.isfile(ped_list_path):
        return None, None
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    ped_list = _load_pickle(ped_list_path)
    return metadata, ped_list


def build_matched_sample_pairs(
    raw_total: int,
    cen_total: int,
    centroid_ped_list,
    centroid_metadata,
    num_samples: int,
    seed: int,
):
    if centroid_ped_list is None or centroid_metadata is None:
        return None, None

    grouped_by_raw: Dict[int, List[int]] = {}
    limit = min(cen_total, len(centroid_ped_list))
    for cen_idx in range(limit):
        centroid_track_id = str(centroid_ped_list[cen_idx])
        meta = centroid_metadata.get(centroid_track_id)
        if not isinstance(meta, dict):
            continue
        raw_idx = meta.get("source_sample_index")
        if isinstance(raw_idx, (int, np.integer)):
            raw_idx = int(raw_idx)
            if 0 <= raw_idx < raw_total:
                grouped_by_raw.setdefault(raw_idx, []).append(cen_idx)

    if not grouped_by_raw:
        return None, None

    rng = random.Random(seed)
    raw_candidates = list(grouped_by_raw.keys())
    rng.shuffle(raw_candidates)
    chosen_raw = sorted(raw_candidates[: min(num_samples, len(raw_candidates))])

    chosen_cen = []
    for raw_idx in chosen_raw:
        candidates = grouped_by_raw[raw_idx]
        chosen_cen.append(candidates[rng.randrange(len(candidates))])

    return chosen_raw, chosen_cen


def sample_indices(total: int, num_samples: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    num = min(num_samples, total)
    return sorted(rng.sample(range(total), num))


def to_xy_mask(traj_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    traj_np = traj_tensor.detach().cpu().numpy()  # [N, T, 1, 3]
    mask_np = mask_tensor.detach().cpu().numpy()  # [N, T, 1]
    xy = traj_np[:, :, 0, :2]
    mask = mask_np[:, :, 0].astype(bool)
    return xy, mask


def _safe_primary_points(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    primary_mask = mask[0]
    return xy[0][primary_mask]


def normalize_xy_by_primary_origin(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    norm_xy = xy.copy()
    if norm_xy.shape[0] == 0:
        return norm_xy
    primary_valid = np.flatnonzero(mask[0])
    if len(primary_valid) == 0:
        return norm_xy
    origin = norm_xy[0, primary_valid[0]].copy()
    norm_xy[mask] = norm_xy[mask] - origin
    return norm_xy


def compute_axis_limits_for_sets(
    xy_mask_sets: List[Tuple[np.ndarray, np.ndarray]],
    min_span: float = 20.0,
    pad_ratio: float = 0.08,
):
    all_pts = []
    for xy, mask in xy_mask_sets:
        pts = xy[mask]
        if len(pts) > 0:
            all_pts.append(pts)

    if not all_pts:
        return None, None

    pts = np.concatenate(all_pts, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    x_span = max(float(maxs[0] - mins[0]), min_span)
    y_span = max(float(maxs[1] - mins[1]), min_span)
    x_center = float((maxs[0] + mins[0]) * 0.5)
    y_center = float((maxs[1] + mins[1]) * 0.5)
    x_half = x_span * (0.5 + pad_ratio)
    y_half = y_span * (0.5 + pad_ratio)

    return (x_center - x_half, x_center + x_half), (y_center - y_half, y_center + y_half)


def compute_stats(
    trajs: List[torch.Tensor],
    masks: List[torch.Tensor],
    sample_idxs: List[int],
    max_heatmap_points: int,
    hist_len: int,
) -> Dict[str, np.ndarray]:
    agent_counts = []
    primary_speeds = []
    primary_displacements = []
    primary_headings_deg = []
    mean_speed_sum = None
    mean_speed_cnt = None

    all_points = []
    accumulated_points = 0

    for idx in sample_idxs:
        xy, mask = to_xy_mask(trajs[idx], masks[idx])
        n_agents, seq_len, _ = xy.shape

        valid_agents = int(np.sum(mask.any(axis=1)))
        agent_counts.append(valid_agents)

        primary_points = _safe_primary_points(xy, mask)
        if len(primary_points) >= 2:
            delta = np.diff(primary_points, axis=0)
            speeds = np.linalg.norm(delta, axis=1)
            primary_speeds.extend(speeds.tolist())

            displacement = np.linalg.norm(primary_points[-1] - primary_points[0])
            primary_displacements.append(float(displacement))

            headings = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
            primary_headings_deg.extend(headings.tolist())

        # Mean primary speed per timestep t->t+1 in canonical sequence (masked)
        if n_agents > 0:
            if mean_speed_sum is None:
                mean_speed_sum = np.zeros(seq_len - 1, dtype=np.float64)
                mean_speed_cnt = np.zeros(seq_len - 1, dtype=np.float64)

            pxy = xy[0]
            pmask = mask[0]
            for t in range(seq_len - 1):
                if pmask[t] and pmask[t + 1]:
                    s = np.linalg.norm(pxy[t + 1] - pxy[t])
                    mean_speed_sum[t] += float(s)
                    mean_speed_cnt[t] += 1.0

        # Heatmap points from all valid coordinates
        if accumulated_points < max_heatmap_points:
            pts = xy[mask]
            remaining = max_heatmap_points - accumulated_points
            if len(pts) > remaining:
                pts = pts[:remaining]
            all_points.append(pts)
            accumulated_points += len(pts)

    mean_primary_speed_per_t = np.divide(
        mean_speed_sum,
        np.maximum(mean_speed_cnt, 1.0),
        out=np.zeros_like(mean_speed_sum),
        where=mean_speed_cnt > 0,
    ) if mean_speed_sum is not None else np.zeros(0, dtype=np.float64)

    heatmap_points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 2), dtype=np.float32)

    return {
        "agent_counts": np.asarray(agent_counts, dtype=np.int32),
        "primary_speeds": np.asarray(primary_speeds, dtype=np.float32),
        "primary_displacements": np.asarray(primary_displacements, dtype=np.float32),
        "primary_headings_deg": np.asarray(primary_headings_deg, dtype=np.float32),
        "mean_primary_speed_per_t": np.asarray(mean_primary_speed_per_t, dtype=np.float32),
        "heatmap_points": np.asarray(heatmap_points, dtype=np.float32),
    }


def _plot_agent_trajectories(
    ax,
    xy: np.ndarray,
    mask: np.ndarray,
    hist_len: int,
    max_agents: int,
    title: str,
):
    n_agents = xy.shape[0]
    show_agents = min(n_agents, max_agents)

    for agent in range(show_agents):
        color = "C0" if agent == 0 else "C1"
        alpha = 0.9 if agent == 0 else 0.35
        lw = 1.8 if agent == 0 else 0.8

        hist_mask = mask[agent, :hist_len]
        fut_mask = mask[agent, hist_len:]

        hist_pts = xy[agent, :hist_len][hist_mask]
        fut_pts = xy[agent, hist_len:][fut_mask]

        if len(hist_pts) >= 2:
            ax.plot(hist_pts[:, 0], hist_pts[:, 1], color=color, linewidth=lw, alpha=alpha)
        if len(fut_pts) >= 2:
            ax.plot(fut_pts[:, 0], fut_pts[:, 1], color=color, linewidth=lw, alpha=alpha, linestyle="--")

        all_valid = xy[agent][mask[agent]]
        if len(all_valid) >= 1:
            ax.scatter(all_valid[0, 0], all_valid[0, 1], s=8, color=color, alpha=alpha)
            ax.scatter(all_valid[-1, 0], all_valid[-1, 1], s=8, color=color, alpha=alpha, marker="x")

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)



def make_samples_grid(
    trajs,
    masks,
    sample_idxs,
    hist_len,
    max_agents,
    title_prefix,
    out_path,
):
    plt, _ = _prepare_matplotlib()
    n = len(sample_idxs)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.7 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for i, idx in enumerate(sample_idxs):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        xy, mask = to_xy_mask(trajs[idx], masks[idx])
        _plot_agent_trajectories(
            ax,
            xy,
            mask,
            hist_len,
            max_agents,
            title=f"idx={idx} | agents={xy.shape[0]}",
        )

    for j in range(len(sample_idxs), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_side_by_side_pairs(
    raw_trajs,
    raw_masks,
    raw_idxs,
    cen_trajs,
    cen_masks,
    cen_idxs,
    hist_len,
    max_agents,
    out_path,
    normalize_origin=True,
    share_axes=True,
):
    plt, _ = _prepare_matplotlib()
    n = min(len(raw_idxs), len(cen_idxs))
    if n == 0:
        raise ValueError("No sample pairs available for side-by-side plotting.")
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        raw_xy, raw_mask = to_xy_mask(raw_trajs[raw_idxs[i]], raw_masks[raw_idxs[i]])
        cen_xy, cen_mask = to_xy_mask(cen_trajs[cen_idxs[i]], cen_masks[cen_idxs[i]])
        if normalize_origin:
            raw_xy = normalize_xy_by_primary_origin(raw_xy, raw_mask)
            cen_xy = normalize_xy_by_primary_origin(cen_xy, cen_mask)

        _plot_agent_trajectories(
            axes[i, 0],
            raw_xy,
            raw_mask,
            hist_len,
            max_agents,
            title=f"RAW idx={raw_idxs[i]} | agents={raw_xy.shape[0]}",
        )
        _plot_agent_trajectories(
            axes[i, 1],
            cen_xy,
            cen_mask,
            hist_len,
            max_agents,
            title=f"CENTROID idx={cen_idxs[i]} | agents={cen_xy.shape[0]}",
        )
        if share_axes:
            xlim, ylim = compute_axis_limits_for_sets([(raw_xy, raw_mask), (cen_xy, cen_mask)])
            if xlim is not None and ylim is not None:
                axes[i, 0].set_xlim(*xlim)
                axes[i, 0].set_ylim(*ylim)
                axes[i, 1].set_xlim(*xlim)
                axes[i, 1].set_ylim(*ylim)

    fig.suptitle("Raw vs Centroid: Sample-wise Side-by-Side", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_before_after_figure(
    raw_trajs,
    raw_masks,
    raw_idx,
    cen_trajs,
    cen_masks,
    cen_idx,
    hist_len,
    max_agents,
    out_path,
    normalize_origin=True,
    share_axes=True,
):
    plt, _ = _prepare_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    raw_xy, raw_mask = to_xy_mask(raw_trajs[raw_idx], raw_masks[raw_idx])
    cen_xy, cen_mask = to_xy_mask(cen_trajs[cen_idx], cen_masks[cen_idx])
    if normalize_origin:
        raw_xy = normalize_xy_by_primary_origin(raw_xy, raw_mask)
        cen_xy = normalize_xy_by_primary_origin(cen_xy, cen_mask)

    _plot_agent_trajectories(
        axes[0],
        raw_xy,
        raw_mask,
        hist_len,
        max_agents,
        title=f"Before (RAW) | idx={raw_idx} | agents={raw_xy.shape[0]}",
    )
    _plot_agent_trajectories(
        axes[1],
        cen_xy,
        cen_mask,
        hist_len,
        max_agents,
        title=f"After (CENTROID) | idx={cen_idx} | agents={cen_xy.shape[0]}",
    )
    if share_axes:
        xlim, ylim = compute_axis_limits_for_sets([(raw_xy, raw_mask), (cen_xy, cen_mask)])
        if xlim is not None and ylim is not None:
            axes[0].set_xlim(*xlim)
            axes[0].set_ylim(*ylim)
            axes[1].set_xlim(*xlim)
            axes[1].set_ylim(*ylim)

    if normalize_origin:
        coord_note = "Paired plots: origin-normalized + shared axes"
    else:
        coord_note = "Paired plots: absolute scene coordinates + shared axes"
    legend_text = (
        "Blue: primary track\n"
        "Orange: context tracks\n"
        "Solid: history, Dashed: future\n"
        f"{coord_note}"
    )
    fig.text(
        0.5,
        0.01,
        legend_text,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    fig.suptitle("Before vs After: Raw Trajectories vs Centroid Representation", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_hist_compare(raw_vals, cen_vals, xlabel, title, out_path, bins=40):
    plt, _ = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(raw_vals, bins=bins, alpha=0.55, label="raw")
    ax.hist(cen_vals, bins=bins, alpha=0.55, label="centroid")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_box_compare(raw_vals, cen_vals, ylabel, title, out_path):
    plt, _ = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([raw_vals, cen_vals], labels=["raw", "centroid"], showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(points, title, out_path):
    plt, _ = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(points) == 0:
        ax.text(0.5, 0.5, "No points", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return

    hb = ax.hexbin(points[:, 0], points[:, 1], gridsize=70, mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log-density")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_mean_speed_curve(raw_curve, cen_curve, hist_len, out_path):
    plt, _ = _prepare_matplotlib()
    t = np.arange(1, len(raw_curve) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, raw_curve, label="raw", linewidth=2)
    ax.plot(t, cen_curve, label="centroid", linewidth=2)
    ax.axvline(hist_len, linestyle="--", color="k", alpha=0.5, linewidth=1)
    ax.set_xlabel("Timestep transition t->t+1")
    ax.set_ylabel("Mean primary speed")
    ax.set_title("Mean Primary Speed by Timestep")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_summary_json(raw_stats, cen_stats, split, raw_samples, cen_samples, out_path):
    summary = {
        "split": split,
        "raw_num_samples_total": int(raw_samples),
        "centroid_num_samples_total": int(cen_samples),
        "raw": {
            "agent_count_mean": float(np.mean(raw_stats["agent_counts"])) if len(raw_stats["agent_counts"]) else 0.0,
            "agent_count_median": float(np.median(raw_stats["agent_counts"])) if len(raw_stats["agent_counts"]) else 0.0,
            "primary_speed_mean": float(np.mean(raw_stats["primary_speeds"])) if len(raw_stats["primary_speeds"]) else 0.0,
            "primary_displacement_mean": float(np.mean(raw_stats["primary_displacements"])) if len(raw_stats["primary_displacements"]) else 0.0,
        },
        "centroid": {
            "agent_count_mean": float(np.mean(cen_stats["agent_counts"])) if len(cen_stats["agent_counts"]) else 0.0,
            "agent_count_median": float(np.median(cen_stats["agent_counts"])) if len(cen_stats["agent_counts"]) else 0.0,
            "primary_speed_mean": float(np.mean(cen_stats["primary_speeds"])) if len(cen_stats["primary_speeds"]) else 0.0,
            "primary_displacement_mean": float(np.mean(cen_stats["primary_displacements"])) if len(cen_stats["primary_displacements"]) else 0.0,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_pdf_report(out_dir: str, ordered_pngs: List[str], pdf_name: str):
    plt, PdfPages = _prepare_matplotlib()
    pdf_path = os.path.join(out_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        for img_path in ordered_pngs:
            if not os.path.isfile(img_path):
                continue
            img = plt.imread(img_path)
            fig, ax = plt.subplots(figsize=(11.5, 8.0))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(img_path), fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join("outputs", "visualizations", f"raw_vs_centroid_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    log_state = None
    if not args.disable_file_logging:
        log_state, log_path = start_run_logging(args.log_dir, script_name="viz")
        print(f"[run-log] Capturing stdout/stderr to {log_path}")

    try:
        print("Loading datasets...")
        raw_trajs, raw_masks = load_split(args.raw_dir, args.split)
        cen_trajs, cen_masks = load_split(args.centroid_dir, args.split)

        print(f"Raw split samples: {len(raw_trajs)}")
        print(f"Centroid split samples: {len(cen_trajs)}")
        if len(raw_trajs) == 0 or len(cen_trajs) == 0:
            raise RuntimeError("Cannot visualize empty split: raw or centroid sample count is zero.")

        raw_sample_idxs = sample_indices(len(raw_trajs), args.num_samples, args.seed)
        cen_sample_idxs = sample_indices(len(cen_trajs), args.num_samples, args.seed + 13)
        centroid_metadata, centroid_ped_list = load_centroid_match_data(args.centroid_dir, args.split)
        matched_raw_idxs, matched_cen_idxs = build_matched_sample_pairs(
            raw_total=len(raw_trajs),
            cen_total=len(cen_trajs),
            centroid_ped_list=centroid_ped_list,
            centroid_metadata=centroid_metadata,
            num_samples=args.num_samples,
            seed=args.seed + 777,
        )
        if matched_raw_idxs and matched_cen_idxs:
            pair_raw_idxs, pair_cen_idxs = matched_raw_idxs, matched_cen_idxs
            print(
                f"Using metadata-matched pairs: {len(pair_raw_idxs)} "
                "(centroid -> source_sample_index)"
            )
        else:
            pair_raw_idxs, pair_cen_idxs = raw_sample_idxs, cen_sample_idxs
            print("Metadata matching unavailable; using independent random samples for pair plots.")

        # Cap stats sample count for speed/memory
        raw_stats_idxs = sample_indices(len(raw_trajs), min(args.stats_max_samples, len(raw_trajs)), args.seed + 101)
        cen_stats_idxs = sample_indices(len(cen_trajs), min(args.stats_max_samples, len(cen_trajs)), args.seed + 202)

        print("Computing summary stats...")
        raw_stats = compute_stats(
            raw_trajs,
            raw_masks,
            raw_stats_idxs,
            args.max_heatmap_points,
            args.hist_len,
        )
        cen_stats = compute_stats(
            cen_trajs,
            cen_masks,
            cen_stats_idxs,
            args.max_heatmap_points,
            args.hist_len,
        )

        print("Generating sample trajectory grids...")
        before_after_abs = os.path.join(out_dir, "00_before_vs_after_raw_vs_centroid.png")
        before_after_norm = os.path.join(
            out_dir,
            "00_before_vs_after_raw_vs_centroid_normalized.png",
        )
        raw_grid = os.path.join(out_dir, "01_raw_samples_grid.png")
        cen_grid = os.path.join(out_dir, "02_centroid_samples_grid.png")
        pair_grid_abs = os.path.join(out_dir, "03_raw_vs_centroid_pairs.png")
        pair_grid_norm = os.path.join(
            out_dir,
            "03_raw_vs_centroid_pairs_normalized.png",
        )

        do_abs = args.pair_coordinate_mode in ("absolute", "both")
        do_norm = args.pair_coordinate_mode in ("normalized", "both")

        if do_abs:
            make_before_after_figure(
                raw_trajs,
                raw_masks,
                pair_raw_idxs[0],
                cen_trajs,
                cen_masks,
                pair_cen_idxs[0],
                args.hist_len,
                args.max_agents_per_plot,
                before_after_abs,
                normalize_origin=False,
                share_axes=True,
            )
            make_side_by_side_pairs(
                raw_trajs,
                raw_masks,
                pair_raw_idxs,
                cen_trajs,
                cen_masks,
                pair_cen_idxs,
                args.hist_len,
                args.max_agents_per_plot,
                pair_grid_abs,
                normalize_origin=False,
                share_axes=True,
            )

        if do_norm:
            make_before_after_figure(
                raw_trajs,
                raw_masks,
                pair_raw_idxs[0],
                cen_trajs,
                cen_masks,
                pair_cen_idxs[0],
                args.hist_len,
                args.max_agents_per_plot,
                before_after_norm,
                normalize_origin=True,
                share_axes=True,
            )
            make_side_by_side_pairs(
                raw_trajs,
                raw_masks,
                pair_raw_idxs,
                cen_trajs,
                cen_masks,
                pair_cen_idxs,
                args.hist_len,
                args.max_agents_per_plot,
                pair_grid_norm,
                normalize_origin=True,
                share_axes=True,
            )

        make_samples_grid(
            raw_trajs,
            raw_masks,
            raw_sample_idxs,
            args.hist_len,
            args.max_agents_per_plot,
            "Raw Dataset: Multi-Agent Trajectory Samples",
            raw_grid,
        )
        make_samples_grid(
            cen_trajs,
            cen_masks,
            cen_sample_idxs,
            args.hist_len,
            args.max_agents_per_plot,
            "Centroid Dataset: Multi-Agent Trajectory Samples",
            cen_grid,
        )

        print("Generating global distribution and heatmap comparisons...")
        raw_heatmap = os.path.join(out_dir, "04_raw_spatial_heatmap.png")
        cen_heatmap = os.path.join(out_dir, "05_centroid_spatial_heatmap.png")
        save_heatmap(raw_stats["heatmap_points"], "Raw Spatial Occupancy Heatmap", raw_heatmap)
        save_heatmap(cen_stats["heatmap_points"], "Centroid Spatial Occupancy Heatmap", cen_heatmap)

        agent_hist = os.path.join(out_dir, "06_agent_count_hist_compare.png")
        speed_hist = os.path.join(out_dir, "07_primary_speed_hist_compare.png")
        disp_hist = os.path.join(out_dir, "08_primary_displacement_hist_compare.png")
        heading_hist = os.path.join(out_dir, "09_primary_heading_hist_compare.png")
        agent_box = os.path.join(out_dir, "10_agent_count_box_compare.png")
        speed_curve = os.path.join(out_dir, "11_mean_primary_speed_curve.png")

        save_hist_compare(
            raw_stats["agent_counts"],
            cen_stats["agent_counts"],
            "Number of agents per sample",
            "Agent Count Distribution: Raw vs Centroid",
            agent_hist,
            bins=30,
        )
        save_hist_compare(
            raw_stats["primary_speeds"],
            cen_stats["primary_speeds"],
            "Primary speed (pixels/frame)",
            "Primary Speed Distribution: Raw vs Centroid",
            speed_hist,
            bins=40,
        )
        save_hist_compare(
            raw_stats["primary_displacements"],
            cen_stats["primary_displacements"],
            "Primary displacement over sequence",
            "Primary Displacement Distribution: Raw vs Centroid",
            disp_hist,
            bins=40,
        )
        save_hist_compare(
            raw_stats["primary_headings_deg"],
            cen_stats["primary_headings_deg"],
            "Primary heading delta (degrees)",
            "Primary Heading Distribution: Raw vs Centroid",
            heading_hist,
            bins=72,
        )
        save_box_compare(
            raw_stats["agent_counts"],
            cen_stats["agent_counts"],
            "Number of agents per sample",
            "Agent Count Boxplot: Raw vs Centroid",
            agent_box,
        )

        curve_len = min(
            len(raw_stats["mean_primary_speed_per_t"]),
            len(cen_stats["mean_primary_speed_per_t"]),
        )
        save_mean_speed_curve(
            raw_stats["mean_primary_speed_per_t"][:curve_len],
            cen_stats["mean_primary_speed_per_t"][:curve_len],
            args.hist_len,
            speed_curve,
        )

        summary_json = os.path.join(out_dir, "summary_stats.json")
        save_summary_json(
            raw_stats,
            cen_stats,
            args.split,
            len(raw_trajs),
            len(cen_trajs),
            summary_json,
        )

        ordered_pngs = [
            before_after_abs if do_abs else before_after_norm,
            raw_grid,
            cen_grid,
            pair_grid_abs if do_abs else pair_grid_norm,
            raw_heatmap,
            cen_heatmap,
            agent_hist,
            speed_hist,
            disp_hist,
            heading_hist,
            agent_box,
            speed_curve,
        ]
        if do_abs and do_norm:
            ordered_pngs.insert(1, before_after_norm)
            ordered_pngs.insert(5, pair_grid_norm)
        pdf_report = build_pdf_report(
            out_dir,
            ordered_pngs,
            pdf_name="raw_vs_centroid_visual_report.pdf",
        )

        print("\nVisualization package complete.")
        print(f"Output directory: {out_dir}")
        print(f"PDF report: {pdf_report}")
        print(f"Summary stats: {summary_json}")
        print(f"Raw sample indices: {raw_sample_idxs}")
        print(f"Centroid sample indices: {cen_sample_idxs}")
        print(f"Pair raw indices: {pair_raw_idxs}")
        print(f"Pair centroid indices: {pair_cen_idxs}")

    finally:
        finalize_run_logging(log_state)


if __name__ == "__main__":
    main()
