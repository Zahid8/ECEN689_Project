import argparse
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from dataset import batch_process_coords, create_dataset
from helper import (
    build_single_query_input,
    get_pool_example,
    run_single_inference,
    select_step1_examples_stes,
    select_step2_examples_with_pges,
    set_seed,
)
from model import create_model
from utils.metrics import mse_primary_min_ade_loss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TrajICL step visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize TrajICL two-step prompting and prediction on val split (one figure)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/TrajICL/worthy-firebrand-50/best_val_checkpoint.pth.tar",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="motsynth",
        help="Dataset name for validation split.",
    )
    parser.add_argument(
        "--prompting_method",
        type=str,
        default="sim",
        help="Prompting for first-step retrieval: random or sim (STES).",
    )
    parser.add_argument(
        "--num_example",
        type=int,
        default=4,
        help="Number of in-context examples (M) in each step.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Val sample index in valid list for the one figure.",
    )
    parser.add_argument(
        "--pges_candidate_top_n",
        type=int,
        default=128,
        help="STES top-N pool used before PG-ES (same as eval).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda/cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/trajicl_steps/trajicl_step_viz.png",
        help="Path to save the single output figure.",
    )
    parser.add_argument(
        "--num_step1_viz",
        type=int,
        default=5,
        help="Number of first-stage prediction modes to draw on the middle panel (out of K).",
    )
    return parser.parse_args()


def _extract_scene_past(
    query_traj: torch.Tensor, hist_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target past, surrounding past, and raw origin (last hist frame) of query primary."""
    query_xy = query_traj[:, :, 0, :2].float()
    origin = query_xy[0, hist_len - 1 : hist_len]
    centered = query_xy - origin
    target_past = centered[0, :hist_len]
    n_agents = int(query_traj.shape[0])
    if n_agents > 1:
        surrounding_past = centered[1:, :hist_len]
    else:
        surrounding_past = torch.empty(0, hist_len, 2, dtype=target_past.dtype)
    return target_past, surrounding_past, origin.squeeze(0)


def _plot_past_and_neighbors(
    ax: Any,
    target_past: torch.Tensor,
    surrounding_past: torch.Tensor,
    use_legend: bool,
) -> None:
    """Plot target (blue) and surrounding agents (black) with at most one label each.

    Coordinates are unscaled, target-centric (``model / resize``), matching
    :func:`batch_process_coords` for the **query** channel.
    """
    target_past = target_past.cpu()
    ax.plot(
        target_past[:, 0],
        target_past[:, 1],
        color="blue",
        linewidth=2.0,
        label="target past" if use_legend else None,
    )
    for k in range(surrounding_past.shape[0]):
        ag = surrounding_past[k]
        ag = ag.cpu() if torch.is_tensor(ag) else ag
        ax.plot(
            ag[:, 0],
            ag[:, 1],
            color="black",
            linewidth=1.0,
            alpha=0.8,
            label="surrounding" if (use_legend and k == 0) else None,
        )


def _plot_step1_predictions_subset(
    ax: Any,
    pred1: torch.Tensor,
    hist_target: torch.Tensor,
    resize: float,
    num_curves: int,
    use_legend: bool,
) -> None:
    """Draw a subset of first-stage (STES) future modes in green (unscaled, query frame)."""
    pred1 = pred1.detach().float().cpu()
    hist_target = hist_target.detach().float().cpu()
    k_all = int(pred1.shape[0])
    if k_all == 0 or num_curves <= 0:
        return
    n_pick = min(num_curves, k_all)
    if n_pick == k_all:
        mode_indices = list(range(k_all))
    else:
        raw_idx = [int(x) for x in torch.linspace(0, k_all - 1, n_pick).round().long().tolist()]
        mode_indices = []
        for ri in raw_idx:
            if ri not in mode_indices:
                mode_indices.append(ri)
    past_end = hist_target[-1] / float(resize)
    label_used = False
    for mi in mode_indices:
        pred_u = (pred1[mi] / float(resize)).numpy()
        ax.plot(
            [past_end[0].item(), pred_u[0, 0]],
            [past_end[1].item(), pred_u[0, 1]],
            color="green",
            linewidth=1.0,
            alpha=0.6,
            label="stage-1 pred" if (use_legend and not label_used) else None,
        )
        label_used = True
        ax.plot(
            pred_u[:, 0],
            pred_u[:, 1],
            color="green",
            linewidth=1.2,
            alpha=0.6,
        )


def _plot_example_histories(
    ax: Any,
    dataset: Any,
    fold: int,
    example_indices: List[int],
    query_origin: torch.Tensor,
    hist_len: int,
    use_legend: bool,
    legend_label: str = "example",
) -> None:
    """Plot example primary past trajectories in gray (label once).

    Args:
        ax: Matplotlib axes.
        dataset: Validation dataset with pool access.
        fold: Cross-validation fold index.
        example_indices: Pool indices to draw (STES or PG-ES stage).
        query_origin: Query primary position at last history step (unscaled).
        hist_len: History length in time steps.
        use_legend: Whether to attach a legend entry to the first curve.
        legend_label: Legend text for the first example curve.
    """
    for j, example_idx in enumerate(example_indices):
        traj_example, _ = get_pool_example(dataset, fold, example_idx)
        seq = traj_example[0, :, 0, :2].float() - query_origin
        seq_hist = seq[:hist_len].cpu()
        ax.plot(
            seq_hist[:, 0],
            seq_hist[:, 1],
            color="gray",
            linewidth=1.0,
            alpha=0.5,
            label=legend_label if (use_legend and j == 0) else None,
        )


def _save_legended_axes(fig: Any, axes_row: List[Any]) -> None:
    for ax in axes_row:
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, la in zip(handles, labels):
            if la and la not in uniq:
                uniq[la] = h
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), loc="best", fontsize=8)
    fig.tight_layout()


def _visualize_one(
    dataset: Any,
    fold: int,
    query_idx: int,
    step1_ids: List[int],
    step2_ids: List[int],
    pred1: torch.Tensor,
    pred2_multimodal: torch.Tensor,
    gt_fut: torch.Tensor,
    hist_target: torch.Tensor,
    surrounding_hist: torch.Tensor,
    resize: float,
    num_step1_viz: int,
    output_path: str,
) -> None:
    """Render three subplots: STES, STES+PG-ES context with stage-1 preds, final minADE pred vs GT.

    ``hist_target`` / ``surrounding_hist`` are from ``batch_process_coords`` (query
    channel); plots use unscaled coordinates (``/ resize``) so they match
    ``pred1`` / ``pred2`` after division by ``resize``.
    """
    hist_len = int(dataset.hist_len)
    query_traj = dataset.trajs[query_idx]
    _, _, query_origin = _extract_scene_past(query_traj, hist_len=hist_len)
    target_past = hist_target / float(resize)
    if surrounding_hist.numel() > 0:
        surrounding_past = (surrounding_hist / float(resize)).permute(1, 0, 2)
    else:
        surrounding_past = torch.empty(0, hist_len, 2, dtype=target_past.dtype)

    with torch.no_grad():
        _, min_idx = mse_primary_min_ade_loss(pred2_multimodal, gt_fut)
    mode_i = int(min_idx[0].item())
    pred_best = (pred2_multimodal[0, mode_i] / float(resize)).detach().cpu()
    gt_cpu = (gt_fut[0] / float(resize)).detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    ax1, ax2, ax3 = axes[0]

    _plot_past_and_neighbors(ax1, target_past, surrounding_past, use_legend=True)
    _plot_example_histories(
        ax1,
        dataset,
        fold,
        step1_ids,
        query_origin,
        hist_len,
        use_legend=True,
        legend_label="STES example",
    )
    ax1.set_title("step 1 (STES)")

    _plot_past_and_neighbors(ax2, target_past, surrounding_past, use_legend=True)
    _plot_step1_predictions_subset(
        ax2,
        pred1,
        hist_target,
        resize,
        num_step1_viz,
        use_legend=True,
    )
    # Gray trajectories: indices from select_step2_examples_with_pges (PG-ES), not step1.
    _plot_example_histories(
        ax2,
        dataset,
        fold,
        step2_ids,
        query_origin,
        hist_len,
        use_legend=True,
        legend_label="PG-ES example",
    )
    ax2.set_title("step 2 (STES + PG-ES)")

    _plot_past_and_neighbors(ax3, target_past, surrounding_past, use_legend=True)
    ax3.plot(
        pred_best[:, 0],
        pred_best[:, 1],
        color="green",
        linewidth=2.2,
        label="prediction",
    )
    ax3.plot(
        gt_cpu[:, 0],
        gt_cpu[:, 1],
        color="red",
        linewidth=2.2,
        label="ground truth",
    )
    ax3.set_title("result (stage-2, minADE mode)")

    for ax in (ax1, ax2, ax3):
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")

    _save_legended_axes(fig, [ax1, ax2, ax3])
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build val sample, run STES then PG-ES, save one three-panel figure."""
    args = parse_args()
    set_seed(args.seed)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    cfg = OmegaConf.create(checkpoint["cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.device = args.device
    cfg.dataset.name = args.dataset_name
    cfg.dataset.prompting = args.prompting_method
    cfg.dataset.num_example = args.num_example

    model = create_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device)
    model.eval()

    dataset_val = create_dataset(split="val", cfg=cfg)
    total = len(dataset_val)
    if args.start_index < 0 or args.start_index >= total:
        raise ValueError(
            f"start_index {args.start_index} out of range (val len={total})"
        )
    sample_i = int(args.start_index)
    fold, query_idx = dataset_val.valid_indices_fold_pairs[sample_i]

    num_ex = int(args.num_example)
    step1_ids = select_step1_examples_stes(
        dataset_val, fold, query_idx, num_ex, str(cfg.dataset.prompting)
    )
    trajs1, masks1, pad1 = build_single_query_input(
        dataset_val, fold, query_idx, step1_ids
    )
    pred1, _ = run_single_inference(cfg, model, trajs1, masks1, pad1)
    pred1 = pred1[0]  # [K, T, 2] on device

    hist_trajs, _, _, _, _, _ = batch_process_coords(
        trajs1, masks1, pad1, cfg, training=False, eval_robust=False
    )
    hist_target = hist_trajs[0, -1, :, 0]  # [hist_len, 2] on device, query channel
    n_agent = int(hist_trajs.shape[3])
    if n_agent > 1:
        surrounding_hist = hist_trajs[0, -1, :, 1:, :]
    else:
        surrounding_hist = hist_trajs.new_empty(0)

    step2_ids = select_step2_examples_with_pges(
        dataset_val,
        fold,
        query_idx,
        pred1,
        hist_target,
        cfg,
        num_ex,
        int(args.pges_candidate_top_n),
    )
    trajs2, masks2, pad2 = build_single_query_input(
        dataset_val, fold, query_idx, step2_ids
    )
    pred2, gt = run_single_inference(cfg, model, trajs2, masks2, pad2)

    _visualize_one(
        dataset_val,
        fold,
        query_idx,
        step1_ids,
        step2_ids,
        pred1,
        pred2,
        gt,
        hist_target,
        surrounding_hist,
        float(cfg.training.resize),
        int(args.num_step1_viz),
        args.output,
    )
    print(
        f"Saved: {args.output} | sample={sample_i} fold={fold} query={query_idx} | "
        f"STES step1={step1_ids} | PG-ES step2={step2_ids}"
    )


if __name__ == "__main__":
    main()
