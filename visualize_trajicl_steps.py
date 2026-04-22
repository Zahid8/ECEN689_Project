import argparse
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from dataset import batch_process_coords, collate_batch
from helper import set_seed
from model import create_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TrajICL step visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize TrajICL two-step prompting and prediction on val split."
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
        help="Prompting method of first-step retrieval (random/sim).",
    )
    parser.add_argument(
        "--num_example",
        type=int,
        default=4,
        help="Number of prompts in each step.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="How many val queries to visualize.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start sample index in val valid list.",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=128,
        help="Second-step retrieval candidates from top sequence neighbors.",
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
        "--output_dir",
        type=str,
        default="results/trajicl_steps",
        help="Directory to save figures.",
    )
    return parser.parse_args()


def build_val_dataset(cfg: Any):
    """Create dataset directly to keep query index/fold mapping."""
    from dataset import create_dataset

    return create_dataset(split="val", cfg=cfg)


def get_prompt_from_dataset(dataset: Any, fold: int, example_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch one example trajectory/mask by prompt index from current fold."""
    if dataset.pool_dc_by_fold is not None:
        pool_bundle = dataset.pool_dc_by_fold[fold]
        return pool_bundle["trajs"][example_idx], pool_bundle["masks"][example_idx]
    if dataset.trajs_dc_by_fold is not None:
        dc_bundle = dataset.trajs_dc_by_fold[fold]
        traj_example = dc_bundle["trajs"][example_idx]
        masks_dc = dc_bundle.get("masks")
        if masks_dc is not None:
            return traj_example, masks_dc[example_idx]
        return traj_example, dataset.masks[example_idx]
    return dataset.trajs[example_idx], dataset.masks[example_idx]


def build_input_from_indices(
    dataset: Any,
    fold: int,
    query_idx: int,
    example_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (trajs, masks, padding_mask) with examples + query as one-item batch."""
    trajs_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []
    for example_idx in example_indices:
        traj_example, mask_example = get_prompt_from_dataset(dataset, fold, example_idx)
        trajs_list.append(traj_example)
        masks_list.append(mask_example)

    trajs_list.append(dataset.trajs[query_idx])
    masks_list.append(dataset.masks[query_idx])

    trajs, masks, padding_mask = collate_batch([(trajs_list, masks_list)])
    return trajs, masks, padding_mask


def infer_one_step(
    cfg: Any,
    model: torch.nn.Module,
    trajs: torch.Tensor,
    masks: torch.Tensor,
    padding_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run one model forward and return key tensors for plotting."""
    hist_trajs, _, fut_trajs, _, example_rel_pos, padding_mask = batch_process_coords(
        trajs,
        masks,
        padding_mask,
        cfg,
        training=False,
        eval_robust=False,
    )
    with torch.no_grad():
        output = model(
            hist_trajs,
            fut_trajs.clone(),
            padding_mask,
            training=False,
            example_primary_rel_pos=example_rel_pos,
        )
    return {
        "hist_trajs": hist_trajs,
        "fut_trajs": fut_trajs,
        "pred": output["primary_pred_fut_traj"],
    }


def select_pred_mode(pred: torch.Tensor) -> torch.Tensor:
    """Pick one mode for visualization."""
    # pred: [1, K, T, 2]
    return pred[0, 0].detach().cpu()


def get_first_step_example_indices(
    dataset: Any,
    fold: int,
    query_idx: int,
    num_example: int,
    prompting_method: str,
) -> List[int]:
    """Use dataset prompting policy for step-1 retrieval."""
    if prompting_method == "random":
        import random

        candidates = list(dataset.similarity_dicts[fold][query_idx])
        if len(candidates) == 0 or num_example <= 0:
            return []
        k = min(num_example, len(candidates))
        return random.sample(candidates, k)

    candidates = list(dataset.similarity_dicts[fold][query_idx])
    k = min(num_example, len(candidates))
    return candidates[:k][::-1]


def get_primary_seq_local(traj: torch.Tensor, hist_len: int) -> torch.Tensor:
    """Get primary trajectory [T,2] in local coordinates (current history frame as origin)."""
    seq = traj[0, :, 0, :2].float()
    origin = seq[hist_len - 1 : hist_len]
    return seq - origin


def second_step_retrieval(
    dataset: Any,
    fold: int,
    query_idx: int,
    pseudo_seq: torch.Tensor,
    num_example: int,
    candidate_pool_size: int,
) -> List[int]:
    """Retrieve step-2 prompts using first prediction as pseudo future."""
    if num_example <= 0:
        return []

    if dataset.similarity_dicts_seq is not None and query_idx in dataset.similarity_dicts_seq[fold]:
        candidate_indices = list(dataset.similarity_dicts_seq[fold][query_idx])[:candidate_pool_size]
    else:
        candidate_indices = list(dataset.similarity_dicts[fold][query_idx])[:candidate_pool_size]

    scored: List[Tuple[float, int]] = []
    hist_len = int(dataset.hist_len)
    for candidate_idx in candidate_indices:
        candidate_traj, _ = get_prompt_from_dataset(dataset, fold, candidate_idx)
        candidate_seq = get_primary_seq_local(candidate_traj, hist_len=hist_len)
        steps = min(pseudo_seq.shape[0], candidate_seq.shape[0])
        if steps <= 0:
            continue
        dist = torch.norm(pseudo_seq[:steps] - candidate_seq[:steps], dim=-1).mean().item()
        scored.append((dist, candidate_idx))

    scored.sort(key=lambda x: x[0])
    return [idx for _, idx in scored[:num_example]]


def extract_scene_for_plot(query_traj: torch.Tensor, hist_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return target past and surrounding past in target-centric coordinates."""
    xy = query_traj[:, :, 0, :2].float()
    origin = xy[0, hist_len - 1 : hist_len]
    xy = xy - origin
    target_past = xy[0, :hist_len]
    surrounding_past = xy[1:, :hist_len] if xy.shape[0] > 1 else torch.empty(0, hist_len, 2)
    return target_past, surrounding_past


def plot_example_primary(
    ax: plt.Axes,
    dataset: Any,
    fold: int,
    example_indices: Sequence[int],
    query_origin: torch.Tensor,
    hist_len: int,
) -> None:
    """Plot selected example primary trajectories in gray."""
    for example_idx in example_indices:
        traj_example, _ = get_prompt_from_dataset(dataset, fold, example_idx)
        seq = traj_example[0, :, 0, :2].float() - query_origin
        seq_hist = seq[:hist_len].cpu()
        ax.plot(seq_hist[:, 0], seq_hist[:, 1], color="gray", linewidth=1.0, alpha=0.9)


def visualize_sample(
    dataset: Any,
    fold: int,
    query_idx: int,
    step1_example_indices: Sequence[int],
    step2_example_indices: Sequence[int],
    pred_step1: torch.Tensor,
    pred_step2: torch.Tensor,
    gt_fut: torch.Tensor,
    output_path: str,
) -> None:
    """Generate and save 3-panel visualization for one query."""
    hist_len = int(dataset.hist_len)
    query_traj = dataset.trajs[query_idx]
    query_xy = query_traj[:, :, 0, :2].float()
    query_origin = query_xy[0, hist_len - 1]
    target_past, surrounding_past = extract_scene_for_plot(query_traj, hist_len=hist_len)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    ax1, ax2, ax3 = axes[0]

    # Step 1
    ax1.plot(target_past[:, 0], target_past[:, 1], color="blue", linewidth=2.0)
    for agent_idx in range(surrounding_past.shape[0]):
        agent = surrounding_past[agent_idx]
        ax1.plot(agent[:, 0], agent[:, 1], color="black", linewidth=1.0, alpha=0.8)
    plot_example_primary(ax1, dataset, fold, step1_example_indices, query_origin, hist_len)
    ax1.set_title("step 1")

    # Step 2
    ax2.plot(target_past[:, 0], target_past[:, 1], color="blue", linewidth=2.0)
    for agent_idx in range(surrounding_past.shape[0]):
        agent = surrounding_past[agent_idx]
        ax2.plot(agent[:, 0], agent[:, 1], color="black", linewidth=1.0, alpha=0.8)
    pred1 = pred_step1.detach().cpu()
    link_x = [target_past[-1, 0].item(), pred1[0, 0].item()]
    link_y = [target_past[-1, 1].item(), pred1[0, 1].item()]
    ax2.plot(link_x, link_y, color="green", linewidth=2.0)
    ax2.plot(pred1[:, 0], pred1[:, 1], color="green", linewidth=2.0)
    plot_example_primary(ax2, dataset, fold, step2_example_indices, query_origin, hist_len)
    ax2.set_title("step 2")

    # Result
    pred2 = pred_step2.detach().cpu()
    gt = gt_fut.detach().cpu()
    ax3.plot(target_past[:, 0], target_past[:, 1], color="blue", linewidth=1.7, alpha=0.8)
    ax3.plot(pred2[:, 0], pred2[:, 1], color="green", linewidth=2.3)
    ax3.plot(gt[:, 0], gt[:, 1], color="red", linewidth=2.3)
    ax3.set_title("result")

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Run TrajICL step visualization over validation queries."""
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

    dataset_val = build_val_dataset(cfg)
    total = len(dataset_val)
    start = max(0, args.start_index)
    end = min(total, start + args.num_samples)
    print(f"Total val samples: {total}, visualizing [{start}, {end})")

    for sample_i in range(start, end):
        fold, query_idx = dataset_val.valid_indices_fold_pairs[sample_i]
        step1_examples = get_first_step_example_indices(
            dataset=dataset_val,
            fold=fold,
            query_idx=query_idx,
            num_example=args.num_example,
            prompting_method=args.prompting_method,
        )
        trajs1, masks1, pad1 = build_input_from_indices(
            dataset=dataset_val,
            fold=fold,
            query_idx=query_idx,
            example_indices=step1_examples,
        )
        out1 = infer_one_step(cfg, model, trajs1, masks1, pad1)
        pred1 = select_pred_mode(out1["pred"])
        hist_target = out1["hist_trajs"][0, -1, :, 0].detach().cpu()
        pseudo_seq = torch.cat([hist_target, pred1], dim=0)

        step2_examples = second_step_retrieval(
            dataset=dataset_val,
            fold=fold,
            query_idx=query_idx,
            pseudo_seq=pseudo_seq,
            num_example=args.num_example,
            candidate_pool_size=args.candidate_pool_size,
        )
        trajs2, masks2, pad2 = build_input_from_indices(
            dataset=dataset_val,
            fold=fold,
            query_idx=query_idx,
            example_indices=step2_examples,
        )
        out2 = infer_one_step(cfg, model, trajs2, masks2, pad2)
        pred2 = select_pred_mode(out2["pred"])
        gt = out2["fut_trajs"][0, -1, :, 0].detach().cpu()

        output_path = os.path.join(
            args.output_dir,
            f"val_{sample_i:05d}_fold_{fold}_query_{query_idx}.png",
        )
        visualize_sample(
            dataset=dataset_val,
            fold=fold,
            query_idx=query_idx,
            step1_example_indices=step1_examples,
            step2_example_indices=step2_examples,
            pred_step1=pred1,
            pred_step2=pred2,
            gt_fut=gt,
            output_path=output_path,
        )
        print(
            f"[{sample_i}] fold={fold} query={query_idx} "
            f"step1={step1_examples} step2={step2_examples} -> {output_path}"
        )


if __name__ == "__main__":
    main()
