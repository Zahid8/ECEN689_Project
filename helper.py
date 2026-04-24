import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import batch_process_coords, collate_batch, create_dataset
from loss import compute_multi_loss
from utils.metrics import mse_primary_min_ade_loss, mse_primary_min_fde_loss
from utils.utils import AverageMeter, save_checkpoint, update_stats


def train(
    cfg,
    epoch,
    dataloader_train,
    model,
    optimizer,
    scheduler=None,
    stats={},
    scaler: GradScaler | None = None,
):
    split = "train"
    losses_avg = defaultdict(lambda: AverageMeter())
    summary = [
        f"{split}",
        f"{str(epoch).zfill(3)}",
    ]
    print(" | ".join(summary))

    use_amp = bool(getattr(cfg.training, "use_amp", cfg.device == "cuda"))
    model.train()

    for trajs, masks, padding_mask in tqdm(dataloader_train, total=len(dataloader_train)):
        optimizer.zero_grad(set_to_none=True)

        B = trajs.shape[0]
        hist_trajs, _, fut_trajs, _, example_primary_rel_pos, padding_mask = (
            batch_process_coords(
                trajs,
                masks,
                padding_mask,
                cfg,
                training=True,
            )
        )

        with autocast(enabled=use_amp):
            res = compute_multi_loss(
                cfg,
                hist_trajs,
                fut_trajs,
                example_primary_rel_pos,
                padding_mask,
                model,
                training=True,
            )

        if use_amp and scaler is not None:
            scaler.scale(res["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            optimizer.step()

        # Update scheduler at every training step (batch).
        if scheduler is not None:
            scheduler.step()

        for key, value in res.items():
            if key == "loss":
                losses_avg[key].update(value.item(), B)
            else:
                losses_avg[key].update(value, B)

    stats = update_stats(
        stats,
        losses_avg,
        split,
    )

    return stats


def evaluate(split, cfg, epoch, model, dataloader, stats, eval_robust=False):
    eval_steps = len(dataloader)
    dataiter = iter(dataloader)
    losses_avg = defaultdict(lambda: AverageMeter())
    summary = [
        f"{split}",
        f"{str(epoch).zfill(3)}",
    ]
    print(" | ".join(summary))
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(eval_steps)):
            try:
                trajs, masks, padding_mask = next(dataiter)
            except StopIteration:
                break
            B = trajs.shape[0]
            hist_trajs, _, fut_trajs, _, example_primary_rel_pos, padding_mask = (
                batch_process_coords(
                    trajs,
                    masks,
                    padding_mask,
                    cfg,
                    training=False,
                    eval_robust=eval_robust,
                )
            )

            res = compute_multi_loss(
                cfg,
                hist_trajs,
                fut_trajs,
                example_primary_rel_pos,
                padding_mask,
                model,
                training=False,
            )

            for key, value in res.items():
                if key == "loss":
                    losses_avg[key].update(value.item(), B)
                else:
                    losses_avg[key].update(value, B)

    stats = update_stats(
        stats,
        losses_avg,
        split,
    )

    return stats


def get_pool_example(
    dataset: Any, fold: int, example_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch one prompt trajectory and mask from the fold-specific pool."""
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


_get_prompt_from_dataset = get_pool_example


def build_single_query_input(
    dataset: Any,
    fold: int,
    query_idx: int,
    example_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one-sample batch tensors using selected examples plus query."""
    trajs_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []
    for example_idx in example_indices:
        traj_example, mask_example = get_pool_example(dataset, fold, example_idx)
        trajs_list.append(traj_example)
        masks_list.append(mask_example)

    trajs_list.append(dataset.trajs[query_idx])
    masks_list.append(dataset.masks[query_idx])

    return collate_batch([(trajs_list, masks_list)])


def run_single_inference(
    cfg: Any,
    model: torch.nn.Module,
    trajs: torch.Tensor,
    masks: torch.Tensor,
    padding_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass and return (multimodal prediction, target GT future)."""
    hist_trajs, _, fut_trajs, _, example_primary_rel_pos, padding_mask = batch_process_coords(
        trajs,
        masks,
        padding_mask,
        cfg,
        training=False,
        eval_robust=False,
    )
    output = model(
        hist_trajs,
        fut_trajs.clone(),
        padding_mask,
        training=False,
        example_primary_rel_pos=example_primary_rel_pos,
    )
    primary_pred_fut_traj = output["primary_pred_fut_traj"]
    primary_gt_fut_traj = fut_trajs[:, -1, :, 0]
    return primary_pred_fut_traj, primary_gt_fut_traj


# Backward compatibility for in-repo callers using private names
_build_single_query_input = build_single_query_input
_run_single_inference = run_single_inference


def apply_cluster_weighted_similarity(
    similar_traj_dict: Dict[int, List[int]],
    similar_scores_dict: Dict[int, Sequence[float]],
    cluster_size_lookup: Dict[int, int],
    alpha: float,
    max_similar: int | None = None,
) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """Re-rank candidates by weighted similarity S*(1 + alpha*log(1+n)).

    Args:
        similar_traj_dict: Query-to-candidate ranking from base similarity.
        similar_scores_dict: Query-to-score list aligned with candidate order.
        cluster_size_lookup: Candidate index to cluster size ``n``.
        alpha: Cluster weighting strength; ``alpha <= 0`` keeps original ranking.
        max_similar: Optional cap on returned candidate count per query.

    Returns:
        Weighted candidate index lists and their weighted scores.
    """
    if alpha <= 0:
        passthrough_scores = {
            int(query_idx): [float(score) for score in scores]
            for query_idx, scores in similar_scores_dict.items()
        }
        return similar_traj_dict, passthrough_scores

    weighted_traj_dict: Dict[int, List[int]] = {}
    weighted_scores_dict: Dict[int, List[float]] = {}
    for query_idx, candidate_indices in similar_traj_dict.items():
        base_scores = list(similar_scores_dict.get(query_idx, []))
        pair_len = min(len(candidate_indices), len(base_scores))
        if pair_len == 0:
            weighted_traj_dict[query_idx] = []
            weighted_scores_dict[query_idx] = []
            continue

        ranked_pairs: List[Tuple[int, float]] = []
        for candidate_idx, score in zip(candidate_indices[:pair_len], base_scores[:pair_len]):
            cluster_size = int(cluster_size_lookup.get(int(candidate_idx), 1))
            weight = 1.0 + alpha * float(np.log1p(cluster_size))
            weighted_score = float(score) * weight
            ranked_pairs.append((int(candidate_idx), weighted_score))

        ranked_pairs.sort(key=lambda x: x[1], reverse=True)
        if max_similar is not None:
            ranked_pairs = ranked_pairs[:max_similar]

        weighted_traj_dict[query_idx] = [candidate_idx for candidate_idx, _ in ranked_pairs]
        weighted_scores_dict[query_idx] = [score for _, score in ranked_pairs]

    return weighted_traj_dict, weighted_scores_dict


def get_cluster_size_lookup(
    dataset: Any,
    fold: int,
    candidate_indices: Sequence[int],
) -> Dict[int, int]:
    """Build candidate-index -> cluster-size map from dataset metadata."""
    if not hasattr(dataset, "cluster_meta_by_fold") or dataset.cluster_meta_by_fold is None:
        return {}
    if fold >= len(dataset.cluster_meta_by_fold):
        return {}
    fold_meta = dataset.cluster_meta_by_fold[fold]
    lookup: Dict[int, int] = {}
    for candidate_idx in candidate_indices:
        meta = fold_meta.get(int(candidate_idx), None)
        if meta is None:
            lookup[int(candidate_idx)] = 1
        else:
            lookup[int(candidate_idx)] = int(meta.get("cluster_weight", 1))
    return lookup


def pges_query_centric_primary_seq(
    candidate_traj: torch.Tensor,
    query_traj: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Primary trajectory in the query last-frame frame with training ``resize``.

    This matches the first re-centering in :func:`dataset.batch_process_coords`
    (subtract query primary at ``hist_len - 1`` after ``resize``) without the
    per-prompt re-centering to each example's own primary. PG-ES must compare
    :math:`[X, \\hat{Y}]` with :math:`[\\tilde{X}, \\tilde{Y}]` in this same frame.

    Args:
        candidate_traj: Raw trajectory of one pool example ``[N, T, J, C]`` or similar.
        query_traj: Raw trajectory of the current query; primary is agent 0.
        cfg: Config with ``model.hist_len``, ``model.fut_len``, ``training.resize``.

    Returns:
        Tensor of shape ``[L, 2]`` with ``L <= hist_len + fut_len``.
    """
    hist_len = int(cfg["model"]["hist_len"])
    fut_len = int(cfg["model"]["fut_len"])
    max_t = int(min(candidate_traj.shape[1], hist_len + fut_len))
    primary = candidate_traj[0, :max_t, 0, :2].float()
    query_last = query_traj[0, hist_len - 1, 0, :2].float()
    resize = float(cfg.training.resize)
    return resize * (primary - query_last)


def select_step2_examples_with_pges(
    dataset: Any,
    fold: int,
    query_idx: int,
    pred_step1: torch.Tensor,
    hist_target: torch.Tensor,
    cfg: Any,
    num_example: int,
    candidate_top_n: int,
) -> List[int]:
    """Select PG-ES examples via fused position + velocity cost (min over K).

    Position and velocity errors follow :func:`preprocess.compute_sim_matrix`
    (mean L2 over time; velocity as consecutive frame differences). Per-candidate
    costs are min-max normalized across the candidate pool, then combined as
    ``dist_weight * dist_norm + vel_weight * vel_norm`` like
    :func:`preprocess.process_file_optimized`.

    Returns:
        Example indices **worst-to-best** among the chosen set (same order as
        :func:`dataset.sim_prompting` / STES).
    """
    if num_example <= 0:
        return []

    stes_sorted = list(dataset.similarity_dicts[fold][query_idx])
    if len(stes_sorted) == 0:
        return []
    if candidate_top_n > 0:
        candidate_indices = stes_sorted[:candidate_top_n]
    else:
        candidate_indices = stes_sorted
    if len(candidate_indices) == 0:
        return []

    device = pred_step1.device
    dist_weight = float(getattr(cfg.dataset, "dist_weight", 1.0))
    vel_weight = float(getattr(cfg.dataset, "vel_weight", 1.0))
    cluster_weight_alpha = float(getattr(cfg.dataset, "cluster_weight_alpha", 0.0))
    pges_cluster_weight_alpha = float(
        getattr(cfg.dataset, "pges_cluster_weight_alpha", cluster_weight_alpha)
    )

    query_seq_by_k = torch.cat(
        [hist_target.unsqueeze(0).repeat(pred_step1.shape[0], 1, 1), pred_step1],
        dim=1,
    )  # [K, T, 2]

    query_traj = dataset.trajs[query_idx]
    candidate_seqs: List[torch.Tensor] = []
    valid_candidate_indices: List[int] = []
    for candidate_idx in candidate_indices:
        candidate_traj, _ = get_pool_example(dataset, fold, candidate_idx)
        candidate_seq = pges_query_centric_primary_seq(
            candidate_traj,
            query_traj,
            cfg,
        )
        candidate_seqs.append(candidate_seq)
        valid_candidate_indices.append(candidate_idx)

    if len(candidate_seqs) == 0:
        return []

    candidate_tensor = torch.stack(candidate_seqs, dim=0).to(device)  # [M, T, 2]
    steps = min(candidate_tensor.shape[1], query_seq_by_k.shape[1])
    candidate_tensor = candidate_tensor[:, :steps]
    query_seq_by_k = query_seq_by_k[:, :steps]

    diff = candidate_tensor.unsqueeze(1) - query_seq_by_k.unsqueeze(0)  # [M, K, T, 2]
    dist = torch.norm(diff, p=2, dim=-1).mean(dim=-1)  # [M, K]
    min_k_dist = torch.min(dist, dim=1).values  # [M]

    eps = 1e-8
    if steps >= 2 and vel_weight != 0.0:
        cand_vel = torch.diff(candidate_tensor, dim=1)  # [M, T-1, 2]
        query_vel = torch.diff(query_seq_by_k, dim=1)  # [K, T-1, 2]
        diff_vel = cand_vel.unsqueeze(1) - query_vel.unsqueeze(0)
        vel_err = torch.norm(diff_vel, p=2, dim=-1).mean(dim=-1)
        min_k_vel = torch.min(vel_err, dim=1).values
    else:
        min_k_vel = torch.zeros_like(min_k_dist)

    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        x_min = torch.min(x)
        x_max = torch.max(x)
        return (x - x_min) / (x_max - x_min + eps)

    if dist_weight != 0.0:
        dist_norm = _minmax_norm(min_k_dist)
    else:
        dist_norm = torch.zeros_like(min_k_dist)

    if vel_weight != 0.0:
        vel_norm = _minmax_norm(min_k_vel)
    else:
        vel_norm = torch.zeros_like(min_k_vel)

    combined = dist_weight * dist_norm + vel_weight * vel_norm

    top_m = min(num_example, combined.shape[0])
    if pges_cluster_weight_alpha > 0.0:
        base_similarity = (1.0 / (1.0 + combined)).detach().cpu().numpy().tolist()
        query_to_candidates = {int(query_idx): list(valid_candidate_indices)}
        query_to_scores = {int(query_idx): base_similarity}
        cluster_size_lookup = get_cluster_size_lookup(
            dataset=dataset,
            fold=fold,
            candidate_indices=valid_candidate_indices,
        )
        weighted_candidates, _ = apply_cluster_weighted_similarity(
            similar_traj_dict=query_to_candidates,
            similar_scores_dict=query_to_scores,
            cluster_size_lookup=cluster_size_lookup,
            alpha=pges_cluster_weight_alpha,
            max_similar=top_m,
        )
        best_to_worst = weighted_candidates[int(query_idx)]
    else:
        selected_positions = torch.topk(-combined, k=top_m).indices.tolist()
        best_to_worst = [valid_candidate_indices[pos] for pos in selected_positions]

    # Match ``sim_prompting`` / STES: feed examples worst-to-best (reverse of
    # similarity ranking among the chosen set).
    return best_to_worst[::-1]


_select_step2_examples_with_pges = select_step2_examples_with_pges


def select_step1_examples_stes(
    dataset: Any,
    fold: int,
    query_idx: int,
    num_example: int,
    prompting: str,
) -> List[int]:
    """Select first-stage (STES) in-context example indices, matching ``evaluate_pges``."""
    if num_example <= 0:
        return []
    if prompting == "random":
        candidates = list(dataset.similarity_dicts[fold][query_idx])
        k = min(num_example, len(candidates))
        return random.sample(candidates, k) if k > 0 else []
    return list(dataset.similarity_dicts[fold][query_idx])[:num_example][::-1]


def evaluate_pges(
    split: str,
    cfg: Any,
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    stats: Dict[str, Any],
    pges_candidate_top_n: int = 128,
) -> Dict[str, Any]:
    """Evaluate with PG-ES: first inference, second-stage retrieval, second inference."""
    losses_avg = defaultdict(lambda: AverageMeter())
    summary = [f"{split}", f"{str(epoch).zfill(3)}", "PGES"]
    print(" | ".join(summary))
    model.eval()

    dataset = dataloader.dataset
    eval_steps = len(dataset)
    num_example = int(cfg.dataset.num_example)

    with torch.no_grad():
        for sample_i in tqdm(range(eval_steps)):
            fold, query_idx = dataset.valid_indices_fold_pairs[sample_i]

            step1_examples = select_step1_examples_stes(
                dataset,
                fold,
                query_idx,
                num_example,
                str(cfg.dataset.prompting),
            )

            trajs1, masks1, pad1 = build_single_query_input(
                dataset=dataset,
                fold=fold,
                query_idx=query_idx,
                example_indices=step1_examples,
            )
            pred_step1, _ = run_single_inference(cfg, model, trajs1, masks1, pad1)
            pred_step1 = pred_step1[0]  # [K, fut_len, 2]

            hist_trajs, _, _, _, _, _ = batch_process_coords(
                trajs1,
                masks1,
                pad1,
                cfg,
                training=False,
                eval_robust=False,
            )
            hist_target = hist_trajs[0, -1, :, 0]  # [hist_len, 2]

            step2_examples = select_step2_examples_with_pges(
                dataset=dataset,
                fold=fold,
                query_idx=query_idx,
                pred_step1=pred_step1,
                hist_target=hist_target,
                cfg=cfg,
                num_example=num_example,
                candidate_top_n=pges_candidate_top_n,
            )

            trajs2, masks2, pad2 = build_single_query_input(
                dataset=dataset,
                fold=fold,
                query_idx=query_idx,
                example_indices=step2_examples,
            )
            pred_step2, gt_target = run_single_inference(cfg, model, trajs2, masks2, pad2)
            loss_ade, _ = mse_primary_min_ade_loss(pred_step2, gt_target)
            loss_fde, _ = mse_primary_min_fde_loss(pred_step2, gt_target)

            losses_avg["loss"].update(loss_ade.item(), 1)
            losses_avg["loss_ade"].update(loss_ade.item(), 1)
            losses_avg["loss_fde"].update(loss_fde.item(), 1)

    stats = update_stats(stats, losses_avg, split)
    return stats


def adjust_learning_rate(optimizer, epoch, config):
    """
    From: https://github.com/microsoft/MeshTransformer/
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs*2/3 = 100
    """
    # dct_multi_overfit_3dpw_allsize_multieval_noseg_rot_permute_id
    lr = config["training"]["lr"] * (
        config["training"]["lr_decay"] ** epoch
    )  # (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
    if "lr_drop" in config["training"] and config["training"]["lr_drop"]:
        lr = lr * (0.1 ** (epoch // (config["training"]["epochs"] * 4.0 / 5.0)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    # print("lr: ", lr)


def prepare_dataloader(cfg, subset=None):

    dataloader_train = create_dataloader(
        split="train", dataset_name=cfg.dataset.name, cfg=cfg
    )
    dataloader_val = create_dataloader(
        split="val", dataset_name=cfg.dataset.name, cfg=cfg,
    )

    return dataloader_train, dataloader_val



def set_seed(seed=0, cfg=None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    deterministic = False
    if cfg is not None:
        deterministic = bool(getattr(getattr(cfg, "training", object()), "deterministic", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Speedups on Ampere+ (A100) with minimal impact on training quality.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def create_dataloader(split, dataset_name, cfg, subset=None):
    if split == "test" and (
        dataset_name
        in [
            "orca_sim",
            "orca_sim_loc",
        ]
        or "motsynth" in dataset_name
        or "finetune" in dataset_name
    ):
        return None

    dataset = create_dataset(split, cfg, )
    print(f"{split}: {len(dataset)} trajectories")
    num_workers = int(cfg.training.num_workers)
    pin_memory = bool(cfg.training.pin_mem)
    persistent_workers = bool(getattr(cfg.training, "persistent_workers", num_workers > 0))
    prefetch_factor = int(getattr(cfg.training, "prefetch_factor", 2))
    dataloader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_batch,
        "shuffle": (split == "train"),
        "drop_last": (split == "train"),
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs,
    )

    return dataloader


def evaluate_and_update_min_val(
    cfg, epoch, model, stats, min_val, output_dir, optimizer, scheduler
):
    """
    Evaluate the model on the test set and update the minimum validation loss
    if the current validation loss is lower than the previous minimum.
    """
    val_loss = stats["loss/val"]
    if min_val["loss_val_loss"] > val_loss:
        min_val["loss_val_loss"] = val_loss
        val_ade = stats["loss_ade/val"]
        val_fde = stats["loss_fde/val"]
        min_val["loss_val_ade"] = val_ade
        min_val["loss_val_fde"] = val_fde
        print(f"min_val_loss updated! val ade: {val_ade} and val fde: {val_fde}.")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            cfg,
            output_dir,
            filename="best_val_checkpoint.pth.tar",
        )
    stats["min_val_loss/val_loss"] = min_val["loss_val_loss"]
    stats["min_val_loss/val_ade"] = min_val["loss_val_ade"]
    stats["min_val_loss/val_fde"] = min_val["loss_val_fde"]

    return stats, min_val
