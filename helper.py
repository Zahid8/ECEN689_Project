import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import batch_process_coords, collate_batch, create_dataset
from loss import compute_multi_loss
from utils.utils import AverageMeter, save_checkpoint, update_stats

import random

# cluster weight function w = f(|C_i|)
def cluster_size_to_weight(size, alpha=1.0): # can add other weight functions here later
    return 1.0 + alpha * float(np.log(size))

# weighted STES score 
# S_x(X_1, C~_i) = w_i * S(X_1, C_i)
def select_examples_weighted_stes(query_idx, stes_candidates_sorted, cluster_sizes, num_example, 
                                  topk_window=32, alpha=1.0, temperature=1.0, stes_scores=None):
    if num_example <= 0 or not stes_candidates_sorted:
        return []

    n = min(max(topk_window, num_example), len(stes_candidates_sorted))
    window = stes_candidates_sorted[:n]

    if stes_scores is not None and query_idx in stes_scores:
        S = np.asarray(stes_scores[query_idx][:n], dtype=np.float64)
    else:
        ranks = np.arange(n, 0, -1, dtype=np.float64)
        S = ranks / ranks.max()

    if cluster_sizes is None:
        w = np.ones(n, dtype=np.float64)
    else:
        sizes = np.array(
            [cluster_sizes.get(i, 0) for i in window], dtype=np.float64
        )
        max_size = sizes.max() if sizes.size else 1.0
        w = np.array([cluster_size_to_weight(s, alpha) for s in sizes], dtype=np.float64)

    S_w = w * S
    order = np.argsort(-S_w, kind="stable")[:num_example]
    picked = [window[i] for i in order]

    return picked[::-1]

def build_example_selector(cfg, cluster_sizes=None, stes_scores=None):

    method = cfg.dataset.prompting
    num_example = int(cfg.dataset.num_example)

    def selector(query_idx, candidates):
        if method == "sim":
            k = min(num_example, len(candidates))
            return candidates[:k][::-1]
        elif method == "weighted_stes":
            return select_examples_weighted_stes(query_idx, candidates, cluster_sizes, num_example, stes_scores=stes_scores)

    return selector

def train(
    cfg,
    epoch,
    dataloader_train,
    model,
    optimizer,
    scheduler=None,
    stats={},
):
    split = "train"
    train_steps = len(dataloader_train)
    dataiter = iter(dataloader_train)
    losses_avg = defaultdict(lambda: AverageMeter())
    summary = [
        f"{split}",
        f"{str(epoch).zfill(3)}",
    ]
    print(" | ".join(summary))

    for i in tqdm(range(train_steps)):
        model.train()
        optimizer.zero_grad()
        try:
            trajs, masks, padding_mask = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader_train)
            trajs, masks, padding_mask = next(dataiter)

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

        res = compute_multi_loss(
            cfg,
            hist_trajs,
            fut_trajs,
            example_primary_rel_pos,
            padding_mask,
            model,
            training=True,
        )
    res["loss"].backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg["training"]["max_grad_norm"]
    )
    optimizer.step()
    # Update scheduler
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



def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_batch,
        shuffle=(split == "train"),
        pin_memory=cfg.training.pin_mem,
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
