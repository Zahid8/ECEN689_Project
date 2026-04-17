import os
from collections import defaultdict

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import batch_process_coords, collate_batch, create_dataset
from loss import compute_multi_loss
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
