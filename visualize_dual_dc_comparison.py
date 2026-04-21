import argparse
import os
import pickle
import random
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch


def load_pickle(path: str):
    with open(path, mode="rb") as file_obj:
        return pickle.load(file_obj)


def load_processed_bundle(processed_name: str, split: str) -> Dict[str, object]:
    """Load trajectories, masks, indices, and similarity dicts for a processed dataset."""
    save_dir = os.path.join("processed_data", processed_name)
    bundle: Dict[str, object] = {
        "save_dir": save_dir,
        "trajs": torch.load(os.path.join(save_dir, f"{split}_trajs.pt")),
        "masks": torch.load(os.path.join(save_dir, f"{split}_masks.pt")),
        "valid_indices_by_fold": load_pickle(
            os.path.join(save_dir, f"{split}_valid_indices_by_fold.pickle")
        ),
        "filename_list": load_pickle(
            os.path.join(save_dir, f"{split}_filename_list.pickle")
        ),
        "similarity_dicts": load_pickle(
            os.path.join(save_dir, f"{split}_similar_traj_dicts_hist.pickle")
        ),
    }
    return bundle


def load_dc_fold_bundle(save_dir: str, split: str, fold: int) -> Dict[str, List[torch.Tensor]]:
    """Load per-fold unique DC pool trajectories."""
    traj_path = os.path.join(save_dir, f"{split}_pool_dc_fold_{fold}_trajs.pt")
    mask_path = os.path.join(save_dir, f"{split}_pool_dc_fold_{fold}_masks.pt")
    if os.path.isfile(traj_path) and os.path.isfile(mask_path):
        return {"trajs": torch.load(traj_path), "masks": torch.load(mask_path)}

    # Backward compatibility with older aligned format.
    path_legacy = os.path.join(save_dir, f"{split}_trajs_dc_fold_{fold}.pt")
    if not os.path.isfile(path_legacy):
        raise FileNotFoundError(f"Missing DC fold tensor: {traj_path}")
    dc_bundle = torch.load(path_legacy)
    if isinstance(dc_bundle, dict):
        return dc_bundle
    return {"trajs": dc_bundle, "masks": None}


def load_dc_cluster_meta(save_dir: str, split: str, fold: int) -> Dict[int, Dict[str, Any]]:
    """Load cluster metadata for one fold if available."""
    meta_path = os.path.join(save_dir, f"{split}_cluster_meta_by_fold.pickle")
    if not os.path.isfile(meta_path):
        return {}
    meta_by_fold = load_pickle(meta_path)
    if fold < 0 or fold >= len(meta_by_fold):
        return {}
    fold_meta = meta_by_fold[fold]
    if isinstance(fold_meta, dict):
        return fold_meta
    return {}


def sample_query_indices(
    valid_indices: Sequence[int],
    n_queries: int,
    seed: int,
) -> List[int]:
    """Pick query indices from the given valid list with deterministic randomness."""
    if len(valid_indices) < n_queries:
        raise ValueError(
            f"Not enough valid queries: requested={n_queries}, available={len(valid_indices)}"
        )
    rng = random.Random(seed)
    return rng.sample(list(valid_indices), n_queries)


def sample_query_indices_distinct_scenes(
    valid_indices: Sequence[int],
    filename_list: Sequence[str],
    n_queries: int,
    seed: int,
) -> List[int]:
    """Pick queries from distinct scenes when possible."""
    scene_to_indices: Dict[str, List[int]] = {}
    for idx in valid_indices:
        scene = filename_list[idx]
        if scene not in scene_to_indices:
            scene_to_indices[scene] = []
        scene_to_indices[scene].append(idx)

    rng = random.Random(seed)
    scenes = list(scene_to_indices.keys())
    rng.shuffle(scenes)
    if len(scenes) >= n_queries:
        chosen_scenes = scenes[:n_queries]
        return [rng.choice(scene_to_indices[scene]) for scene in chosen_scenes]

    # Fallback: use all available scenes first, then fill remaining from leftovers.
    selected: List[int] = [rng.choice(scene_to_indices[scene]) for scene in scenes]
    used = set(selected)
    leftovers = [idx for idx in valid_indices if idx not in used]
    needed = n_queries - len(selected)
    if len(leftovers) < needed:
        raise ValueError(
            f"Not enough valid queries: requested={n_queries}, available={len(valid_indices)}"
        )
    selected.extend(rng.sample(leftovers, needed))
    return selected


def sample_query_indices_by_scenes(
    valid_indices: Sequence[int],
    filename_list: Sequence[str],
    scenes: Sequence[str],
    seed: int,
) -> List[int]:
    """Pick one valid query per requested scene."""
    rng = random.Random(seed)
    selected: List[int] = []
    for scene in scenes:
        scene_candidates = [idx for idx in valid_indices if filename_list[idx] == scene]
        if len(scene_candidates) == 0:
            raise ValueError(
                f"Scene '{scene}' has no valid query in selected fold."
            )
        selected.append(rng.choice(scene_candidates))
    return selected


def get_primary_xy(traj: torch.Tensor, mask: torch.Tensor) -> Tuple[List[float], List[float]]:
    """Extract valid primary-agent (x, y) points."""
    traj_xy = traj[0, :, 0, :2].detach().cpu()
    mask_1d = mask[0, :, 0].detach().cpu().bool()
    points = traj_xy[mask_1d]
    if points.numel() == 0:
        return [], []
    return points[:, 0].tolist(), points[:, 1].tolist()


def draw_query_with_topk(
    ax,
    query_idx: int,
    query_trajs: Sequence[torch.Tensor],
    query_masks: Sequence[torch.Tensor],
    pool_trajs: Sequence[torch.Tensor],
    pool_masks: Sequence[torch.Tensor],
    similarity_dict: Dict[int, List[int]],
    top_k: int,
    title: str,
) -> None:
    """Draw one panel: query trajectory + top-k similar pool trajectories."""
    similar_indices = similarity_dict[query_idx][:top_k]
    for rank, pool_idx in enumerate(similar_indices):
        px, py = get_primary_xy(pool_trajs[pool_idx], pool_masks[pool_idx])
        if len(px) > 0:
            ax.plot(px, py, linewidth=1.1, alpha=0.6, label=f"pool#{rank + 1}")

    qx, qy = get_primary_xy(query_trajs[query_idx], query_masks[query_idx])
    if len(qx) > 0:
        ax.plot(qx, qy, linewidth=2.4, color="black", label="query")
        ax.scatter([qx[-1]], [qy[-1]], color="black", s=16)

    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")


def build_dc_debug_lines(
    query_idx: int,
    selected_cluster_indices: Sequence[int],
    cluster_meta: Dict[int, Dict[str, Any]],
) -> List[str]:
    """Build human-readable debug lines for selected DC clusters."""
    lines: List[str] = [f"query={query_idx}"]
    for rank, cluster_idx in enumerate(selected_cluster_indices, start=1):
        meta = cluster_meta.get(cluster_idx, {})
        members = meta.get("member_original_indices", [])
        weight = meta.get("cluster_weight", len(members))
        lines.append(
            f"top{rank}: cid={cluster_idx}, weight={weight}, members={len(members)}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize raw-pool vs DC-pool top-k similar trajectories for matched queries."
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--raw_name", type=str, required=True)
    parser.add_argument("--dc_name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=-1, help="Use -1 for last fold.")
    parser.add_argument("--n_queries", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated scene names. If set, sample one query per scene.",
    )
    parser.add_argument(
        "--allow_same_scene",
        action="store_true",
        help="Allow repeated scenes when scenes are not specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/raw_vs_dc",
    )
    parser.add_argument(
        "--debug_clusters",
        action="store_true",
        help="Print selected DC clusters with weights/members for each query.",
    )
    parser.add_argument(
        "--overlay_cluster_members",
        action="store_true",
        help="Overlay raw member trajectories of each selected DC cluster.",
    )
    args = parser.parse_args()

    raw_bundle = load_processed_bundle(args.raw_name, args.split)
    dc_bundle = load_processed_bundle(args.dc_name, args.split)

    raw_valid_folds: List[List[int]] = raw_bundle["valid_indices_by_fold"]  # type: ignore[assignment]
    dc_valid_folds: List[List[int]] = dc_bundle["valid_indices_by_fold"]  # type: ignore[assignment]
    fold = args.fold if args.fold >= 0 else len(raw_valid_folds) - 1
    if fold >= len(raw_valid_folds) or fold >= len(dc_valid_folds):
        raise ValueError(f"Invalid fold={fold}")

    raw_valid = raw_valid_folds[fold]
    dc_valid = dc_valid_folds[fold]
    if raw_valid != dc_valid:
        shared = sorted(set(raw_valid).intersection(set(dc_valid)))
        if len(shared) < args.n_queries:
            raise ValueError(
                "Raw/DC valid indices diverged and shared set is too small. "
                f"shared={len(shared)}, required={args.n_queries}"
            )
        query_indices = sample_query_indices(shared, args.n_queries, args.seed)
    else:
        query_indices = sample_query_indices(raw_valid, args.n_queries, args.seed)

    raw_trajs: List[torch.Tensor] = raw_bundle["trajs"]  # type: ignore[assignment]
    raw_masks: List[torch.Tensor] = raw_bundle["masks"]  # type: ignore[assignment]
    raw_filename_list: List[str] = raw_bundle["filename_list"]  # type: ignore[assignment]
    raw_similarity_dicts: List[Dict[int, List[int]]] = raw_bundle["similarity_dicts"]  # type: ignore[assignment]

    dc_trajs_raw: List[torch.Tensor] = dc_bundle["trajs"]  # type: ignore[assignment]
    dc_masks_raw: List[torch.Tensor] = dc_bundle["masks"]  # type: ignore[assignment]
    dc_similarity_dicts: List[Dict[int, List[int]]] = dc_bundle["similarity_dicts"]  # type: ignore[assignment]
    dc_fold_bundle = load_dc_fold_bundle(dc_bundle["save_dir"], args.split, fold)  # type: ignore[arg-type]
    dc_cluster_meta = load_dc_cluster_meta(dc_bundle["save_dir"], args.split, fold)  # type: ignore[arg-type]
    dc_pool_trajs: List[torch.Tensor] = dc_fold_bundle["trajs"]
    if dc_fold_bundle.get("masks") is not None:
        dc_pool_masks: List[torch.Tensor] = dc_fold_bundle["masks"]  # type: ignore[assignment]
    else:
        dc_pool_masks = dc_masks_raw

    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip() != ""]
    if len(scenes) > 0:
        query_indices = sample_query_indices_by_scenes(
            valid_indices=raw_valid,
            filename_list=raw_filename_list,
            scenes=scenes,
            seed=args.seed,
        )
        args.n_queries = len(query_indices)
    elif not args.allow_same_scene:
        query_indices = sample_query_indices_distinct_scenes(
            valid_indices=query_indices,
            filename_list=raw_filename_list,
            n_queries=args.n_queries,
            seed=args.seed,
        )

    fig, axes = plt.subplots(args.n_queries, 2, figsize=(13, 4 * args.n_queries), squeeze=False)
    for row, query_idx in enumerate(query_indices):
        draw_query_with_topk(
            ax=axes[row][0],
            query_idx=query_idx,
            query_trajs=raw_trajs,
            query_masks=raw_masks,
            pool_trajs=raw_trajs,
            pool_masks=raw_masks,
            similarity_dict=raw_similarity_dicts[fold],
            top_k=args.top_k,
            title=f"Raw Pool | fold={fold} | query={query_idx}",
        )
        draw_query_with_topk(
            ax=axes[row][1],
            query_idx=query_idx,
            query_trajs=dc_trajs_raw,
            query_masks=dc_masks_raw,
            pool_trajs=dc_pool_trajs,
            pool_masks=dc_pool_masks,
            similarity_dict=dc_similarity_dicts[fold],
            top_k=args.top_k,
            title=f"DC Pool | fold={fold} | query={query_idx}",
        )
        selected_cluster_indices = dc_similarity_dicts[fold][query_idx][: args.top_k]
        if args.overlay_cluster_members and len(dc_cluster_meta) > 0:
            for cluster_idx in selected_cluster_indices:
                meta = dc_cluster_meta.get(cluster_idx, {})
                members = meta.get("member_original_indices", [])
                for raw_member_idx in members:
                    px, py = get_primary_xy(raw_trajs[raw_member_idx], raw_masks[raw_member_idx])
                    if len(px) > 0:
                        axes[row][1].plot(px, py, linewidth=0.6, alpha=0.22, color="gray")
        if args.debug_clusters and len(dc_cluster_meta) > 0:
            debug_lines = build_dc_debug_lines(
                query_idx=query_idx,
                selected_cluster_indices=selected_cluster_indices,
                cluster_meta=dc_cluster_meta,
            )
            print(" | ".join(debug_lines))

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)))
    fig.suptitle(
        f"Query Matched Comparison: raw={args.raw_name} vs dc={args.dc_name} ({args.split})",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.dirname(args.output)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    img_name = f"{out_dir}/raw_vs_dc_seed_{args.seed}.png"
    fig.savefig(img_name, dpi=180)
    print(f"Saved figure to: {img_name}")
    print(f"Fold={fold}, query_indices={query_indices}, top_k={args.top_k}")


if __name__ == "__main__":
    main()
