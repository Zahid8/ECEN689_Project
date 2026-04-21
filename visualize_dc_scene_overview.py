import argparse
import os
import random
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import torch

from visualize_dual_dc_comparison import get_primary_xy, load_pickle


def get_all_agent_xy(traj: torch.Tensor, mask: torch.Tensor) -> List[List[List[float]]]:
    """Extract valid (x, y) polylines for all agents in one sample."""
    all_polylines: List[List[List[float]]] = []
    num_agents = int(traj.shape[0])
    for agent_idx in range(num_agents):
        traj_xy = traj[agent_idx, :, 0, :2].detach().cpu()
        mask_1d = mask[agent_idx, :, 0].detach().cpu().bool()
        points = traj_xy[mask_1d]
        if points.numel() == 0:
            continue
        xs = points[:, 0].tolist()
        ys = points[:, 1].tolist()
        all_polylines.append([xs, ys])
    return all_polylines


def load_raw_scene_bundle(raw_name: str, split: str, fold: int) -> Dict[str, object]:
    """Load raw trajectories and fold-specific pool indices."""
    save_dir = os.path.join("processed_data", raw_name)
    filename_list: List[str] = load_pickle(os.path.join(save_dir, f"{split}_filename_list.pickle"))
    pool_indices_by_fold: List[List[int]] = load_pickle(
        os.path.join(save_dir, f"{split}_pool_indices_by_fold.pickle")
    )
    if fold < 0 or fold >= len(pool_indices_by_fold):
        raise ValueError(f"Invalid fold={fold} for raw dataset {raw_name}")
    return {
        "trajs": torch.load(os.path.join(save_dir, f"{split}_trajs.pt")),
        "masks": torch.load(os.path.join(save_dir, f"{split}_masks.pt")),
        "filename_list": filename_list,
        "pool_indices": pool_indices_by_fold[fold],
        "all_indices": list(range(len(filename_list))),
    }


def load_dc_scene_bundle(dc_name: str, split: str, fold: int) -> Dict[str, object]:
    """Load unique cluster pool trajectories for one fold."""
    save_dir = os.path.join("processed_data", dc_name)
    traj_path = os.path.join(save_dir, f"{split}_pool_dc_fold_{fold}_trajs.pt")
    mask_path = os.path.join(save_dir, f"{split}_pool_dc_fold_{fold}_masks.pt")
    filename2idxs_path = os.path.join(
        save_dir, f"{split}_pool_dc_fold_{fold}_filename2idxs_dict.pickle"
    )
    if not (os.path.isfile(traj_path) and os.path.isfile(mask_path) and os.path.isfile(filename2idxs_path)):
        raise FileNotFoundError(
            "Missing DC fold files. Expected unique pool files under "
            f"{save_dir} for split={split}, fold={fold}."
        )

    cluster_meta_by_fold: List[Dict[int, Dict[str, object]]] = load_pickle(
        os.path.join(save_dir, f"{split}_cluster_meta_by_fold.pickle")
    )
    if fold < 0 or fold >= len(cluster_meta_by_fold):
        raise ValueError(f"Invalid fold={fold} for dc dataset {dc_name}")

    return {
        "trajs": torch.load(traj_path),
        "masks": torch.load(mask_path),
        "filename2idxs_dict": load_pickle(filename2idxs_path),
        "cluster_meta": cluster_meta_by_fold[fold],
    }


def select_scenes(
    requested_scenes: Sequence[str],
    candidate_scenes: Sequence[str],
    n_scenes: int,
    seed: int,
) -> List[str]:
    """Return exactly n_scenes from candidates or requested list."""
    if len(requested_scenes) > 0:
        scenes = [scene.strip() for scene in requested_scenes if scene.strip() != ""]
        if len(scenes) != n_scenes:
            raise ValueError(f"Please provide exactly {n_scenes} scenes.")
        return scenes

    unique = sorted(set(candidate_scenes))
    if len(unique) < n_scenes:
        raise ValueError(f"Not enough scenes to sample: have {len(unique)}, need {n_scenes}.")
    rng = random.Random(seed)
    return rng.sample(unique, n_scenes)


def plot_scene_panel(
    ax,
    scene: str,
    raw_trajs: Sequence[torch.Tensor],
    raw_masks: Sequence[torch.Tensor],
    raw_indices: Sequence[int],
    dc_trajs: Sequence[torch.Tensor],
    dc_masks: Sequence[torch.Tensor],
    dc_indices: Sequence[int],
    cluster_meta: Dict[int, Dict[str, object]],
) -> None:
    """Plot one scene panel: raw (gray) + clustered (black)."""
    for raw_idx in raw_indices:
        all_polylines = get_all_agent_xy(raw_trajs[raw_idx], raw_masks[raw_idx])
        for px, py in all_polylines:
            if len(px) > 0:
                ax.plot(px, py, color="gray", alpha=0.22, linewidth=0.7)

    for dc_idx in dc_indices:
        px, py = get_primary_xy(dc_trajs[dc_idx], dc_masks[dc_idx])
        if len(px) > 0:
            meta = cluster_meta.get(dc_idx, {})
            weight = int(meta.get("cluster_weight", 1))
            line_width = 1.4 if weight <= 1 else 2.0
            ax.plot(px, py, color="black", alpha=0.9, linewidth=line_width)

    ax.set_title(
        f"scene={scene} | raw={len(raw_indices)} | clusters={len(dc_indices)}",
        fontsize=10,
    )
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2x2 scene overview: raw pool (gray) vs DC clusters (black)."
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--raw_name", type=str, default="motsynth")
    parser.add_argument("--dc_name", type=str, default="motsynth_dual_dc")
    parser.add_argument("--n_scenes", type=int, default=4)
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated scene names. If empty, random scenes are sampled.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw_scope",
        type=str,
        default="all",
        choices=["all", "pool"],
        help="Raw trajectories to draw in gray: all scene trajectories or fold pool only.",
    )
    parser.add_argument("--output", type=str, default="results/dc_scene_overview.png")
    args = parser.parse_args()

    if args.n_scenes != 4:
        raise ValueError("This script currently supports 2x2 layout, so n_scenes must be 4.")

    raw_bundle = load_raw_scene_bundle(args.raw_name, args.split, args.fold)
    dc_bundle = load_dc_scene_bundle(args.dc_name, args.split, args.fold)

    raw_trajs: List[torch.Tensor] = raw_bundle["trajs"]  # type: ignore[assignment]
    raw_masks: List[torch.Tensor] = raw_bundle["masks"]  # type: ignore[assignment]
    raw_filename_list: List[str] = raw_bundle["filename_list"]  # type: ignore[assignment]
    pool_indices: List[int] = raw_bundle["pool_indices"]  # type: ignore[assignment]
    all_indices: List[int] = raw_bundle["all_indices"]  # type: ignore[assignment]

    dc_trajs: List[torch.Tensor] = dc_bundle["trajs"]  # type: ignore[assignment]
    dc_masks: List[torch.Tensor] = dc_bundle["masks"]  # type: ignore[assignment]
    dc_filename2idxs: Dict[str, List[int]] = dc_bundle["filename2idxs_dict"]  # type: ignore[assignment]
    cluster_meta: Dict[int, Dict[str, object]] = dc_bundle["cluster_meta"]  # type: ignore[assignment]

    source_indices = all_indices if args.raw_scope == "all" else pool_indices
    pool_scenes = [raw_filename_list[idx] for idx in source_indices]
    requested_scenes = [item.strip() for item in args.scenes.split(",") if item.strip() != ""]
    scenes = select_scenes(requested_scenes, pool_scenes, args.n_scenes, args.seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for panel_idx, scene in enumerate(scenes):
        row = panel_idx // 2
        col = panel_idx % 2
        ax = axes[row][col]
        raw_indices = [idx for idx in source_indices if raw_filename_list[idx] == scene]
        dc_indices = dc_filename2idxs.get(scene, [])
        plot_scene_panel(
            ax=ax,
            scene=scene,
            raw_trajs=raw_trajs,
            raw_masks=raw_masks,
            raw_indices=raw_indices,
            dc_trajs=dc_trajs,
            dc_masks=dc_masks,
            dc_indices=dc_indices,
            cluster_meta=cluster_meta,
        )

    fig.suptitle(
        f"DC Scene Overview ({args.split}, fold={args.fold}) | raw={args.raw_name} vs dc={args.dc_name}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir = os.path.dirname(args.output)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved figure to: {args.output}")
    print(f"Scenes: {scenes}")


if __name__ == "__main__":
    main()
