import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List

from utils.plotting import plot_metric_improvement, plot_metric_lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two checkpoints across pool types and save a full report."
    )
    parser.add_argument("--baseline_model_path", type=str, required=True)
    parser.add_argument("--candidate_model_path", type=str, required=True)
    parser.add_argument("--baseline_label", type=str, default="original_trained")
    parser.add_argument("--candidate_label", type=str, default="centroid_trained")
    parser.add_argument("--dataset_name", type=str, default="motsynth")
    parser.add_argument("--prompting_method", type=str, default="sim")
    parser.add_argument("--shots", type=str, default="0,2,4,8")
    parser.add_argument(
        "--pools",
        type=str,
        default="raw,centroid",
        help="Comma-separated pools to evaluate: raw, centroid",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--processed_root",
        type=str,
        default="outputs/processed_data",
        help="Processed dataset root containing raw/centroid pools.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Benchmark artifact root. Default: outputs/comparison/<timestamp>",
    )
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--disable_file_logging", action="store_true")
    return parser.parse_args()


def parse_shots(shots_text: str) -> List[int]:
    shots = [int(x.strip()) for x in shots_text.split(",") if x.strip()]
    return sorted(set(shots))


def parse_pools(pools_text: str) -> List[str]:
    pools = [x.strip() for x in pools_text.split(",") if x.strip()]
    valid = {"raw", "centroid"}
    unknown = [p for p in pools if p not in valid]
    if unknown:
        raise ValueError(f"Unsupported pool values: {unknown}. Use only raw,centroid")
    return sorted(set(pools))


def sanitize_label(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def ensure_pool_exists(processed_root: str, dataset_name: str, pool_type: str) -> str:
    name = dataset_name if pool_type == "raw" else f"{dataset_name}_centroid"
    path = os.path.join(processed_root, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing processed pool directory: {path}. "
            f"Run preprocessing for pool={pool_type} first."
        )
    return path


def run_eval_for_model_pool(
    model_path: str,
    dataset_name: str,
    prompting_method: str,
    pool_type: str,
    shots: List[int],
    device: str,
    processed_root: str,
) -> Dict[int, Dict[str, float]]:
    import torch
    from omegaconf import OmegaConf

    from helper import create_dataloader, evaluate
    from model import create_model

    checkpoint = torch.load(model_path, map_location=device)
    cfg = OmegaConf.create(checkpoint["cfg"])
    OmegaConf.set_struct(cfg, False)

    cfg.device = device
    cfg.dataset.name = dataset_name
    cfg.dataset.prompting = prompting_method
    cfg.dataset.example_pool_type = pool_type
    cfg.dataset.processed_root = processed_root

    model = create_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    metrics = {}
    split = "val"

    for shot in shots:
        print(f"[{pool_type}] evaluating shot={shot} using {os.path.basename(model_path)}")
        cfg.dataset.num_example = shot
        dataloader = create_dataloader(split=split, dataset_name=cfg.dataset.name, cfg=cfg)

        stats = {}
        stats = evaluate("test", cfg, 0, model, dataloader, stats)

        for key, val in list(stats.items()):
            if isinstance(val, (int, float)):
                stats[key] = val / cfg.training.resize

        metrics[int(shot)] = {
            "ade": float(stats["loss_ade/test"]),
            "fde": float(stats["loss_fde/test"]),
        }

    return metrics


def save_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    from utils.run_logging import finalize_run_logging, start_run_logging

    shots = parse_shots(args.shots)
    pools = parse_pools(args.pools)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("outputs", "comparison", f"checkpoint_vs_checkpoint_{ts}")
    plots_dir = os.path.join("outputs", "plots")
    graphs_dir = os.path.join("outputs", "graphs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    log_state = None
    if not args.disable_file_logging:
        log_state, log_path = start_run_logging(args.log_dir, script_name="compare_checkpoints")
        print(f"[run-log] Capturing stdout/stderr to {log_path}")

    try:
        for pool in pools:
            ensure_pool_exists(args.processed_root, args.dataset_name, pool)

        print("Running baseline checkpoint benchmark...")
        baseline_metrics = {
            pool: run_eval_for_model_pool(
                model_path=args.baseline_model_path,
                dataset_name=args.dataset_name,
                prompting_method=args.prompting_method,
                pool_type=pool,
                shots=shots,
                device=args.device,
                processed_root=args.processed_root,
            )
            for pool in pools
        }

        print("Running candidate checkpoint benchmark...")
        candidate_metrics = {
            pool: run_eval_for_model_pool(
                model_path=args.candidate_model_path,
                dataset_name=args.dataset_name,
                prompting_method=args.prompting_method,
                pool_type=pool,
                shots=shots,
                device=args.device,
                processed_root=args.processed_root,
            )
            for pool in pools
        }

        long_rows = []
        pairwise_rows = []

        pool_summary = {}
        all_ade_improves = []
        all_fde_improves = []

        for pool in pools:
            ade_improves = []
            fde_improves = []

            for shot in shots:
                ba = baseline_metrics[pool][shot]["ade"]
                bf = baseline_metrics[pool][shot]["fde"]
                ca = candidate_metrics[pool][shot]["ade"]
                cf = candidate_metrics[pool][shot]["fde"]

                ade_delta = ba - ca
                fde_delta = bf - cf
                ade_improve_pct = (ade_delta / ba * 100.0) if ba != 0 else 0.0
                fde_improve_pct = (fde_delta / bf * 100.0) if bf != 0 else 0.0

                ade_improves.append(ade_improve_pct)
                fde_improves.append(fde_improve_pct)

                long_rows.append(
                    {
                        "pool": pool,
                        "shot": shot,
                        "model": args.baseline_label,
                        "ade": ba,
                        "fde": bf,
                    }
                )
                long_rows.append(
                    {
                        "pool": pool,
                        "shot": shot,
                        "model": args.candidate_label,
                        "ade": ca,
                        "fde": cf,
                    }
                )

                pairwise_rows.append(
                    {
                        "pool": pool,
                        "shot": shot,
                        "baseline_ade": ba,
                        "candidate_ade": ca,
                        "ade_delta_baseline_minus_candidate": ade_delta,
                        "ade_improve_pct_candidate_vs_baseline": ade_improve_pct,
                        "baseline_fde": bf,
                        "candidate_fde": cf,
                        "fde_delta_baseline_minus_candidate": fde_delta,
                        "fde_improve_pct_candidate_vs_baseline": fde_improve_pct,
                    }
                )

            pool_summary[pool] = {
                "mean_ade_improve_pct_candidate_vs_baseline": float(sum(ade_improves) / len(ade_improves)),
                "mean_fde_improve_pct_candidate_vs_baseline": float(sum(fde_improves) / len(fde_improves)),
            }
            all_ade_improves.extend(ade_improves)
            all_fde_improves.extend(fde_improves)

        summary = {
            "baseline_model_path": args.baseline_model_path,
            "candidate_model_path": args.candidate_model_path,
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "dataset_name": args.dataset_name,
            "prompting_method": args.prompting_method,
            "shots": shots,
            "pools": pools,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "pool_summary": pool_summary,
            "overall_mean_ade_improve_pct_candidate_vs_baseline": float(sum(all_ade_improves) / len(all_ade_improves)),
            "overall_mean_fde_improve_pct_candidate_vs_baseline": float(sum(all_fde_improves) / len(all_fde_improves)),
        }

        json_path = os.path.join(output_dir, "checkpoint_comparison.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        long_csv = os.path.join(output_dir, "metrics_long.csv")
        pairwise_csv = os.path.join(output_dir, "metrics_pairwise.csv")

        save_csv(long_csv, long_rows, ["pool", "shot", "model", "ade", "fde"])
        save_csv(
            pairwise_csv,
            pairwise_rows,
            [
                "pool",
                "shot",
                "baseline_ade",
                "candidate_ade",
                "ade_delta_baseline_minus_candidate",
                "ade_improve_pct_candidate_vs_baseline",
                "baseline_fde",
                "candidate_fde",
                "fde_delta_baseline_minus_candidate",
                "fde_improve_pct_candidate_vs_baseline",
            ],
        )

        base_tag = sanitize_label(args.baseline_label)
        cand_tag = sanitize_label(args.candidate_label)

        for pool in pools:
            baseline_ade = [baseline_metrics[pool][s]["ade"] for s in shots]
            candidate_ade = [candidate_metrics[pool][s]["ade"] for s in shots]
            baseline_fde = [baseline_metrics[pool][s]["fde"] for s in shots]
            candidate_fde = [candidate_metrics[pool][s]["fde"] for s in shots]

            ade_improve = [
                (b - c) / b * 100.0 if b != 0 else 0.0
                for b, c in zip(baseline_ade, candidate_ade)
            ]
            fde_improve = [
                (b - c) / b * 100.0 if b != 0 else 0.0
                for b, c in zip(baseline_fde, candidate_fde)
            ]

            prefix = f"{pool}_{base_tag}_vs_{cand_tag}"
            ade_paths = [
                os.path.join(output_dir, f"ade_vs_shot_{prefix}.png"),
                os.path.join(plots_dir, f"ade_vs_shot_{prefix}.png"),
                os.path.join(graphs_dir, f"ade_vs_shot_{prefix}.png"),
            ]
            fde_paths = [
                os.path.join(output_dir, f"fde_vs_shot_{prefix}.png"),
                os.path.join(plots_dir, f"fde_vs_shot_{prefix}.png"),
                os.path.join(graphs_dir, f"fde_vs_shot_{prefix}.png"),
            ]
            ade_imp_paths = [
                os.path.join(output_dir, f"ade_improve_pct_vs_shot_{prefix}.png"),
                os.path.join(plots_dir, f"ade_improve_pct_vs_shot_{prefix}.png"),
                os.path.join(graphs_dir, f"ade_improve_pct_vs_shot_{prefix}.png"),
            ]
            fde_imp_paths = [
                os.path.join(output_dir, f"fde_improve_pct_vs_shot_{prefix}.png"),
                os.path.join(plots_dir, f"fde_improve_pct_vs_shot_{prefix}.png"),
                os.path.join(graphs_dir, f"fde_improve_pct_vs_shot_{prefix}.png"),
            ]

            title_suffix = f"({pool} pool)"
            plot_metric_lines(
                shots,
                baseline_ade,
                candidate_ade,
                "minADE",
                f"minADE vs Shot {title_suffix}",
                args.baseline_label,
                args.candidate_label,
                ade_paths,
            )
            plot_metric_lines(
                shots,
                baseline_fde,
                candidate_fde,
                "minFDE",
                f"minFDE vs Shot {title_suffix}",
                args.baseline_label,
                args.candidate_label,
                fde_paths,
            )
            plot_metric_improvement(
                shots,
                ade_improve,
                "ADE Improvement (%)",
                f"{args.candidate_label} vs {args.baseline_label} ADE {title_suffix}",
                ade_imp_paths,
            )
            plot_metric_improvement(
                shots,
                fde_improve,
                "FDE Improvement (%)",
                f"{args.candidate_label} vs {args.baseline_label} FDE {title_suffix}",
                fde_imp_paths,
            )

        print("\nCheckpoint comparison complete.")
        print(f"Summary JSON: {json_path}")
        print(f"Long CSV:     {long_csv}")
        print(f"Pairwise CSV: {pairwise_csv}")
        print(f"Artifacts saved in: {output_dir}, {plots_dir}, {graphs_dir}")
        print(
            "Overall mean ADE improvement (%), "
            f"candidate vs baseline: {summary['overall_mean_ade_improve_pct_candidate_vs_baseline']:.4f}"
        )
        print(
            "Overall mean FDE improvement (%), "
            f"candidate vs baseline: {summary['overall_mean_fde_improve_pct_candidate_vs_baseline']:.4f}"
        )

    finally:
        finalize_run_logging(log_state)


if __name__ == "__main__":
    main()
