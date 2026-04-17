import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run raw vs centroid benchmark and save metrics/plots."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="motsynth")
    parser.add_argument("--prompting_method", type=str, default="sim")
    parser.add_argument("--shots", type=str, default="0,2,4,8")
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


def ensure_pool_exists(processed_root: str, dataset_name: str, pool_type: str) -> str:
    name = dataset_name if pool_type == "raw" else f"{dataset_name}_centroid"
    path = os.path.join(processed_root, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing processed pool directory: {path}. "
            f"Run preprocessing for pool={pool_type} first."
        )
    return path


def run_eval_for_pool(
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
        print(f"[{pool_type}] evaluating shot={shot}")
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


def plot_lines(shots, raw_vals, cen_vals, ylabel, title, out_paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(shots, raw_vals, marker="o", linewidth=2, label="raw")
    plt.plot(shots, cen_vals, marker="o", linewidth=2, label="centroid")
    plt.xlabel("num_example (shot)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    for p in out_paths:
        plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()


def plot_improvement(shots, improves, ylabel, title, out_paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.bar([str(s) for s in shots], improves)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("num_example (shot)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    for p in out_paths:
        plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    args = parse_args()
    from utils.run_logging import finalize_run_logging, start_run_logging

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("outputs", "comparison", f"raw_vs_centroid_{ts}")
    plots_dir = os.path.join("outputs", "plots")
    graphs_dir = os.path.join("outputs", "graphs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    log_state = None
    if not args.disable_file_logging:
        log_state, log_path = start_run_logging(args.log_dir, script_name="compare_raw_vs_centroid")
        print(f"[run-log] Capturing stdout/stderr to {log_path}")

    try:
        shots = parse_shots(args.shots)
        ensure_pool_exists(args.processed_root, args.dataset_name, "raw")
        ensure_pool_exists(args.processed_root, args.dataset_name, "centroid")

        print("Running benchmark: raw pool")
        raw_metrics = run_eval_for_pool(
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            prompting_method=args.prompting_method,
            pool_type="raw",
            shots=shots,
            device=args.device,
            processed_root=args.processed_root,
        )

        print("Running benchmark: centroid pool")
        centroid_metrics = run_eval_for_pool(
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            prompting_method=args.prompting_method,
            pool_type="centroid",
            shots=shots,
            device=args.device,
            processed_root=args.processed_root,
        )

        long_rows = []
        summary_rows = []

        ade_improve_pct_values = []
        fde_improve_pct_values = []

        for shot in shots:
            ra = raw_metrics[shot]["ade"]
            rf = raw_metrics[shot]["fde"]
            ca = centroid_metrics[shot]["ade"]
            cf = centroid_metrics[shot]["fde"]

            ade_delta = ra - ca
            fde_delta = rf - cf
            ade_improve_pct = (ade_delta / ra * 100.0) if ra != 0 else 0.0
            fde_improve_pct = (fde_delta / rf * 100.0) if rf != 0 else 0.0

            ade_improve_pct_values.append(ade_improve_pct)
            fde_improve_pct_values.append(fde_improve_pct)

            long_rows.append({"shot": shot, "pool": "raw", "ade": ra, "fde": rf})
            long_rows.append({"shot": shot, "pool": "centroid", "ade": ca, "fde": cf})

            summary_rows.append(
                {
                    "shot": shot,
                    "raw_ade": ra,
                    "centroid_ade": ca,
                    "ade_delta_raw_minus_centroid": ade_delta,
                    "ade_improve_pct": ade_improve_pct,
                    "raw_fde": rf,
                    "centroid_fde": cf,
                    "fde_delta_raw_minus_centroid": fde_delta,
                    "fde_improve_pct": fde_improve_pct,
                }
            )

        summary = {
            "model_path": args.model_path,
            "dataset_name": args.dataset_name,
            "prompting_method": args.prompting_method,
            "shots": shots,
            "raw": raw_metrics,
            "centroid": centroid_metrics,
            "mean_ade_improve_pct": float(sum(ade_improve_pct_values) / len(ade_improve_pct_values)),
            "mean_fde_improve_pct": float(sum(fde_improve_pct_values) / len(fde_improve_pct_values)),
        }

        json_path = os.path.join(output_dir, "metrics_comparison.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        long_csv = os.path.join(output_dir, "metrics_long.csv")
        summary_csv = os.path.join(output_dir, "metrics_summary.csv")

        save_csv(long_csv, long_rows, ["shot", "pool", "ade", "fde"])
        save_csv(
            summary_csv,
            summary_rows,
            [
                "shot",
                "raw_ade",
                "centroid_ade",
                "ade_delta_raw_minus_centroid",
                "ade_improve_pct",
                "raw_fde",
                "centroid_fde",
                "fde_delta_raw_minus_centroid",
                "fde_improve_pct",
            ],
        )

        raw_ade = [raw_metrics[s]["ade"] for s in shots]
        cen_ade = [centroid_metrics[s]["ade"] for s in shots]
        raw_fde = [raw_metrics[s]["fde"] for s in shots]
        cen_fde = [centroid_metrics[s]["fde"] for s in shots]

        ade_improve = [(r - c) / r * 100.0 if r != 0 else 0.0 for r, c in zip(raw_ade, cen_ade)]
        fde_improve = [(r - c) / r * 100.0 if r != 0 else 0.0 for r, c in zip(raw_fde, cen_fde)]

        ade_paths = [
            os.path.join(output_dir, "ade_vs_shot_raw_vs_centroid.png"),
            os.path.join(plots_dir, "ade_vs_shot_raw_vs_centroid.png"),
            os.path.join(graphs_dir, "ade_vs_shot_raw_vs_centroid.png"),
        ]
        fde_paths = [
            os.path.join(output_dir, "fde_vs_shot_raw_vs_centroid.png"),
            os.path.join(plots_dir, "fde_vs_shot_raw_vs_centroid.png"),
            os.path.join(graphs_dir, "fde_vs_shot_raw_vs_centroid.png"),
        ]
        ade_imp_paths = [
            os.path.join(output_dir, "ade_improve_pct_vs_shot.png"),
            os.path.join(plots_dir, "ade_improve_pct_vs_shot.png"),
            os.path.join(graphs_dir, "ade_improve_pct_vs_shot.png"),
        ]
        fde_imp_paths = [
            os.path.join(output_dir, "fde_improve_pct_vs_shot.png"),
            os.path.join(plots_dir, "fde_improve_pct_vs_shot.png"),
            os.path.join(graphs_dir, "fde_improve_pct_vs_shot.png"),
        ]

        plot_lines(shots, raw_ade, cen_ade, "minADE", "minADE vs Shot: Raw vs Centroid", ade_paths)
        plot_lines(shots, raw_fde, cen_fde, "minFDE", "minFDE vs Shot: Raw vs Centroid", fde_paths)
        plot_improvement(shots, ade_improve, "ADE Improvement (%)", "Centroid Improvement vs Raw (ADE)", ade_imp_paths)
        plot_improvement(shots, fde_improve, "FDE Improvement (%)", "Centroid Improvement vs Raw (FDE)", fde_imp_paths)

        print("\nComparison complete.")
        print(f"Summary JSON: {json_path}")
        print(f"Summary CSV:  {summary_csv}")
        print(f"Plots saved in: {output_dir}, {plots_dir}, {graphs_dir}")
        print(f"Mean ADE improvement (%): {summary['mean_ade_improve_pct']:.4f}")
        print(f"Mean FDE improvement (%): {summary['mean_fde_improve_pct']:.4f}")

    finally:
        finalize_run_logging(log_state)


if __name__ == "__main__":
    main()
