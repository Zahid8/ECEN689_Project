import os
from typing import Iterable, Sequence


def prepare_matplotlib(use_agg: bool = True):
    try:
        import matplotlib

        if use_agg:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc
    return plt


def _ensure_parent_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def plot_metric_lines(
    shots: Sequence[int],
    left_vals: Sequence[float],
    right_vals: Sequence[float],
    ylabel: str,
    title: str,
    left_label: str,
    right_label: str,
    out_paths: Sequence[str],
) -> None:
    plt = prepare_matplotlib(use_agg=True)
    _ensure_parent_dirs(out_paths)

    plt.figure(figsize=(8, 5))
    plt.plot(shots, left_vals, marker="o", linewidth=2, label=left_label)
    plt.plot(shots, right_vals, marker="o", linewidth=2, label=right_label)
    plt.xlabel("num_example (shot)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    for path in out_paths:
        plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_metric_improvement(
    shots: Sequence[int],
    improves: Sequence[float],
    ylabel: str,
    title: str,
    out_paths: Sequence[str],
) -> None:
    plt = prepare_matplotlib(use_agg=True)
    _ensure_parent_dirs(out_paths)

    plt.figure(figsize=(8, 5))
    plt.bar([str(s) for s in shots], improves)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("num_example (shot)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    for path in out_paths:
        plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
