import os
import warnings
from colorsys import hsv_to_rgb
from typing import Iterable, List, Sequence, Tuple, Union

SCIENCE_STYLE_STACK = ("science", "no-latex", "bright", "grid")
SCIENTIFIC_COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "gray": "#6E7783",
    "black": "#111111",
}
QUALITATIVE_BASE = [
    SCIENTIFIC_COLORS["blue"],
    SCIENTIFIC_COLORS["orange"],
    SCIENTIFIC_COLORS["green"],
    SCIENTIFIC_COLORS["purple"],
    SCIENTIFIC_COLORS["sky"],
    "#8C564B",
    "#7F7F7F",
    "#E69F00",
    "#17BECF",
    "#BCBD22",
]


def _apply_professional_style(plt) -> None:
    # Prefer SciencePlots if available; gracefully fall back to sane defaults.
    try:
        import scienceplots  # noqa: F401

        plt.style.use(list(SCIENCE_STYLE_STACK))
        return
    except Exception:
        pass

    warnings.warn(
        "SciencePlots style was not applied. Falling back to Matplotlib defaults.",
        RuntimeWarning,
        stacklevel=2,
    )
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.prop_cycle": plt.cycler(color=QUALITATIVE_BASE),
        }
    )


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
    _apply_professional_style(plt)
    return plt


def _ensure_parent_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _hsv_color(i: int) -> Tuple[float, float, float]:
    # Golden-ratio hue stepping gives visually separated colors for large-N tracks.
    hue = (0.13 + (i * 0.61803398875)) % 1.0
    sat = 0.58 + 0.10 * ((i % 3) / 2.0)
    val = 0.78 + 0.08 * ((i % 2))
    r, g, b = hsv_to_rgb(hue, min(0.85, sat), min(0.92, val))
    return r, g, b


def get_distinct_colors(count: int) -> List[Union[str, Tuple[float, float, float]]]:
    if count <= 0:
        return []
    if count <= len(QUALITATIVE_BASE):
        return QUALITATIVE_BASE[:count]
    return [_hsv_color(i) for i in range(count)]


def plot_metric_lines(
    shots: Sequence[int],
    left_vals: Sequence[float],
    right_vals: Sequence[float],
    ylabel: str,
    title: str,
    left_label: str,
    right_label: str,
    out_paths: Sequence[str],
    left_color: str = SCIENTIFIC_COLORS["blue"],
    right_color: str = SCIENTIFIC_COLORS["orange"],
) -> None:
    plt = prepare_matplotlib(use_agg=True)
    _ensure_parent_dirs(out_paths)

    plt.figure(figsize=(8, 5))
    plt.plot(shots, left_vals, marker="o", linewidth=2.2, markersize=5, label=left_label, color=left_color)
    plt.plot(shots, right_vals, marker="o", linewidth=2.2, markersize=5, label=right_label, color=right_color)
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
    bar_colors = [
        SCIENTIFIC_COLORS["green"] if v >= 0 else "#C44E52"
        for v in improves
    ]
    plt.bar([str(s) for s in shots], improves, color=bar_colors, edgecolor=SCIENTIFIC_COLORS["black"], linewidth=0.3)
    plt.axhline(0.0, color=SCIENTIFIC_COLORS["black"], linewidth=1)
    plt.xlabel("num_example (shot)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    for path in out_paths:
        plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
