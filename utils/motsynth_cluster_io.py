"""MOTSynth paths and track loading for dynamic clustering (x–z or x–y plane)."""

from __future__ import annotations

import configparser
import shutil
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from utils.data import make_mot_standard_gt_df

PositionPlane = Literal["xz", "xy", "bbox_center"]


def parse_seqinfo_seq_length(seqinfo_path: Path) -> Optional[int]:
    """Return ``seqLength`` from MOTSynth ``seqinfo.ini`` if present."""
    if not seqinfo_path.is_file():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in cfg or "seqLength" not in cfg["Sequence"]:
        return None
    return int(cfg["Sequence"]["seqLength"])


def motsynth_source_gt_path(data_dir: Path, scene: str) -> Path:
    """Path to raw ``gt.txt`` under ``data/motsynth/mot_annotations/<scene>/gt/``."""
    return data_dir / "motsynth" / "mot_annotations" / scene / "gt" / "gt.txt"


def motsynth_cluster_scene_dir(data_dir: Path, scene: str) -> Path:
    """Mirror of ``mot_annotations/<scene>/`` under ``data/motsynth_cluster``."""
    return data_dir / "motsynth_cluster" / "mot_annotations" / scene


def motsynth_cluster_gt_dir(data_dir: Path, scene: str) -> Path:
    """``.../motsynth_cluster/mot_annotations/<scene>/gt/``."""
    return motsynth_cluster_scene_dir(data_dir, scene) / "gt"


def load_scene_ids_from_file(list_path: Path) -> List[str]:
    """One scene id per line (e.g. ``motsynth_val.txt``)."""
    with open(list_path, mode="r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_motsynth_gt_tracks_df(
    gt_path: Path,
    plane: PositionPlane = "xz",
) -> pd.DataFrame:
    """Load per-row detections as ``frame``, ``id``, ``x``, ``y`` (second dim name fixed).

    For CrowdCluster, the third and fourth exported columns are always two spatial
    coordinates; we store **x** and **z** in the ``x`` / ``y`` columns when ``plane``
    is ``xz`` (horizontal plane). For ``xy``, file columns ``..., x, y, z`` use the
    trailing **x, y**. For ``bbox_center``, use MOT bbox centers (first six columns).

    Args:
        gt_path: Path to ``gt.txt``.
        plane: Which 2D coordinates to use.

    Returns:
        DataFrame with columns ``frame``, ``id``, ``x``, ``y`` (``y`` holds z if xz).
    """
    if plane == "bbox_center":
        df = make_mot_standard_gt_df(str(gt_path))
        df = df.groupby(["frame", "id"], as_index=False).mean(numeric_only=True)
        df = df.rename(columns={"bb_center_x": "x", "bb_center_y": "y"})
    else:
        raw = pd.read_csv(gt_path, header=None)
        n = raw.shape[1]
        if n < 12:
            raise ValueError(
                f"Expected MOTSynth-style gt with >=12 columns, got {n} in {gt_path}"
            )
        if plane == "xz":
            dim1 = raw.iloc[:, -3].astype(np.float64)
            dim2 = raw.iloc[:, -1].astype(np.float64)
        elif plane == "xy":
            dim1 = raw.iloc[:, -3].astype(np.float64)
            dim2 = raw.iloc[:, -2].astype(np.float64)
        else:
            raise ValueError(f"Unknown plane: {plane!r}")
        df = pd.DataFrame(
            {
                "frame": raw.iloc[:, 0].astype(np.int64),
                "id": raw.iloc[:, 1].astype(np.int64),
                "x": dim1,
                "y": dim2,
            }
        )
        df = df.groupby(["frame", "id"], as_index=False).mean(numeric_only=True)
    df = df.sort_values(["id", "frame"])
    df["frame"] = df["frame"].astype(np.int64)
    df["id"] = df["id"].astype(np.int64)
    return df[["frame", "id", "x", "y"]]


def write_crowdcluster_tracks_txt(df: pd.DataFrame, out_path: Path) -> None:
    """Write space-separated ``frame id x y`` (second spatial dim may be z)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["id", "frame"]).to_csv(
        out_path,
        sep=" ",
        header=False,
        index=False,
        float_format="%.6f",
    )


def copy_seqinfo_into_cluster_tree(data_dir: Path, scene: str) -> None:
    """Copy ``seqinfo.ini`` from raw MOTSynth tree into ``motsynth_cluster`` mirror."""
    src = data_dir / "motsynth" / "mot_annotations" / scene / "seqinfo.ini"
    dst = motsynth_cluster_scene_dir(data_dir, scene) / "seqinfo.ini"
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def suggest_finish_frame(
    seqinfo_path: Path,
    finish: Optional[int],
    cap: int = 950,
) -> int:
    """Default finish (exclusive) capped by ``seqLength`` from seqinfo."""
    seq_len = parse_seqinfo_seq_length(seqinfo_path)
    if finish is not None:
        return finish
    if seq_len is not None:
        return min(cap, seq_len)
    return cap
