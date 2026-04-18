#!/usr/bin/env python3
"""Export MOTSynth MOT gt.txt to CrowdCluster whitespace format (frame id x y).

Rows are sorted by pedestrian id then frame, as required by CrowdCluster
``preparethedata`` (see CrowdCluster-main/main.py).
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# TrajICL-main on path
_TRAJICL_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAJICL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAJICL_ROOT))

from utils.data import make_mot_standard_gt_df


def parse_seqinfo_seq_length(seqinfo_path: Path) -> Optional[int]:
    """Read seqLength from MOTSynth seqinfo.ini.

    Args:
        seqinfo_path: Path to seqinfo.ini.

    Returns:
        Sequence length in frames, or None if missing.
    """
    if not seqinfo_path.is_file():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in cfg or "seqLength" not in cfg["Sequence"]:
        return None
    return int(cfg["Sequence"]["seqLength"])


def mot_gt_to_crowdcluster_df(gt_path: Path) -> pd.DataFrame:
    """Load MOT gt and return frame, id, x, y sorted for CrowdCluster export.

    Args:
        gt_path: Path to gt.txt.

    Returns:
        DataFrame with int frame/id and float centers, one row per (frame, id).
    """
    df = make_mot_standard_gt_df(str(gt_path))
    df = df.groupby(["frame", "id"], as_index=False).mean(numeric_only=True)
    df = df.rename(
        columns={"bb_center_x": "x", "bb_center_y": "y"}
    ).sort_values(["id", "frame"])
    df["frame"] = df["frame"].astype(np.int64)
    df["id"] = df["id"].astype(np.int64)
    return df[["frame", "id", "x", "y"]]


def export_scene_to_txt(
    scene: str,
    data_dir: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Write one scene to CrowdCluster-compatible .txt.

    Args:
        scene: Scene folder name (e.g. ``000``).
        data_dir: Dataset root containing ``motsynth/mot_annotations``.
        out_path: Output file path.

    Returns:
        The exported DataFrame.
    """
    gt_path = data_dir / "motsynth" / "mot_annotations" / scene / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing GT: {gt_path}")
    df = mot_gt_to_crowdcluster_df(gt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_path,
        sep=" ",
        header=False,
        index=False,
        float_format="%.6f",
    )
    return df


def read_agent_histories_from_mot_gt(gt_path: Path) -> Dict[int, np.ndarray]:
    """Per-agent (frame, x, y) histories from MOT gt.txt.

    Args:
        gt_path: Path to gt.txt.

    Returns:
        Mapping pedestrian id -> array of shape (T, 3) with columns frame, x, y
        sorted by frame.
    """
    df = mot_gt_to_crowdcluster_df(gt_path)
    histories: Dict[int, np.ndarray] = {}
    for ped_id, g in df.groupby("id"):
        arr = g.sort_values("frame")[["frame", "x", "y"]].to_numpy(dtype=np.float64)
        histories[int(ped_id)] = arr
    return histories


def read_agent_history_from_mot_gt(gt_path: Path) -> Dict[int, np.ndarray]:
    """Alias for :func:`read_agent_histories_from_mot_gt` (plan naming)."""
    return read_agent_histories_from_mot_gt(gt_path)


def load_scene_ids_from_file(list_path: Path) -> List[str]:
    """Load non-empty stripped lines from a scene list file."""
    with open(list_path, mode="r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_motsynth_settings_py(
    out_py: Path,
    directory_txt: Path,
    start: int,
    finish: int,
    tdist: int,
    tdirect: int,
    filename: str,
) -> None:
    """Write CrowdCluster motsynth_settings.py override module."""
    out_py.parent.mkdir(parents=True, exist_ok=True)
    txt_abs = directory_txt.resolve()
    csv_abs = (out_py.parent / filename).resolve()
    content = f'''"""CrowdCluster path overrides for MOTSynth (generated)."""

directoryGT = r"{txt_abs.as_posix()}"
directory = directoryGT
start = {start}
finish = {finish}
tdist = {tdist}
tdirect = {tdirect}
filename = r"{csv_abs.as_posix()}"
'''
    out_py.write_text(content, encoding="utf-8")


def default_data_dir() -> Path:
    """``Code/data`` when script lives under ``Code/TrajICL-main/scripts``."""
    return _TRAJICL_ROOT.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Scene ids (e.g. 000 089). If empty, use --scene-list.",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=None,
        help="File with one scene id per line (e.g. motsynth_val.txt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Dataset root containing motsynth/ (default: sibling data/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for .txt files (default: DATA_DIR/motsynth/crowdcluster_input).",
    )
    parser.add_argument(
        "--write-settings",
        type=Path,
        default=None,
        help="Write CrowdCluster motsynth_settings.py to this path.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=75,
        help="Suggested start frame for CrowdCluster (used in --write-settings).",
    )
    parser.add_argument(
        "--finish",
        type=int,
        default=None,
        help="Suggested finish frame for CrowdCluster (exclusive in getdatabynframe).",
    )
    parser.add_argument(
        "--tdist",
        type=int,
        default=800,
        help="Spatial clustering threshold in pixels (MOTSynth scale).",
    )
    parser.add_argument(
        "--tdirect",
        type=int,
        default=50,
        help="Direction threshold in degrees (CrowdCluster hyperparameter).",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help="Centroid CSV basename for --write-settings (default: motsynth_SCENE_centroids.csv).",
    )
    parser.add_argument(
        "--settings-scene",
        type=str,
        default=None,
        help="Scene id for --write-settings (default: first scene in the export list).",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.expanduser().resolve()
    out_dir: Path = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else (data_dir / "motsynth" / "crowdcluster_input")
    )

    scenes: List[str] = list(args.scenes)
    if args.scene_list is not None:
        scenes.extend(load_scene_ids_from_file(args.scene_list.expanduser().resolve()))
    if not scenes:
        parser.error("Provide scene ids or --scene-list.")

    seen = set()
    unique_scenes = []
    for s in scenes:
        if s not in seen:
            seen.add(s)
            unique_scenes.append(s)

    settings_scene = args.settings_scene or unique_scenes[0]
    if args.settings_scene is not None and args.settings_scene not in seen:
        parser.error(f"--settings-scene {args.settings_scene!r} not in export list.")

    for scene in unique_scenes:
        out_path = out_dir / f"{scene}.txt"
        df = export_scene_to_txt(scene, data_dir, out_path)
        seqinfo = (
            data_dir / "motsynth" / "mot_annotations" / scene / "seqinfo.ini"
        )
        seq_len = parse_seqinfo_seq_length(seqinfo)
        finish = args.finish
        if finish is None:
            finish = min(950, seq_len) if seq_len is not None else 950
        print(f"Exported scene {scene}: {len(df)} rows -> {out_path}")
        if seq_len is not None:
            print(f"  seqinfo seqLength={seq_len}; suggested finish={finish}")

    if args.write_settings is not None:
        seqinfo_ss = (
            data_dir / "motsynth" / "mot_annotations" / settings_scene / "seqinfo.ini"
        )
        seq_len_ss = parse_seqinfo_seq_length(seqinfo_ss)
        finish_ss = args.finish
        if finish_ss is None:
            finish_ss = min(950, seq_len_ss) if seq_len_ss is not None else 950
        out_path_ss = out_dir / f"{settings_scene}.txt"
        csv_name = args.csv_name or f"motsynth_{settings_scene}_centroids.csv"
        write_motsynth_settings_py(
            args.write_settings.expanduser().resolve(),
            out_path_ss,
            start=args.start,
            finish=finish_ss,
            tdist=args.tdist,
            tdirect=args.tdirect,
            filename=csv_name,
        )
        print(f"Wrote CrowdCluster settings: {args.write_settings}")


if __name__ == "__main__":
    main()
