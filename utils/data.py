import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from utils.trajnetplusplustools import Reader_jrdb_2dbox, Reader_jta_all_visual_cues


def load_data_jrdb_2dbox(split, r, data_dir="data"):
    joint_and_mask = []
    # change dataset path
    name = "jrdb_2dbox"
    train_scenes, _, _ = prepare_data(
        f"{data_dir}/jrdb_2dbox/", subset=split, sample=1.0, goals=False, dataset_name=name
    )  # train_scenes: list of trajectories
    filename_list = []
    frames_list = []
    pedestrians_list = []
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):  # trajctories
        scene_train, frames, pedestrians = Reader_jrdb_2dbox.paths_to_xy(
            paths
        )  # (21, N, 8) (21, 28, 8)
        filename_list.append(filename)
        frames_list.append(frames)
        pedestrians_list.append(pedestrians)
        scene_train = drop_ped_with_missing_frame(
            scene_train
        )  # 観測フレームにnanが入っていたら削除
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train)  # (21, n, 8)

        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )  # (21, n, 2, 4)
        scene_train_real_ped = np.transpose(
            scene_train_real, (1, 0, 2, 3)
        )  # (n, 21, 2, 4)

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])  # (n, 21, 2)
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3],  # (n, 21, 1, 3)
                np.asarray(scene_train_mask)[:, :, 0:1],  # (n, 21, 2)
            )
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list


def load_data_jta_all_visual_cues(split, r=6, data_dir="data"):
    joint_and_mask = []
    # change dataset path
    name = "jta_all_visual_cues"

    train_scenes, _, _ = prepare_data(
        f"{data_dir}/jta_all_visual_cues/",
        subset=split,
        sample=1.0,
        goals=False,
        dataset_name=name,
    )
    filename_list = []
    frames_list = []
    pedestrians_list = []
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_train, frames, pedestrians = Reader_jta_all_visual_cues.paths_to_xy(paths)
        filename_list.append(filename)
        frames_list.append(frames)
        pedestrians_list.append(pedestrians)
        scene_train = drop_ped_with_missing_frame(scene_train)
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train, r=r)
        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )
        scene_train_real_ped = np.transpose(scene_train_real, (1, 0, 2, 3))

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3],
                np.asarray(scene_train_mask)[:, :, 0:1],
            )
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list


def prepare_data(path, subset="/train/", sample=1.0, goals=True, dataset_name=""):
    all_scenes = []

    # List file names
    files = [
        f.split(".")[-2] for f in os.listdir(path + subset) if f.endswith(".ndjson")
    ]
    # Iterate over file names
    if dataset_name == "jta_all_visual_cues":
        for file in files:
            reader = Reader_jta_all_visual_cues(
                path + subset + "/" + file + ".ndjson", scene_type="paths"
            )
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
        return all_scenes, None, True
    elif dataset_name == "jrdb_2dbox":
        for file in files:
            reader = Reader_jrdb_2dbox(
                path + subset + "/" + file + ".ndjson", scene_type="paths"
            )
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
        return all_scenes, None, True
    else:
        print("not implement this dataset, error from utils/data.py")
        exit()


def drop_ped_with_missing_frame(xy):
    xy_n_t = np.transpose(xy, (1, 0, 2))
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(xy_n_t.shape[0] - 1):
        for t in range(9):
            if np.isnan(xy_n_t[n + 1, t, 0]):
                mask[n + 1] = False
                break
    return np.transpose(xy_n_t[mask], (1, 0, 2))


def drop_distant_far(xy, r=6):
    distance_2 = np.sum(np.square(xy[:, :, 0:2] - xy[:, 0:1, 0:2]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


# motsynth ###
def load_motsynth(split, r=50, resize=1, stride=21, data_dir="data"):

    with open(f"{data_dir}/motsynth/motsynth_{split}.txt", "r") as file:
        static_scenes = [line.strip() for line in file]

    trajectories, filename_list, frames_list, pedestrians_list = (
        prepare_data_motsynth(static_scenes, stride=stride, data_dir=data_dir)
    )

    joint_and_mask = []

    for scene_i, scene_train in enumerate(trajectories):  # trajctories
        scene_train = scene_train[:, :, 2:]  # (21, 14, 4) -> (21, 14, 2)
        scene_train_tmp = np.zeros(
            [scene_train.shape[0], scene_train.shape[1], 8]
        )  # (21, N, 8)
        scene_train_tmp[:, :, :2] = scene_train
        scene_train = scene_train_tmp
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train, r=r)  # (21, n, 8)
        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )  # (21, n, 2, 4)
        scene_train_real_ped = np.transpose(
            scene_train_real, (1, 0, 2, 3)
        )  # (n, 21, 2, 4)
        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3] * resize,
                np.asarray(scene_train_mask)[:, :, 0:1],
            ),
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list


def make_motsynth_df(file_path):

    # Define column names for the DataFrame
    column_names = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "conf",
        "x",
        "y",
        "z",
    ]

    # Read the gt.txt file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None, names=column_names, index_col=False)
    df["bb_center_x"] = df["bb_left"] + (df["bb_width"] / 2)
    df["bb_center_y"] = df["bb_top"] + (df["bb_height"] / 2)
    df = df[["frame", "id", "bb_center_x", "bb_center_y"]]

    return df


def make_mot_standard_gt_df(file_path: str) -> pd.DataFrame:
    """Load MOT-style gt.txt using bounding boxes (first six columns).

    Supports standard MOTChallenge GT (frame, id, bb_left, bb_top, bb_width, bb_height, ...).

    Args:
        file_path: Path to gt.txt.

    Returns:
        DataFrame with columns frame, id, bb_center_x, bb_center_y.
    """
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < 6:
        raise ValueError(
            f"Expected gt with at least 6 columns, got {df.shape[1]} in {file_path}"
        )
    df = df.iloc[:, :6].copy()
    df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"]
    df["bb_center_x"] = df["bb_left"] + (df["bb_width"] / 2)
    df["bb_center_y"] = df["bb_top"] + (df["bb_height"] / 2)
    return df[["frame", "id", "bb_center_x", "bb_center_y"]]


def prepare_data_mot_gt(
    static_scenes: List[str],
    gt_path_fn: Callable[[str], str],
    make_df_fn: Callable[[str], pd.DataFrame],
    step: int = 10,
    seq_len: int = 21,
    stride: int = 21,
):
    """Chunk MOT-format trajectories the same way as MOTSynth preprocessing.

    Args:
        static_scenes: Scene folder names (used as filename keys in the pipeline).
        gt_path_fn: Maps scene name to gt.txt path.
        make_df_fn: Reads gt.txt to a frame/id/bbox-center DataFrame.
        step: Frame subsample step along the dense frame index list.
        seq_len: Sliding window length (21 matches hist+fut pipeline).
        stride: Stride between window starts.

    Returns:
        trajectories, filename_list, frames_list, pedestrians_list (same as MOTSynth).
    """
    trajectories = []
    filename_list = []
    frames_list = []
    pedestrians_list = []

    for static_scene in static_scenes:
        file_path = gt_path_fn(static_scene)
        df = make_df_fn(file_path)
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1)]
        frames = frames[::step]
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
            frames_in_curr_seq = np.unique(curr_seq_data[:, 0])
            if len(frames_in_curr_seq) != seq_len:
                continue
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            trajectory = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                if len(curr_ped_seq) != seq_len:
                    continue
                trajectory.append(curr_ped_seq[None])
            if len(trajectory) > 0:
                trajectory = np.concatenate(trajectory, axis=0).transpose(1, 0, 2)
                for n in range(trajectory.shape[1]):
                    trajectory_data = np.concatenate(
                        [
                            trajectory[:, n:].copy(),
                            trajectory[:, :n].copy(),
                        ],
                        axis=1,
                    )
                    trajectories.append(trajectory_data)
                    filename_list.append(static_scene)
                    frames_list.append(curr_seq_frames)
                    pedestrians_list.append(trajectory_data[0, 0, 1])

    return trajectories, filename_list, frames_list, pedestrians_list


def prepare_data_motsynth(
    static_scenes,
    step=10,
    seq_len=21,
    stride=21,
    data_dir="data",
):
    """MOTSynth: scene ids listed in motsynth_{split}.txt under mot_annotations."""

    return prepare_data_mot_gt(
        static_scenes,
        lambda scene: f"{data_dir}/motsynth/mot_annotations/{scene}/gt/gt.txt",
        make_motsynth_df,
        step=step,
        seq_len=seq_len,
        stride=stride,
    )


def _ht21_scan_gt_scenes(root: str) -> List[str]:
    """Return sorted sequence folder names that contain gt/gt.txt."""
    if not os.path.isdir(root):
        return []
    return sorted(
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.isfile(os.path.join(root, d, "gt", "gt.txt"))
    )


def _ht21_resolve_root_and_scenes(split: str, data_dir: str) -> Tuple[str, List[str]]:
    """Pick HT21 root folder and scene ids (same tensor pipeline as MOTSynth).

    Optional scene lists (one name per line), same idea as ``motsynth_{split}.txt``:
    ``{data_dir}/HT21/ht21_{split}.txt`` — scenes are loaded from ``HT21/train``.

    Args:
        split: train, val, or test.
        data_dir: Dataset root containing ``HT21``.

    Returns:
        (root_dir, scene_names).

    Raises:
        FileNotFoundError: If no GT is available for the requested split.
    """
    ht21 = os.path.join(data_dir, "HT21")
    list_path = os.path.join(ht21, f"ht21_{split}.txt")
    if os.path.isfile(list_path):
        with open(list_path, mode="r", encoding="utf-8") as f:
            scenes = [line.strip() for line in f if line.strip()]
        root = os.path.join(ht21, "train")
        return root, scenes

    if split == "train":
        root = os.path.join(ht21, "train")
        scenes = _ht21_scan_gt_scenes(root)
        if not scenes:
            raise FileNotFoundError(
                f"No sequences with gt/gt.txt under {root}. "
                "Or add data/HT21/ht21_train.txt listing scene folder names."
            )
        return root, scenes

    if split == "val":
        test_root = os.path.join(ht21, "test")
        test_scenes = _ht21_scan_gt_scenes(test_root)
        if test_scenes:
            return test_root, test_scenes
        val_list = os.path.join(ht21, "ht21_val.txt")
        if os.path.isfile(val_list):
            with open(val_list, mode="r", encoding="utf-8") as f:
                scenes = [line.strip() for line in f if line.strip()]
            root = os.path.join(ht21, "train")
            return root, scenes
        raise FileNotFoundError(
            "HT21 val: benchmark test set has no public gt/gt.txt. "
            "Create data/HT21/ht21_val.txt with one train sequence name per line "
            "(e.g. HT21-03) so validation uses HT21/train/.../gt/gt.txt."
        )

    if split == "test":
        root = os.path.join(ht21, "test")
        scenes = _ht21_scan_gt_scenes(root)
        if not scenes:
            raise FileNotFoundError(
                f"No sequences with gt/gt.txt under {root}."
            )
        return root, scenes

    raise ValueError(f"Unsupported HT21 split: {split!r}")


def load_ht21(
    split: str,
    r: float = 50,
    resize: float = 1,
    stride: int = 21,
    data_dir: str = "data",
    step: int = 10,
):
    """Load HumanTrajectory21 (HT21) MOT data with the same tensors as MOTSynth.

    GT is read from bounding-box centers (MOT format), chunked like MOTSynth, then
    passed through the same ``drop_distant_far`` and mask path as ``load_motsynth``.

    Args:
        split: train, val, or test.
        r: Distance threshold for drop_distant_far (same default as MOTSynth).
        resize: Scale applied to positions (same as MOTSynth).
        stride: Window stride passed to chunking.
        data_dir: Root that contains the ``HT21`` directory.
        step: Frame subsample step (same default as MOTSynth).

    Returns:
        Same tuple as load_motsynth: joint_and_mask, filename_list, frames_list,
        pedestrians_list.
    """
    root, static_scenes = _ht21_resolve_root_and_scenes(split, data_dir)
    for scene in static_scenes:
        gt_path = os.path.join(root, scene, "gt", "gt.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Missing GT for scene {scene!r}: {gt_path}")

    trajectories, filename_list, frames_list, pedestrians_list = prepare_data_mot_gt(
        static_scenes,
        lambda scene, _root=root: os.path.join(_root, scene, "gt", "gt.txt"),
        make_mot_standard_gt_df,
        step=step,
        seq_len=21,
        stride=stride,
    )

    joint_and_mask = []

    for scene_train in trajectories:
        scene_train = scene_train[:, :, 2:]
        scene_train_tmp = np.zeros([scene_train.shape[0], scene_train.shape[1], 8])
        scene_train_tmp[:, :, :2] = scene_train
        scene_train = scene_train_tmp
        if r is not None:
            scene_train, _ = drop_distant_far(scene_train, r=r)
        scene_train_real = scene_train.reshape(
            scene_train.shape[0], scene_train.shape[1], -1, 4
        )
        scene_train_real_ped = np.transpose(scene_train_real, (1, 0, 2, 3))
        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append(
            (
                np.asarray(scene_train_real_ped)[:, :, 0:1, :3] * resize,
                np.asarray(scene_train_mask)[:, :, 0:1],
            ),
        )

    return joint_and_mask, filename_list, frames_list, pedestrians_list
