<div align="center">

# TrajICL: End-to-End Guide (Raw + Dynamic-Centroid Example Pools)

**Original paper:** *Towards Predicting Any Human Trajectory In Context (NeurIPS 2025)*  
**Paper:** https://arxiv.org/abs/2506.00871

</div>

This repository contains the TrajICL training/evaluation pipeline and an extended preprocessing path that builds centroid-based example pools using dynamic clustering.

The instructions below are written for the current local setup where data is stored in:

- `TrajICL/dataset`

---

## What TrajICL Does

TrajICL is an in-context trajectory prediction framework that selects supporting examples from a pool and conditions prediction on those examples at inference/training time.

Core capabilities from the original codebase:

1. Standard trajectory preprocessing
2. Similarity-based example retrieval
3. In-context training and evaluation

---

## What Is Added in This Repo (Beyond Original)

This branch adds production-ready extensions on top of the original pipeline:

1. **Dynamic-clustering centroid preprocessing** (`preprocess_centroids.py`)
   - Nested clustering: direction -> location
   - LOF-based reassignment every 10 frames
   - Temporary pool re-clustering for new cluster formation
   - Delta-based centroid trajectory updates

2. **Config-driven pool switching**
   - `dataset.example_pool_type: raw | centroid`
   - No model-core redesign required

3. **Automatic dataset resolution from `dataset/`**
   - MOTSynth annotations are discovered automatically
   - Split fallback (80/20 train/val) when split text files are missing

4. **Unified output routing under `outputs/`**
   - Processed data, checkpoints, logs, plots/graphs in one root

5. **Automatic full terminal logging**
   - `stdout` + `stderr` are captured to timestamped log files for all main entry scripts

---

## Environment Setup

From repository root:

```bash
cd /home/zahid/Projects/TrajICL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Requirements

Expected annotation layout (already present in your setup):

- `dataset/mot_annotations/<scene_id>/gt/gt.txt`

The loader resolves MOTSynth from these locations (in order):

1. `<data_dir>/motsynth/mot_annotations/...`
2. `<data_dir>/mot_annotations/...`
3. `dataset/mot_annotations/...` (repo-local fallback)

Optional split files (used if present):

- `dataset/motsynth_train.txt`
- `dataset/motsynth_val.txt`

If split files are missing, deterministic split fallback is applied:

1. first 80% scenes -> train
2. last 20% scenes -> val

---

## End-to-End Run (Recommended)

Run the full raw + centroid pipeline, then train/evaluate with centroid pool:

```bash
python preprocess.py --name motsynth --stage all
python preprocess_centroids.py --name motsynth --stage all
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
python eval.py --dataset_name motsynth --example_pool_type centroid
```

---

## Preprocessing Workflows

## 1) Raw example-pool preprocessing

```bash
python preprocess.py --name motsynth --stage all
```

Output:

- `outputs/processed_data/motsynth`

## 2) Centroid example-pool preprocessing

```bash
python preprocess_centroids.py --name motsynth --stage all
```

Output:

- `outputs/processed_data/motsynth_centroid`

Centroid metadata sidecars:

1. `train_centroid_metadata.json`
2. `val_centroid_metadata.json`
3. `train_centroid_metadata_by_scene.json`
4. `val_centroid_metadata_by_scene.json`

### Centroid defaults (paper-aligned)

```bash
--direction_thresh_deg 50
--distance_thresh_px 120
--lof_contamination 0.2
--lof_neighbor_ratio 0.8
--reeval_interval 10
--temporary_recluster_min_size 10
```

---

## Training

## Train with raw pool

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=raw
```

## Train with centroid pool

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
```

---

## Evaluation

## Evaluate with raw pool

```bash
python eval.py --dataset_name motsynth --example_pool_type raw
```

## Evaluate with centroid pool

```bash
python eval.py --dataset_name motsynth --example_pool_type centroid
```

---

## Output Structure

All artifacts are stored under `outputs/`:

1. `outputs/processed_data/` -> raw/centroid processed datasets
2. `outputs/TrajICL/<run_name>/` -> checkpoints
3. `outputs/logs/` -> terminal run logs
4. `outputs/wandb/` -> wandb local files
5. `outputs/plots/` -> plots
6. `outputs/graphs/` -> graphs

---

## Automatic Terminal Logging

The following scripts automatically capture full terminal output (`stdout` + `stderr`) to timestamped log files:

1. `preprocess.py` -> `outputs/logs/preprocess_<timestamp>.log`
2. `preprocess_centroids.py` -> `outputs/logs/preprocess_centroids_<timestamp>.log`
3. `train.py` -> `outputs/logs/train_<timestamp>.log`
4. `eval.py` -> `outputs/logs/eval_<timestamp>.log`

This includes:

1. `print(...)` messages
2. progress bars
3. warnings
4. exceptions and tracebacks

### Optional logging flags

For preprocess/eval scripts:

```bash
--log_dir <path>
--disable_file_logging
```

Examples:

```bash
python preprocess.py --name motsynth --stage all --log_dir outputs/logs/custom
python preprocess_centroids.py --name motsynth --stage all --disable_file_logging
python eval.py --dataset_name motsynth --example_pool_type centroid --log_dir outputs/logs/custom
```

---

## Key Config Fields

`configs/config.yaml`

```yaml
dataset:
  name: motsynth
  example_pool_type: raw        # raw | centroid
  centroid_suffix: _centroid
  processed_root: outputs/processed_data

output_dir: outputs
```

---

## Troubleshooting

## Dataset not found

If preprocessing cannot find MOTSynth annotations:

```bash
python preprocess.py --name motsynth --stage all --data_dir dataset
python preprocess_centroids.py --name motsynth --stage all --data_dir dataset
```

## Centroid training cannot load data

Verify these files exist:

1. `outputs/processed_data/motsynth_centroid/train_trajs.pt`
2. `outputs/processed_data/motsynth_centroid/train_similar_traj_dicts_hist.pickle`

## Dependencies missing

If imports fail (for example `omegaconf`), reinstall environment:

```bash
pip install -r requirements.txt
```

---

## Acknowledgement

This codebase builds on the original TrajICL release and related prior implementations acknowledged by the authors.
