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

## Raw vs Centroid Benchmark (Metrics + Plots)

Use the comparison script to evaluate both pool types with the same checkpoint and save a full report.

Prerequisites:

1. raw processed pool exists: `outputs/processed_data/motsynth`
2. centroid processed pool exists: `outputs/processed_data/motsynth_centroid`
3. trained checkpoint exists (for example):
   1. `outputs/TrajICL/raw/best_val_checkpoint.pth.tar`
   2. `outputs/TrajICL/centroid/best_val_checkpoint.pth.tar`

Run:

```bash
python compare_raw_vs_centroid.py \
  --model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --prompting_method sim \
  --device cuda
```

Saved outputs:

1. `outputs/comparison/raw_vs_centroid_<timestamp>/metrics_comparison.json`
2. `outputs/comparison/raw_vs_centroid_<timestamp>/metrics_long.csv`
3. `outputs/comparison/raw_vs_centroid_<timestamp>/metrics_summary.csv`
4. `outputs/comparison/raw_vs_centroid_<timestamp>/ade_vs_shot_raw_vs_centroid.png`
5. `outputs/comparison/raw_vs_centroid_<timestamp>/fde_vs_shot_raw_vs_centroid.png`
6. `outputs/comparison/raw_vs_centroid_<timestamp>/ade_improve_pct_vs_shot.png`
7. `outputs/comparison/raw_vs_centroid_<timestamp>/fde_improve_pct_vs_shot.png`

Plots are also copied to:

1. `outputs/plots/`
2. `outputs/graphs/`

Terminal log:

1. `outputs/logs/compare_raw_vs_centroid_<timestamp>.log`

---

## Checkpoint vs Checkpoint Benchmark (Original-trained vs Centroid-trained)

Use this script when you want to compare **two different checkpoints** directly in one report.

Example: compare an original-trained checkpoint (baseline) against a centroid-trained checkpoint (candidate).

Run:

```bash
python compare_checkpoints.py \
  --baseline_model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar \
  --candidate_model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar \
  --baseline_label original_trained \
  --candidate_label centroid_trained \
  --dataset_name motsynth \
  --prompting_method sim \
  --pools raw,centroid \
  --shots 0,2,4,8 \
  --device cuda
```

Saved outputs:

1. `outputs/comparison/checkpoint_vs_checkpoint_<timestamp>/checkpoint_comparison.json`
2. `outputs/comparison/checkpoint_vs_checkpoint_<timestamp>/metrics_long.csv`
3. `outputs/comparison/checkpoint_vs_checkpoint_<timestamp>/metrics_pairwise.csv`
4. per-pool ADE/FDE line plots and improvement plots

Plots are also copied to:

1. `outputs/plots/`
2. `outputs/graphs/`

Terminal log:

1. `outputs/logs/compare_checkpoints_<timestamp>.log`

---

## Visualization Script for Slides (`viz.py`)

Use `viz.py` to generate a presentation-ready raw-vs-centroid visualization package.

It includes:

1. raw sample trajectory grids (at least 10 samples by default)
2. centroid sample trajectory grids (at least 10 samples by default)
3. raw vs centroid side-by-side sample panels
4. spatial occupancy heatmaps
5. agent-count distributions and boxplots
6. primary speed/displacement/heading distributions
7. mean primary speed curves over time
8. a combined multi-page PDF report
9. summary JSON stats

Run:

```bash
python viz.py \
  --raw_dir outputs/processed_data/motsynth \
  --centroid_dir outputs/processed_data/motsynth_centroid \
  --split train \
  --num_samples 10
```

Outputs:

1. `outputs/visualizations/raw_vs_centroid_<timestamp>/` (all PNGs + PDF + summary JSON)
2. `outputs/logs/viz_<timestamp>.log` (terminal capture)

If matplotlib is missing:

```bash
pip install matplotlib
```

---

## Output Structure

All artifacts are stored under `outputs/`:

1. `outputs/processed_data/` -> raw/centroid processed datasets
2. `outputs/TrajICL/raw/` and `outputs/TrajICL/centroid/` -> checkpoints
3. `outputs/logs/` -> terminal run logs
4. `outputs/wandb/` -> wandb local files
5. `outputs/plots/` -> plots
6. `outputs/graphs/` -> graphs
7. `outputs/comparison/` -> raw-vs-centroid benchmark reports
8. `outputs/visualizations/` -> raw-vs-centroid visual reports

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
  load_similarity_seq: false    # keep false unless seq-similarity is explicitly needed

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

## Training gets killed with no traceback

If training stops with just `Killed`, this is usually an OS OOM kill (memory pressure),
most commonly with centroid pool + multi-worker dataloading.

Use this safer command:

```bash
python train.py -m \
  dataset.name=motsynth \
  dataset.example_pool_type=centroid \
  dataset.load_similarity_seq=false \
  training.num_workers=0 \
  training.batch_size=8 \
  wandb=False
```

Why this helps:

1. `dataset.load_similarity_seq=false` avoids loading the very large optional seq-similarity pickle.
2. `training.num_workers=0` prevents worker memory amplification.
3. smaller `training.batch_size` reduces runtime memory use.

## Stage 3 (`traj_sim`) appears stuck for hours

If progress stops during:

1. `===== Stage 3: compute trajectory similarity dicts =====`
2. `Processing files: ...`

run Stage 3 only with fewer workers:

```bash
python preprocess.py --name motsynth --stage traj_sim --max_workers 4
python preprocess_centroids.py --name motsynth --stage traj_sim --max_workers 4
```

This repository also now initializes similarity workers with single-threaded BLAS/Torch
settings and uses a safer default worker cap to reduce process/thread oversubscription.

## Dependencies missing

If imports fail (for example `omegaconf`), reinstall environment:

```bash
pip install -r requirements.txt
```

---
# Quick Run Checklist

## 0. Optional: Activate your environment

```bash
source .venv/bin/activate
```

## 1. Ensure raw and centroid processed pools exist

```bash
python preprocess.py --name motsynth --stage all
python preprocess_centroids.py --name motsynth --stage all
```

## 2. Train the baseline model (original / raw pool)

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=raw
```

After training, note the checkpoint path:

```text
outputs/TrajICL/raw/best_val_checkpoint.pth.tar
```

## 3. Train the centroid-integrated model

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
```

After training, note the checkpoint path:

```text
outputs/TrajICL/centroid/best_val_checkpoint.pth.tar
```

## 4. Benchmark A: Compare raw vs centroid pools using the same checkpoint

```bash
python compare_raw_vs_centroid.py \
  --model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --prompting_method sim \
  --device cuda
```

## 5. Benchmark B: Compare baseline checkpoint vs centroid checkpoint

This generates a full comparison report.

```bash
python compare_checkpoints.py \
  --baseline_model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar \
  --candidate_model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar \
  --baseline_label original_trained \
  --prompting_method sim \
  --pools raw,centroid \
  --shots 0,2,4,8 \
  --device cuda
```

## Output locations

Results and logs are saved under:

- `outputs/comparison/`
- `outputs/plots/`
- `outputs/graphs/`
- `outputs/logs/`


## Acknowledgement

This codebase builds on the original TrajICL release and related prior implementations acknowledged by the authors.
