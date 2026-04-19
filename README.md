# TrajICL with Dynamic-Centroid Example Pool

This repository extends the TrajICL pipeline with a centroid-based example-pool construction path for dense crowd trajectory modeling. The objective is to preserve the original TrajICL training and inference interface while enabling an alternative preprocessing mode where individual pedestrian trajectories are transformed into dynamically maintained cluster-centroid trajectories.

The centroid construction follows the dynamic clustering framework from *Efficient Dense Crowd Trajectory Prediction Via Dynamic Clustering*: nested direction/location agglomerative clustering, periodic LOF-based reassignment, temporary-pool reclustering, and delta-based centroid trajectory propagation under membership changes.

## Scope of This Extension

- Adds a centroid preprocessing entry point: `preprocess_centroids.py`
- Keeps the original raw workflow intact (`preprocess.py`, raw pool training/evaluation)
- Supports runtime switching between:
  - `dataset.example_pool_type=raw`
  - `dataset.example_pool_type=centroid`
- Stores checkpoints separately for clean comparison:
  - `outputs/TrajICL/raw/`
  - `outputs/TrajICL/centroid/`

## Dataset Layout

Expected dataset location for MOTSynth annotations:

`dataset/mot_annotations/<scene_id>/gt/gt.txt`

The code resolves dataset roots automatically from repository-local dataset structure.

## Environment Setup

```bash
cd ~/Projects/TrajICL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## End-to-End Workflow

### 1) Build Raw Processed Pool

```bash
python preprocess.py --name motsynth --stage all
```

### 2) Build Centroid Processed Pool

```bash
python preprocess_centroids.py --name motsynth --stage all
```

Paper-aligned defaults used by centroid preprocessing:
- direction threshold: `50` degrees
- location threshold: `120` pixels
- LOF contamination: `0.2`
- LOF neighbor ratio: `0.8`
- cluster re-evaluation interval: `10` frames
- temporary recluster minimum size: `10`
- centroid delta update interval: `10` frames

### 3) Train (Raw Pool)

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=raw
```

Primary checkpoint target:
`outputs/TrajICL/raw/best_val_checkpoint.pth.tar`

### 4) Train (Centroid Pool)

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
```

Primary checkpoint target:
`outputs/TrajICL/centroid/best_val_checkpoint.pth.tar`

### 5) Evaluate

Raw:
```bash
python eval.py \
  --model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --example_pool_type raw
```

Centroid:
```bash
python eval.py \
  --model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --example_pool_type centroid
```

## Benchmarking: Raw vs Centroid Models

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

## Output Structure

- `outputs/processed_data/motsynth/` -> raw processed pool
- `outputs/processed_data/motsynth_centroid/` -> centroid processed pool
- `clustered_dataset/motsynth_centroid/<split>/{scene_num}.csv` -> per-scene clustered centroid trajectories
- `outputs/TrajICL/raw/` -> raw-model checkpoints
- `outputs/TrajICL/centroid/` -> centroid-model checkpoints
- `outputs/comparison/` -> benchmark reports
- `outputs/logs/` -> captured run logs

## Notes for Reproducibility

- If centroid preprocessing logic changes, regenerate centroid processed data before training/evaluation.
- Raw and centroid checkpoints should be trained/evaluated with matched hyperparameters for fair comparison.
- If runs are terminated by OOM (`Killed`), reduce worker count and batch size.

## Upstream References

- TrajICL foundation: *[Towards Predicting Any Human Trajectory In Context](https://arxiv.org/abs/2506.00871)* (NeurIPS 2025)
- Dynamic clustering reference: *[Efficient Dense Crowd Trajectory Prediction Via Dynamic Clustering](https://arxiv.org/abs/2603.18166)*
