<div align="center">

# TrajICL + Dynamic-Centroid Extension

Official TrajICL foundation: **Towards Predicting Any Human Trajectory In Context (NeurIPS 2025)**  
Paper: https://arxiv.org/abs/2506.00871

</div>

## Overview
This repository extends the original TrajICL pipeline with a production-ready centroid-based data path for dense crowd trajectory modeling. The core objective is to keep the TrajICL model and example-selection flow intact while enabling a second preprocessing mode that converts raw multi-pedestrian tracks into dynamically maintained centroid tracks using nested direction/location clustering, LOF-based reassignment, temporary-pool reclustering, and delta-updated centroid trajectories. In practice, this gives you two interchangeable example pools (`raw` and `centroid`) under the same training/evaluation interface, plus full benchmarking and visualization tooling for direct performance and data-behavior comparison.

## What This Fork Adds
- Dynamic-clustering centroid preprocessing (`preprocess_centroids.py`)
- Config switch for example pool type (`raw` or `centroid`)
- Automatic dataset discovery from `dataset/` layout
- Deterministic checkpoint directories:
  - `outputs/TrajICL/raw/`
  - `outputs/TrajICL/centroid/`
- Full terminal log capture for all major scripts (`outputs/logs/*.log`)
- Raw-vs-centroid benchmark script (`compare_raw_vs_centroid.py`)
- Checkpoint-vs-checkpoint benchmark script (`compare_checkpoints.py`)
- Slide-ready visualization package generator (`viz.py`)

## Quick Start

### 1) Environment
```bash
cd ~/Projects/TrajICL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data Assumption
Expected MOTSynth annotation structure:
- `dataset/mot_annotations/<scene_id>/gt/gt.txt`

### 3) Build Processed Pools
Raw pool:
```bash
python preprocess.py --name motsynth --stage all
```

Centroid pool:
```bash
python preprocess_centroids.py --name motsynth --stage all
```

### 4) Train
Raw checkpoint target:
- `outputs/TrajICL/raw/best_val_checkpoint.pth.tar`

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=raw
```

Centroid checkpoint target:
- `outputs/TrajICL/centroid/best_val_checkpoint.pth.tar`

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
```

### 5) Evaluate
Raw:
```bash
python eval.py --model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar --dataset_name motsynth --example_pool_type raw
```

Centroid:
```bash
python eval.py --model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar --dataset_name motsynth --example_pool_type centroid
```

## Benchmarking

### A) Same checkpoint, different pools
```bash
python compare_raw_vs_centroid.py \
  --model_path outputs/TrajICL/centroid/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --prompting_method sim \
  --device cuda
```

### B) Two checkpoints in one report (baseline vs candidate)
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

## Visualization (For Slides)
```bash
python viz.py \
  --raw_dir outputs/processed_data/motsynth \
  --centroid_dir outputs/processed_data/motsynth_centroid \
  --split train \
  --num_samples 10
```

Artifacts are saved under:
- `outputs/visualizations/raw_vs_centroid_<timestamp>/`

## Output Layout
- `outputs/processed_data/` -> raw/centroid processed datasets
- `outputs/TrajICL/raw/` -> raw-model checkpoints
- `outputs/TrajICL/centroid/` -> centroid-model checkpoints
- `outputs/comparison/` -> benchmark reports
- `outputs/visualizations/` -> visualization reports
- `outputs/logs/` -> terminal logs
- `outputs/plots/`, `outputs/graphs/` -> exported plot copies

## Common Stability Overrides
If training is killed due to memory pressure:
```bash
python train.py -m \
  dataset.name=motsynth \
  dataset.example_pool_type=centroid \
  dataset.load_similarity_seq=false \
  training.num_workers=0 \
  training.batch_size=8 \
  wandb=False
```

## Documentation
- `details.md` -> full end-to-end operational guide
- `info.md` -> technical implementation notes, function-level details, and interpretation notes

## Acknowledgement
This codebase builds on the original TrajICL release and its cited upstream dependencies.
