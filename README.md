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
- Shared plotting backend (`utils/plotting.py`) used by benchmark/report/visualization scripts
- Professional plot styling with SciencePlots (`science`, `no-latex`, `bright`, `grid`) via shared backend
- Colorblind-safe scientific color scheme (Okabe-Ito inspired) applied across plot functions
- Slide-ready visualization package generator (`viz.py`)
- Annotation-scene visualization generator (`viz_scene.py`) for direct scene ids (e.g. `000`, `001`)
- CrowdCluster-style raw-vs-cluster visualizer (`viz_agent.py`) + notebook (`viz_agent.ipynb`)

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

Benchmark/report plotting now routes through the same backend (`utils/plotting.py`), so chart style/formatting is consistent across:
- `compare_raw_vs_centroid.py`
- `compare_checkpoints.py`
- `viz.py`, `viz_scene.py`, and `viz_agent.py` (matplotlib setup path)

If `SciencePlots` is installed, the shared plotting backend automatically applies a professional style preset.  
Install once:
```bash
pip install SciencePlots
```
The plotting backend also applies a consistent scientific palette:
- line metrics: raw/baseline in blue, centroid/candidate in orange
- improvement bars: green for positive, red for negative
- dense scene trajectories: categorical palette tuned for readability (replacing HSV rainbow)

## Visualization (For Slides)
```bash
python viz.py \
  --raw_dir outputs/processed_data/motsynth \
  --centroid_dir outputs/processed_data/motsynth_centroid \
  --split train \
  --num_samples 10 \
  --pair_coordinate_mode both
```

Artifacts are saved under:
- `outputs/visualizations/raw_vs_centroid_<timestamp>/`

Figure interpretation:
- `00_before_vs_after_raw_vs_centroid.png` and `03_raw_vs_centroid_pairs.png` are now metadata-matched (`centroid_metadata[*].source_sample_index`) so raw and centroid panels come from the same source sample.
- By default (`--pair_coordinate_mode both`), paired plots are exported in two variants:
  - absolute coordinates:
    - `00_before_vs_after_raw_vs_centroid.png`
    - `03_raw_vs_centroid_pairs.png`
  - normalized coordinates (shape-focused):
    - `00_before_vs_after_raw_vs_centroid_normalized.png`
    - `03_raw_vs_centroid_pairs_normalized.png`
- Absolute mode is best for validating whether centroid tracks are spatially plausible in the original scene.
- Normalized mode is best for trajectory-shape comparison independent of global translation.
- `01_raw_samples_grid.png` and `02_centroid_samples_grid.png` are independent sample grids for each representation (not one-to-one pairs).
- Color semantics in all trajectory panels:
  - blue = primary trajectory
  - orange = context trajectories
  - solid = history
  - dashed = future

If centroid trajectories appear as staircase/dot-only artifacts, regenerate centroid processed data with the current `preprocess_centroids.py` (older centroid outputs may still contain stale held-position behavior):
```bash
python preprocess_centroids.py --name motsynth --stage all
```

## Annotation Scene Visualization (No Sample Indices)
To visualize raw vs centroid trajectories directly from annotation scene ids:
```bash
python viz_scene.py \
  --scenes 000,001 \
  --data_dir dataset \
  --n_agents 57 \
  --n_clusters 32
```

Optional filtering/subsampling:
```bash
python viz_scene.py \
  --scenes 000 \
  --data_dir dataset \
  --frame_step 1 \
  --min_track_len 1 \
  --n_agents 57 \
  --n_clusters 32
```

This reads from:
- `dataset/mot_annotations/<scene_id>/gt/gt.txt`

Outputs per scene are saved under:
- `outputs/visualizations/scene_annotations_<timestamp>/`
- `<scene>_raw_vs_centroid.png`
- `<scene>_raw_tracks.csv`
- `<scene>_centroid_tracks.csv`
- `<scene>_centroid_metadata.json`
- `manifest.json`

Notes:
- `--n_agents` controls how many raw agent trajectories are plotted/exported per scene.
- `--raw_total` is an alias for `--n_agents` (if set, it overrides `--n_agents`).
- `--raw_total_mode` controls how raw agents are picked when limited:
  - `most_active` (default): longest tracks first
  - `least_active`: shortest tracks first
- `--n_clusters` controls how many centroid trajectories are plotted/exported per scene (top-ranked by trajectory support score); use `0` to keep all.
- Colors are unique per raw agent id and unique per plotted centroid cluster in each scene panel.

## Raw vs Cluster Plotting (CrowdCluster Style)
This plotting flow mirrors your teammate’s side-by-side style:
- left panel: raw trajectories (one color per pedestrian id)
- right panel: cluster centroid trajectories (one color per cluster id)
- raw coordinates are bbox center (`bb_center_x`, `bb_center_y`) from `gt.txt`

CLI:
```bash
python viz_agent.py \
  --scene_id 000 \
  --data_dir dataset \
  --start 75 \
  --finish 950
```

Optional:
```bash
python viz_agent.py \
  --scene_id 000 \
  --data_dir dataset \
  --start 75 \
  --finish 950 \
  --show_plot \
  --no_save_image
```

Notebook:
- Open [viz_agent.ipynb](/home/zahid/Projects/TrajICL/viz_agent.ipynb) to run the same plotting pipeline inline.

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
