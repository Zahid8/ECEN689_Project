# TrajICL End-to-End Guide (Raw + Dynamic-Centroid Example Pools)

This README is an end-to-end workflow for running TrajICL in this repository with your dataset placed in:

- `TrajICL/dataset`

It covers:

1. automatic dataset loading from `dataset/`
2. raw preprocessing pipeline
3. dynamic-clustering centroid preprocessing pipeline
4. training with raw or centroid pools
5. evaluation with raw or centroid pools

## 1. Environment Setup

From the repo root:

```bash
cd /home/zahid/Projects/TrajICL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset Layout (Automatic Loading)

The code now auto-resolves MOTSynth annotations from these layouts, in order:

1. `<data_dir>/motsynth/mot_annotations/...`
2. `<data_dir>/mot_annotations/...`
3. `dataset/mot_annotations/...` (repo-local fallback)

Since your data is inside `TrajICL/dataset`, you can run preprocessing without extra path arguments.

Required annotation files per scene:

- `dataset/mot_annotations/<scene_id>/gt/gt.txt`

Optional split files (if present, they are used):

- `dataset/motsynth_train.txt`
- `dataset/motsynth_val.txt`

If split files are missing, the loader auto-builds deterministic splits from available scenes:

- first 80% scenes -> train
- last 20% scenes -> val

## 3. Raw Preprocessing (Standard TrajICL Pool)

Run full raw preprocessing (window extraction, fold split, similarity matrix, similar-trajectory dicts):

```bash
python preprocess.py --name motsynth --stage all
```

Output directory:

- `processed_data/motsynth`

Key artifacts:

- `train_trajs.pt`, `val_trajs.pt`
- `train_masks.pt`, `val_masks.pt`
- `*_pool_indices_by_fold.pickle`
- `*_valid_indices_by_fold.pickle`
- `*_sim_matrix_dicts.pt`
- `*_similar_traj_dicts_hist.pickle`
- `*_similar_traj_dicts_seq.pickle`

## 4. Centroid Preprocessing (Dynamic Clustering, Paper-Grounded)

Run centroid preprocessing end-to-end:

```bash
python preprocess_centroids.py --name motsynth --stage all
```

This runs:

1. dynamic nested clustering per sequence (`direction -> location`)
2. LOF-based re-evaluation every 10 frames
3. outlier reassignment + temporary-pool re-clustering
4. centroid trajectory generation using delta updates (not per-frame full raw mean)
5. similarity/preselection artifacts in the same format as raw preprocessing

Paper-default parameters used by default:

- `--distance_thresh_px 120`
- `--direction_thresh_deg 50`
- `--lof_contamination 0.2`
- `--lof_neighbor_ratio 0.8`
- `--reeval_interval 10`
- `--temporary_recluster_min_size 10`

Output directory:

- `processed_data/motsynth_centroid`

Centroid metadata sidecars:

- `train_centroid_metadata.json`
- `val_centroid_metadata.json`
- `train_centroid_metadata_by_scene.json`
- `val_centroid_metadata_by_scene.json`

## 5. Train with Raw Pool

Raw pool is default (`dataset.example_pool_type=raw`):

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=raw
```

## 6. Train with Centroid Pool

Switch only the example pool type:

```bash
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
```

The dataloader automatically reads from:

- `processed_data/motsynth_centroid`

No model-core changes are required.

## 7. Evaluate with Raw or Centroid Pool

Raw pool:

```bash
python eval.py --dataset_name motsynth --example_pool_type raw
```

Centroid pool:

```bash
python eval.py --dataset_name motsynth --example_pool_type centroid
```

## 8. One-Pass End-to-End Command Sequence

Run this once from repo root:

```bash
python preprocess.py --name motsynth --stage all
python preprocess_centroids.py --name motsynth --stage all
python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid
python eval.py --dataset_name motsynth --example_pool_type centroid
```

## 9. Quick Troubleshooting

If preprocessing says it cannot find MOTSynth annotations:

1. check `dataset/mot_annotations/<scene_id>/gt/gt.txt` exists
2. rerun with explicit path:

```bash
python preprocess.py --name motsynth --stage all --data_dir dataset
python preprocess_centroids.py --name motsynth --stage all --data_dir dataset
```

If centroid training fails due to missing processed files, ensure this exists first:

- `processed_data/motsynth_centroid/train_trajs.pt`
- `processed_data/motsynth_centroid/train_similar_traj_dicts_hist.pickle`

## 10. Config Knobs You Can Use

Default config file:

- `configs/config.yaml`

Relevant fields:

- `dataset.name: motsynth`
- `dataset.example_pool_type: raw | centroid`
- `dataset.centroid_suffix: _centroid`
- `dataset.prompting: sim | random`
- `dataset.num_example: <int>`

This is sufficient to run TrajICL end-to-end directly from `TrajICL/dataset` with either raw or centroid example pools.
