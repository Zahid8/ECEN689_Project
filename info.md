# TrajICL Implementation Info (Centroid Pipeline + Integration)

This document is a full technical implementation record for the work added to this repository, including:

1. Dynamic-clustering centroid example-pool generation
2. Integration into TrajICL data loading/training/evaluation paths
3. Dataset auto-resolution from `dataset/`
4. Unified artifact routing to `outputs/`
5. Full terminal stdout/stderr run logging
6. Automated raw-vs-centroid benchmarking with metrics/CSV/plots
7. Automated checkpoint-vs-checkpoint benchmarking (baseline vs candidate)
8. Comprehensive raw-vs-centroid visualization package generation for slides (`viz.py`)

---

## 1) Primary Objective Implemented

The repository now supports two interchangeable example-pool modes:

1. `raw` (existing behavior)
2. `centroid` (new behavior) based on dynamic clustering and centroid trajectory generation

The centroid path is implemented in `preprocess_centroids.py` and integrated such that existing model code can consume centroid pools without model architecture changes.

---

## 2) Changed Files and Scope

### New files

1. `preprocess_centroids.py`
2. `utils/run_logging.py`
3. `compare_raw_vs_centroid.py`
4. `compare_checkpoints.py`
5. `viz.py`
6. `viz_scene.py`
7. `info.md` (this document)

### Modified files

1. `utils/data.py`
2. `load_data.py`
3. `dataset.py`
4. `preprocess.py`
5. `train.py`
6. `eval.py`
7. `utils/utils.py`
8. `configs/config.yaml`
9. `README.md`

---

## 3) Centroid Pipeline Design (`preprocess_centroids.py`)

## 3.1 Constants and runtime data structures

1. `EPS = 1e-6`
Purpose: numeric stability in divisions and near-zero checks.

2. `CENTROID_ID_OFFSET = 1_000_000`
Purpose: synthetic centroid track IDs are generated in a high numeric range to avoid collisions with raw pedestrian IDs.

3. `PedestrianState` dataclass
Fields:
1. `ped_id`
2. `x`, `y`
3. `vx`, `vy`
4. `theta`
5. `cluster_id`
Purpose: paper-style per-ped frame state container for clustering/membership operations.

4. `ClusterRuntime` dataclass
Fields:
1. cluster identity (`cluster_id`, `created_frame_idx`)
2. active members (`members`)
3. centroid trajectory cache (`centroid_by_frame`)
4. centroid direction cache (`direction_by_frame`)
5. membership/size histories (`member_history`, `size_history`)
6. lifecycle control (`last_nonempty_frame_idx`, `active`)
Purpose: full evolving cluster state across frames.

## 3.2 Direction and distance primitives

1. `smallest_angular_distance(theta_a, theta_b)`
Purpose: computes minimal angular gap in degrees using wraparound in `[-pi, pi]`.
Use: direction compatibility checks.

2. `_latest_value(history, frame_idx)` and `_latest_angle(history, frame_idx)`
Purpose: fetch last available historical value at-or-before current frame when exact frame key is missing.
Use: stable centroid and direction lookup.

3. `_pairwise_distances(features)`
Purpose: NxN Euclidean distances used by custom LOF.

## 3.3 LOF implementation details

1. `compute_lof_scores(features, n_neighbors)`
Purpose: custom Local Outlier Factor calculation implemented without external dependencies.
How:
1. builds pairwise distance matrix
2. computes k-distance and neighborhoods
3. computes local reachability density (LRD)
4. computes LOF score = neighbor LRD ratio to point LRD
Behavior:
1. returns ones for tiny sets (`n_samples <= 2`)
2. clamps neighbor count to valid range

2. `build_cluster_feature_matrix(cluster_members, states)`
Purpose: builds LOF feature vectors from both location and direction components.
Feature format per member:
1. `x`
2. `y`
3. `cos(theta)`
4. `sin(theta)`
Then z-score normalization per feature dimension.

3. `evaluate_cluster_members_with_lof(cluster_members, states, contamination, n_neighbors_ratio)`
Purpose: identify cluster outliers at reevaluation frames.
How:
1. neighbor count = `ceil(ratio * cluster_size)` then clamped
2. number of outliers = `ceil(contamination * cluster_size)` then clamped
3. top LOF scores above `1.0` marked as outliers

## 3.4 Nested clustering implementation

1. `threshold_agglomerative(ped_ids, distance_fn, threshold)`
Purpose: thresholded agglomerative grouping (single-link connectivity) via union-find.

2. `agglomerative_cluster_by_direction(...)`
Purpose: group by smallest angular distance threshold.

3. `agglomerative_cluster_by_location(...)`
Purpose: group by Euclidean location threshold.

4. `nested_initial_clustering(active_states, direction_thresh_deg, distance_thresh_px)`
Purpose: paper pipeline:
1. direction clustering first
2. location clustering inside each direction cluster
Output: final location clusters as initial runtime clusters.

## 3.5 Motion-state construction

1. `compute_motion_state(track_xy, track_mask, ped_id, frame_idx, assignments)`
Purpose: compute per-ped frame state from trajectory tensors.
How:
1. read current `(x,y)`
2. search backward for previous valid frame
3. velocity = delta position
4. heading `theta = atan2(vy, vx)` (or 0 if stationary/no history)

2. `get_active_pedestrians(...)`
Purpose: construct state map for all active peds at frame.

## 3.6 Cluster lifecycle and assignment

1. `create_cluster(...)`
Purpose: instantiate a cluster from member IDs at current frame.
How:
1. initialize centroid with member mean at creation frame
2. initialize direction to 0
3. write first member/size history
4. update global member assignments

2. `find_nearest_compatible_cluster(outlier_state, active_clusters, frame_idx, direction_thresh_deg, distance_thresh_px, skip_cluster_id=None)`
Purpose: nearest reassignment target lookup.
Compatibility constraints:
1. angular distance <= direction threshold
2. centroid distance <= location threshold
Selection criterion: minimum location distance among compatible clusters.

3. `update_cluster_assignments(...)`
Purpose: reevaluation-time maintenance logic.
Flow:
1. run LOF inside each active cluster
2. remove outliers from source clusters
3. try reassignment to nearest compatible cluster
4. if no compatible cluster -> temporary pool
5. process still-unassigned active peds similarly
6. remove stale temporary entries (inactive or reassigned)
7. if temporary pool size >= `temporary_recluster_min_size`, run nested clustering on temporary set and spawn new clusters

Important note implemented in code comment:
Paper prose contains both 5 and 10 as thresholds; default uses 10 from algorithm box.

## 3.7 Centroid trajectory update rule

1. `initialize_centroid(cluster_members, active_states)`
Purpose: creation-time centroid by average member position.

2. `update_centroid_with_delta(prev_centroid_xy, cluster_members_curr, active_states_curr, track_xy, track_mask, frame_idx)`
Purpose: centroid delta update from average member displacement:
1. collect each member displacement `(x_t - x_{t-1}, y_t - y_{t-1})`
2. average displacements
3. centroid_t = centroid_{t-1} + avg_delta
4. if no valid member displacement, keep previous centroid

3. Centroid direction update in runtime loop:
`direction_vec = centroid_{t-1} - centroid_t`, then `atan2`.
If near-zero movement, previous direction is reused.

## 3.8 Scene/window processing loop

1. `run_dynamic_clustering_scene(...)`
Purpose: frame-by-frame dynamic clustering evolution for one trajectory window.

Detailed behavior:
1. builds active states each frame
2. initial clustering starts when non-empty active set exists after frame 0
3. newly appearing unassigned peds are assigned to nearest compatible cluster or temporary pool
4. every `reeval_interval` frames, LOF-based maintenance runs
5. centroid updates occur every `centroid_update_interval` frames using delta rule; in-between frames copy previous centroid
6. clusters become archived when empty for longer than `cluster_empty_tolerance`

2. `build_centroid_tracks_from_clusters(clusters, frames)`
Purpose: convert cluster runtime history into trajectory arrays + metadata.

Outputs per centroid:
1. trajectory `[seq_len, 2]`
2. mask `[seq_len]`
3. metadata fields:
1. `cluster_size` (mean nonzero size)
2. `start_frame`
3. `end_frame`
4. `member_ids`
5. `cluster_size_history` keyed by frame ID

3. `convert_scene_to_centroid_samples(...)`
Purpose: convert one window into TrajICL-compatible sample tuples.

Important integration detail:
For each centroid as primary target, this creates one sample where that centroid is index 0 and all other centroids follow, matching existing TrajICL sample assumptions.

Also assigns global synthetic centroid IDs:
`global_id = CENTROID_ID_OFFSET + local_index_offset`.

4. `write_centroid_scene(scene_output_path, centroid_tracks, metadata)`
Purpose: summary JSON writer.

## 3.9 Split-level pipeline and staging

1. `process_split(...)`
Purpose: complete split pipeline for centroid mode.
Steps:
1. load raw split windows via existing `preprocess.load_data`
2. run dynamic clustering window-by-window
3. build centroid samples in TrajICL format
4. run pool/valid splitting on centroid IDs
5. save all standard artifacts via existing `save_data`
6. save centroid metadata sidecars
7. compute and print sanity stats:
1. raw windows vs centroid windows
2. average cluster size
3. single-member cluster count
4. average centroid track length

2. `main()` in centroid script supports stages:
1. `preprocess`
2. `sim_matrix`
3. `traj_sim`
4. `all`

Stage behavior mirrors `preprocess.py`, so centroid datasets produce the same compatibility artifacts (`*_sim_matrix`, `*_similar_traj_dicts_*`).

## 3.10 CLI parameters and defaults

Centroid script default values:

1. `--distance_thresh_px 120`
2. `--direction_thresh_deg 50`
3. `--lof_contamination 0.2`
4. `--lof_neighbor_ratio 0.8`
5. `--reeval_interval 10`
6. `--temporary_recluster_min_size 10`
7. `--centroid_update_interval 10`
8. `--cluster_empty_tolerance 3`
9. `--save_root outputs/processed_data`
10. `--log_dir outputs/logs`

---

## 4) Data Format Compatibility Strategy

Centroid outputs are intentionally saved in the same core format expected by TrajICL:

1. `*_trajs.pt`
2. `*_masks.pt`
3. fold split pickles
4. filename/index mapping pickles
5. similarity matrices and similar-trajectory dictionaries

This preserves downstream loader/model behavior.

Additional centroid-only sidecars:

1. `<split>_centroid_metadata.json`
2. `<split>_centroid_metadata_by_scene.json`
3. `<split>_centroid_scene_summary.json`

---

## 5) Integration Changes Outside Centroid Script

## 5.1 Dataset auto-detection and split fallback (`utils/data.py`)

### Added `resolve_motsynth_root(data_dir)`
Purpose: automatic MOTSynth root discovery from these layouts:

1. `<data_dir>/motsynth/mot_annotations`
2. `<data_dir>/mot_annotations`
3. `dataset/mot_annotations` fallback

### Added `get_motsynth_scene_split(motsynth_root, split)`
Purpose:
1. use `motsynth_<split>.txt` if available
2. otherwise deterministic fallback split from directory listing with `gt/gt.txt`
1. first 80% -> train
2. last 20% -> val/test

### Updated `load_motsynth(...)`
Now resolves root automatically and uses split resolver.

### Updated `prepare_data_motsynth(...)`
Now receives `motsynth_root` and reads from `motsynth_root/mot_annotations/<scene>/gt/gt.txt`.

## 5.2 Processed-data loader path control (`load_data.py`)

Updated `load_processed_data(...)` signature:

1. `example_pool_type` (`raw` or `centroid`)
2. `centroid_suffix` (default `_centroid`)
3. `processed_root` (default `outputs/processed_data`)
4. `load_similarity_seq` (default `False`)

Behavior:
1. when centroid mode is selected, dataset name maps to `<name><suffix>` unless suffix already present
2. processed artifacts loaded from `processed_root/<resolved_name>`
3. sequence-similarity dictionary loading is optional and disabled by default to reduce training RAM footprint

## 5.3 Dataset object wiring (`dataset.py`)

`Dataset.__init__` now accepts:

1. `example_pool_type`
2. `centroid_suffix`
3. `processed_root`
4. `load_similarity_seq`

`create_dataset(...)` now pulls these from config and passes to loader.

Result: model side is unchanged; pool source becomes a config switch.

## 5.4 Preprocess defaults updated (`preprocess.py`)

Defaults changed to route artifacts under outputs:

1. `save_root` default from `processed_data` -> `outputs/processed_data`
2. same for helper `save_data` and `load_processed_data` defaults
3. Stage-3 similarity worker argument passing was optimized so each worker receives
   only per-file matrices (`dist_matrix`, `vel_matrix`) instead of the entire
   similarity dictionary, eliminating multi-GB per-task IPC overhead.
4. Stage-3 worker runtime stabilization was added:
   1. ProcessPool worker initializer sets single-thread settings (`OMP/MKL/OPENBLAS/NUMEXPR`)
   2. Torch worker threads are set to 1 (`set_num_threads`, `set_num_interop_threads`)
   3. default parallel worker cap for similarity stage is now conservative (`min(8, cpu_count, num_files)`)

## 5.5 Train/eval/output configuration

1. `configs/config.yaml`
1. `output_dir: outputs`
2. `dataset.processed_root: outputs/processed_data`
3. keeps `dataset.example_pool_type` and `dataset.centroid_suffix`
4. sets `dataset.load_similarity_seq: False` by default for memory stability

2. `eval.py`
1. default model path changed to `outputs/TrajICL/raw/best_val_checkpoint.pth.tar`
2. supports `--example_pool_type`

3. `utils/utils.py` (`setup_wandb_logging`)
1. ensures output subdirs exist (`logs`, `plots`, `graphs`, `wandb`)
2. sets wandb local dir to `outputs/wandb`
3. routes checkpoints by pool type to deterministic directories:
   1. raw -> `outputs/TrajICL/raw/`
   2. centroid -> `outputs/TrajICL/centroid/`
4. `save_checkpoint` now writes checkpoints regardless of `wandb` mode, so disabling wandb does not disable checkpoint saving

---

## 6) Full Terminal Logging Implementation

A standalone utility was added: `utils/run_logging.py`.

## 6.1 `TeeStream`
Purpose: duplicate writes to terminal stream and log file.

## 6.2 `start_run_logging(log_dir, script_name)`
Purpose: create timestamped log file and replace `sys.stdout` / `sys.stderr` with tee streams.
Return:
1. state object
2. absolute/relative log path

## 6.3 `stop_run_logging(state)`
Purpose: restore original streams and close file.

## 6.4 `finalize_run_logging(state)`
Purpose: safe cleanup at function end.
Special behavior:
If exception is propagating (`sys.exc_info()[0] is not None`), do not restore streams immediately so interpreter-emitted traceback is still captured in file.

This ensures "every single thing" from terminal, including uncaught tracebacks, is logged.

## 6.5 Script integrations

### `preprocess.py`
Added CLI:
1. `--log_dir` (default `outputs/logs`)
2. `--disable_file_logging`

Lifecycle:
1. starts logging early in `main`
2. finalizes logging in `finally`

### `preprocess_centroids.py`
Same as above (`--log_dir`, `--disable_file_logging`, `try/finally` around pipeline).

### `train.py`
Auto-starts logging to `os.path.join(cfg.output_dir, "logs")` at run start and finalizes at end.

### `eval.py`
Added CLI:
1. `--log_dir`
2. `--disable_file_logging`

Auto starts and finalizes logging in `main`.

---

## 7) End-to-End Runtime Flow

## 7.1 Raw pool flow

1. `python preprocess.py --name motsynth --stage all`
2. artifacts -> `outputs/processed_data/motsynth`
3. `python train.py -m dataset.name=motsynth dataset.example_pool_type=raw`
4. `python eval.py --dataset_name motsynth --example_pool_type raw`

## 7.2 Centroid pool flow

1. `python preprocess_centroids.py --name motsynth --stage all`
2. artifacts -> `outputs/processed_data/motsynth_centroid`
3. `python train.py -m dataset.name=motsynth dataset.example_pool_type=centroid`
4. `python eval.py --dataset_name motsynth --example_pool_type centroid`

## 7.3 Logs and run artifacts

1. terminal logs -> `outputs/logs/*.log`
2. checkpoints -> `outputs/TrajICL/raw/` and `outputs/TrajICL/centroid/`
3. wandb local files -> `outputs/wandb/`
4. processed datasets -> `outputs/processed_data/`
5. benchmark reports -> `outputs/comparison/`

---

## 8) Edge Cases and Handling

1. Missing previous frame for direction
Handled by backward search; if none found, `(vx, vy, theta) = (0, 0, 0)`.

2. Stationary pedestrians
`theta` set to 0 when near-zero velocity.

3. Sparse membership for LOF
Clusters with <3 members skip LOF outlier removal.

4. Outlier without compatible cluster
Moved to temporary pool.

5. Temporary pool growth
When size reaches threshold, nested clustering on temporary set spawns new clusters.

6. Cluster empty periods
Cluster archived/deactivated after `cluster_empty_tolerance` empty frames.

7. Missing per-member displacement during centroid update
If no valid per-member delta at update frame, centroid remains unchanged.

8. No centroid trajectories generated for split
`process_split` raises runtime error with explicit message.

9. Exception logging
Tracebacks are captured due to `finalize_run_logging` behavior.

---

## 9) Key Implementation Choices and Rationale

1. No new dependencies for LOF
Custom LOF math was implemented with NumPy to avoid introducing `scikit-learn` and keep environment minimal.

2. Reuse existing TrajICL preprocess contracts
Centroid pipeline reuses `save_data`, `pool_valid_split`, similarity stages from existing preprocessing code, minimizing invasive changes.

3. Preserve selector/model behavior
By keeping trajectory tensor format identical, selector logic is reused unchanged.

4. Synthetic centroid IDs
Using offset IDs avoids conflicts and supports metadata mapping clarity.

5. Output and logging centralization under `outputs/`
Single root for all artifacts simplifies reproducibility and cleanup.

---

## 10) Function-Level Map (Added/Changed Core Functions)

### `preprocess_centroids.py`

1. `smallest_angular_distance`: wrapped angular delta in degrees
2. `_latest_value`: historical numeric retrieval
3. `_latest_angle`: historical angle retrieval
4. `_pairwise_distances`: LOF distance matrix
5. `compute_lof_scores`: LOF core
6. `threshold_agglomerative`: union-find threshold clustering
7. `agglomerative_cluster_by_direction`: direction threshold clustering
8. `agglomerative_cluster_by_location`: spatial threshold clustering
9. `nested_initial_clustering`: direction->location nesting
10. `build_cluster_feature_matrix`: LOF feature construction
11. `evaluate_cluster_members_with_lof`: outlier selection
12. `find_nearest_compatible_cluster`: reassignment target search
13. `compute_motion_state`: per-ped state extraction
14. `get_active_pedestrians`: active state map
15. `create_cluster`: runtime cluster creation
16. `update_cluster_assignments`: dynamic maintenance pass
17. `initialize_centroid`: first centroid mean
18. `update_centroid_with_delta`: delta propagation
19. `run_dynamic_clustering_scene`: full frame loop
20. `build_centroid_tracks_from_clusters`: runtime->track conversion
21. `convert_scene_to_centroid_samples`: TrajICL sample construction
22. `write_centroid_scene`: summary JSON writer
23. `process_split`: full split pipeline
24. `main`: staged CLI driver

### `utils/run_logging.py`

1. `TeeStream`
2. `start_run_logging`
3. `stop_run_logging`
4. `finalize_run_logging`

### `utils/data.py`

1. `resolve_motsynth_root`
2. `get_motsynth_scene_split`
3. `load_motsynth` updated to use above
4. `prepare_data_motsynth` updated root argument

### `load_data.py`

1. `load_processed_data` now supports `example_pool_type`, `centroid_suffix`, `processed_root`, `load_similarity_seq`

### `dataset.py`

1. `Dataset.__init__` now accepts/propagates `example_pool_type`, `centroid_suffix`, `processed_root`, `load_similarity_seq`
2. `create_dataset` reads same config keys

### `preprocess.py`

1. defaults route to `outputs/processed_data`
2. run-log CLI options (`--log_dir`, `--disable_file_logging`)
3. automatic run logging via `start_run_logging` + `finalize_run_logging`
4. Stage-3 parallel similarity computation now passes only per-file matrices to workers
5. Stage-3 ProcessPool now uses `_init_similarity_worker` to limit thread oversubscription

### `train.py`

1. automatic run logging at startup and safe finalization

### `eval.py`

1. model path default updated to `outputs/...`
2. run-log CLI options
3. automatic run logging + finalization

### `utils/utils.py`

1. `setup_wandb_logging` now creates `outputs` subdirs and routes wandb local files into `outputs/wandb`
2. checkpoint output directory is deterministic by pool type (`raw` / `centroid`)
3. checkpoint persistence is no longer gated on `cfg.wandb`

### `compare_raw_vs_centroid.py`

1. runs evaluation for `raw` and `centroid` pools with a shared checkpoint
2. supports configurable shots (`--shots`, default `0,2,4,8`)
3. emits machine-readable metrics:
   1. `metrics_comparison.json`
   2. `metrics_long.csv`
   3. `metrics_summary.csv`
4. emits plots:
   1. ADE vs shot (`raw` vs `centroid`)
   2. FDE vs shot (`raw` vs `centroid`)
   3. ADE improvement % vs shot
   4. FDE improvement % vs shot
5. writes benchmark outputs under `outputs/comparison/raw_vs_centroid_<timestamp>/`
6. copies plot artifacts into `outputs/plots/` and `outputs/graphs/`
7. supports automatic run logging via `utils/run_logging.py`

### `compare_checkpoints.py`

1. compares two checkpoints directly:
   1. baseline checkpoint (`--baseline_model_path`)
   2. candidate checkpoint (`--candidate_model_path`)
2. evaluates across configurable pools (`--pools raw,centroid`) and shots (`--shots`)
3. produces:
   1. `checkpoint_comparison.json`
   2. `metrics_long.csv`
   3. `metrics_pairwise.csv`
4. computes candidate-vs-baseline deltas/improvement percentages for ADE/FDE
5. generates per-pool plots:
   1. ADE vs shot
   2. FDE vs shot
   3. ADE improvement %
   4. FDE improvement %
6. writes artifacts under `outputs/comparison/checkpoint_vs_checkpoint_<timestamp>/`
7. copies plots to `outputs/plots/` and `outputs/graphs/`
8. supports automatic terminal logging to `outputs/logs/compare_checkpoints_<timestamp>.log`

### `viz.py`

1. loads raw and centroid split datasets (`train` or `val`) in TrajICL tensor format
2. selects at least `--num_samples` per pool (default 10) for sample visual panels
3. loads centroid metadata (`<split>_centroid_metadata.json`) + centroid primary-id list (`<split>_pedestrians_list.pickle`) and builds metadata-matched raw-vs-centroid sample pairs using `source_sample_index`
4. generates sample-level visuals:
   1. raw multi-agent sample grid
   2. centroid multi-agent sample grid
   3. raw-vs-centroid side-by-side sample pairs (matched by source sample where metadata exists)
5. paired comparison figures support coordinate modes controlled by `--pair_coordinate_mode`:
   1. `absolute`: keeps original scene coordinates
   2. `normalized`: subtracts each panel primary origin
   3. `both` (default): exports both absolute and normalized variants
6. paired comparison figures (`00_before_vs_after...`, `03_raw_vs_centroid_pairs...`) apply:
   1. shared axis limits across raw and centroid panel in the same pair
   2. origin normalization only for the `normalized` variants
7. generates dataset-level visuals:
   1. raw spatial occupancy heatmap
   2. centroid spatial occupancy heatmap
   3. agent count histogram/boxplot comparisons
   4. primary speed/displacement/heading distributions
   5. mean primary speed over timestep curve
8. saves summary statistics to JSON (`summary_stats.json`)
9. compiles all generated PNGs into a multi-page PDF report
10. writes outputs to `outputs/visualizations/raw_vs_centroid_<timestamp>/`
11. supports automatic run logging to `outputs/logs/viz_<timestamp>.log`
12. lazy-loads matplotlib so `--help` works even when plotting dependencies are not installed

### `viz_scene.py`

Purpose: visualize raw vs centroid trajectories by annotation scene id (for example `000`, `001`) instead of processed sample indices.

1. resolves MOTSynth root using the same loader logic as preprocessing (`resolve_motsynth_root`)
2. reads annotation files from `dataset/mot_annotations/<scene>/gt/gt.txt` (or equivalent resolved root)
3. builds raw per-agent trajectories directly from annotation frames
4. runs the same dynamic clustering runtime used in centroid preprocessing:
   1. nested direction+location initialization
   2. LOF-based reassignment every `reeval_interval`
   3. temporary-pool reclustering
   4. centroid track construction from cluster runtime
5. writes per-scene comparison artifacts:
   1. `<scene>_raw_vs_centroid.png`
   2. `<scene>_raw_tracks.csv`
   3. `<scene>_centroid_tracks.csv`
   4. `<scene>_centroid_metadata.json`
6. writes run manifest (`manifest.json`) containing counts and file paths
7. supports multiple scenes in one command via `--scenes 000,001,...`
8. supports user-provided raw agent count via `--n_agents`:
   1. `0` keeps all eligible raw agents
   2. `N>0` keeps top `N` raw agents (ranked by valid trajectory length) for plotting/export
9. supports user-provided centroid count via `--n_clusters`:
   1. `0` keeps all generated centroid tracks
   2. `N>0` keeps top `N` centroid tracks (ranked by `track_length * cluster_size`) for plotting/export
10. uses one distinct color per raw agent id and one distinct color per plotted centroid cluster

---

## 10) Visualization Interpretation Notes (`viz.py` outputs)

The first sample-grid images are often the most discussed in presentations:

1. `01_raw_samples_grid.png`
2. `02_centroid_samples_grid.png`

Color/style semantics in both:

1. blue trajectory = primary target track (index 0 in the sample tensor)
2. orange trajectories = other context tracks in the same sample
3. solid line segment = history (observed part, first `hist_len` frames)
4. dashed line segment = future segment
5. dot marker = start point
6. `x` marker = end point

Important nuance:

1. In `01_raw_samples_grid.png`, blue is a real pedestrian trajectory.
2. In `02_centroid_samples_grid.png`, blue is a centroid (cluster-representative) trajectory.
3. `01` and `02` are independent grids; they are not one-to-one matched pairs.
4. For one-to-one comparisons, use:
   1. `00_before_vs_after_raw_vs_centroid.png`
   2. `03_raw_vs_centroid_pairs.png`
5. For normalized shape comparisons, use:
   1. `00_before_vs_after_raw_vs_centroid_normalized.png`
   2. `03_raw_vs_centroid_pairs_normalized.png`
6. `00` and `03` are generated from metadata-matched samples and plotted with shared axis limits.
7. If your goal is to verify physical scene consistency (\"is this centroid where the raw group is?\"), use the absolute-coordinate files (`00`, `03`) instead of normalized files.

### Interpreting raw vs centroid summary stats

An observed pattern like this is expected in part:

```json
\"raw\": {
  \"agent_count_mean\": 8.8012,
  \"agent_count_median\": 7.0,
  \"primary_speed_mean\": 19.48,
  \"primary_displacement_mean\": 272.08
},
\"centroid\": {
  \"agent_count_mean\": 4.0398,
  \"agent_count_median\": 4.0,
  \"primary_speed_mean\": 0.97,
  \"primary_displacement_mean\": 18.39
}
```

Interpretation:

1. Lower centroid `agent_count_mean` is expected because multiple pedestrians are replaced by fewer cluster-representative tracks.
2. Centroid `primary_speed_mean` / `primary_displacement_mean` can still be lower than raw because one centroid represents multi-agent group motion, but centroid paths are now propagated per frame via delta updates (Eq. 3-4) rather than held constant between evaluation frames.
3. If you see staircase or dot-like centroid tracks, check whether the dataset was generated with an older preprocessing run and regenerate centroid data.

So these raw-vs-centroid speed/displacement values are still **not strictly apples-to-apples** because representation semantics differ (individual trajectories vs cluster representatives).

Recommended slide note:

1. treat agent-count reduction as the direct structural effect of clustering
2. treat speed/displacement comparisons as representation-dependent unless re-normalized

---

## 11) Benchmark Script Details (`compare_raw_vs_centroid.py`)

Purpose: reproducible side-by-side benchmark package for raw vs centroid pool performance.

### Inputs

1. `--model_path` (required)
2. `--dataset_name` (default `motsynth`)
3. `--prompting_method` (default `sim`)
4. `--shots` (default `0,2,4,8`)
5. `--processed_root` (default `outputs/processed_data`)
6. `--device` (default `cuda`)

### Pool checks

Before evaluation, script validates:

1. `outputs/processed_data/<dataset_name>` exists (raw)
2. `outputs/processed_data/<dataset_name>_centroid` exists (centroid)

### Metric computation

For each shot:

1. evaluates raw pool
2. evaluates centroid pool
3. records ADE/FDE for each
4. computes deltas and relative improvements:
   1. `ade_delta_raw_minus_centroid`
   2. `fde_delta_raw_minus_centroid`
   3. `ade_improve_pct`
   4. `fde_improve_pct`

### Outputs

In `outputs/comparison/raw_vs_centroid_<timestamp>/`:

1. `metrics_comparison.json`
2. `metrics_long.csv`
3. `metrics_summary.csv`
4. `ade_vs_shot_raw_vs_centroid.png`
5. `fde_vs_shot_raw_vs_centroid.png`
6. `ade_improve_pct_vs_shot.png`
7. `fde_improve_pct_vs_shot.png`

### Logging

Standard output/error is automatically logged to:

1. `outputs/logs/compare_raw_vs_centroid_<timestamp>.log`

### Dependency behavior

The script lazy-loads heavy dependencies so CLI help can run even if runtime
packages are missing in the current shell.

---

## 12) Benchmark Script Details (`compare_checkpoints.py`)

Purpose: compare two trained checkpoints in a single consolidated report.

Typical use case:

1. baseline = original-trained checkpoint
2. candidate = centroid-trained checkpoint

### Inputs

1. `--baseline_model_path` (required)
2. `--candidate_model_path` (required)
3. `--baseline_label` / `--candidate_label`
4. `--dataset_name`
5. `--prompting_method`
6. `--shots`
7. `--pools` (raw, centroid, or both)
8. `--processed_root`
9. `--device`

### Output metrics

For each pool and shot, script records:

1. baseline ADE/FDE
2. candidate ADE/FDE
3. delta (`baseline - candidate`)
4. improvement % (`(baseline - candidate)/baseline * 100`)

### Output files

In `outputs/comparison/checkpoint_vs_checkpoint_<timestamp>/`:

1. `checkpoint_comparison.json`
2. `metrics_long.csv`
3. `metrics_pairwise.csv`
4. plot set for each pool:
   1. ADE vs shot
   2. FDE vs shot
   3. ADE improvement % vs shot
   4. FDE improvement % vs shot

### Logging

Automatic run log:

1. `outputs/logs/compare_checkpoints_<timestamp>.log`

---

## 13) Operational Notes

1. If environment lacks required packages (`omegaconf`, etc.), script startup will fail before model/pipeline logic.
2. `preprocess.py` supports only known dataset names in `infer_r_stride` (`motsynth`, `jrdb`, `jta`).
3. `preprocess_centroids.py` currently processes each existing trajectory window independently (matching repository's sample-level data layout).
4. If training exits with only `Killed`, this is typically an OS OOM kill; recommended overrides are:
   1. `dataset.load_similarity_seq=false`
   2. `training.num_workers=0`
   3. lower `training.batch_size`
5. `viz.py` requires matplotlib at execution time; install with `pip install matplotlib` if missing.

---

## 14) Quick Verification Commands

```bash
python -m py_compile \
  utils/run_logging.py utils/data.py load_data.py dataset.py \
  compare_raw_vs_centroid.py compare_checkpoints.py viz.py \
  preprocess.py preprocess_centroids.py train.py eval.py

python preprocess.py --help
python preprocess_centroids.py --help
python eval.py --help
python compare_raw_vs_centroid.py --help
python compare_checkpoints.py --help
python viz.py --help

ls -1t outputs/logs | head
```

This verifies syntax, CLI, and log-file generation plumbing.
