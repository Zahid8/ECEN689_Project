<!-- <div align="center"> -->

# 

## Run Preprocessing Code

### DC-related Arguments

- `-dc` / `--dynamic_cluster_processing`: Build a clustered-pool dataset and save
  it as `processed_data/{name}_clustered/`. This path rewrites the processed
  dataset to use cluster centroids in the pool.
- `--dual_track_dc`: Keep target/query trajectories raw while storing DC pool
  examples separately as per-fold centroid pools. This saves to
  `processed_data/{name}_dual_dc/` and is the recommended option when evaluation
  should use the same raw target trajectories.
- `--dc_config <path>`: YAML config for dynamic clustering thresholds and
  clustering parameters. Use this with either `-dc` or `--dual_track_dc`.
- `--cluster_weight_alpha <float>`: Generate weighted STES dictionaries using
  cluster size:

  ```text
  S_weighted = S * (1 + alpha * log(1 + cluster_size))
  ```

The weighted file is saved as:

```text
processed_data/motsynth_dual_dc/{split}_similar_traj_dicts_hist_weighted.pickle
```

`load_data.py` supports three modes:

- `auto`: prefer weighted files if they exist.
- `on`: require weighted hist similarity.
- `off`: force the original non-weighted similarity.

For normal STES training, set this in the config if needed:

```yaml
dataset:
  use_weighted_similarity: auto  # auto | on | off
```

PG-ES also supports cluster weighting at runtime through:

```yaml
dataset:
  cluster_weight_alpha: 0.5
  pges_cluster_weight_alpha: 0.5  # optional PG-ES override
```


## Evaluation
Run evaluation thourgh:
```bash
python3 eval.py \
  --model_path <ckpt> \
  --dataset_name motsynth_dual_dc \
  --prompting_method sim \
  --weighted_similarity on
```

- `--weighted_similarity {auto,on,off}`: Control whether evaluation uses the
  weighted STES dictionary. `auto` prefers weighted files when present, `on`
  requires weighted hist similarity, and `off` forces the original non-weighted
  similarity.
- `--use_pges`: Enable PG-ES evaluation. It first uses hist STES to get the
  candidate pool, then re-ranks candidates using
  `[target_past + stage1_prediction]` against each pool example's
  `[past + future]`.

