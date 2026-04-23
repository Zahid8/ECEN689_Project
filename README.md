
### Evaluation branch

Raw model + Raw pool (baseline TrajICL):
```bash
python eval.py \
  --model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --example_pool_type raw \
  --prompting_method sim
```

Centroid pool + no weighting:
```bash
python eval.py \
  --model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --example_pool_type centroid \
  --prompting_method sim
```

Raw model + Centroid pool (this project):
```bash
python eval.py \
  --model_path outputs/TrajICL/raw/best_val_checkpoint.pth.tar \
  --dataset_name motsynth \
  --example_pool_type centroid \
  --prompting_method weighted_sim
```

## Output Structure

- `outputs/processed_data/motsynth/` -> raw processed pool
- `outputs/processed_data/motsynth_centroid/` -> centroid processed pool
- `clustered_dataset/motsynth_centroid/<split>/{scene_num}.csv` -> per-scene clustered centroid trajectories
- `outputs/TrajICL/raw/` -> raw-model checkpoints
- `outputs/TrajICL/centroid/` -> centroid-model checkpoints
- `outputs/comparison/` -> benchmark reports
- `outputs/logs/` -> captured run logs
