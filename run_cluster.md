# Run Raw-to-Cluster Pipeline

Quick reference for `raw2cluster.py`.

## What this script does

- Runs clustering on trajectory data.
- For MOT-style `gt.txt` (6+ columns), uses bbox center coordinates (`bb_center_x`, `bb_center_y`), same idea as `utils/data.py`.
- For simple 4-column `frame id x y` text, uses columns 2–3 as x/y.
- Can save:
  - cluster CSV output
  - side-by-side raw vs cluster trajectory figure
- Can process one scene or batch scenes.

## Frame range (`start` / `finish`)

- `start`: inclusive first frame
- `finish`: exclusive end frame
- Processed range is `[start, finish)`

Example: `start=75`, `finish=120` processes frames `75..119`.

## Single-scene run

```bash
MPLBACKEND=Agg ../.venv/bin/python raw2cluster.py \
  --input ../data/motsynth/mot_annotations/000/gt/gt.txt \
  --start 75 --finish 120 \
  --tdist 110 --tdirect 50
```

Defaults for single-scene mode:
- CSV: `../data/motsynth_cluster/<scene>.csv`
- Figure: `../figures/<scene>_raw_vs_cluster.png`

## Batch run (all MOTSynth scenes)

```bash
MPLBACKEND=Agg ../.venv/bin/python raw2cluster.py \
  --batch-data-root ../data \
  --batch-output-root ../figures/raw2cluster_batch \
  --start 75 --finish 950 \
  --tdist 110 --tdirect 50
```

## Batch run (selected scenes only)

```bash
MPLBACKEND=Agg ../.venv/bin/python raw2cluster.py \
  --batch-data-root ../data \
  --batch-output-root ../figures/raw2cluster_batch \
  --batch-scenes 000,001 \
  --start 75 --finish 950
```

## Most useful parameters

- `--tdist`: spatial threshold for cluster assignment
- `--tdirect`: direction threshold for cluster assignment
- `--n-initial-cluster`: initial cluster count on start frame
- `--eval-frame-interval`: how often distance-eval rows are computed

## Notes

- Use `MPLBACKEND=Agg` in headless environments.
- Outputs in batch mode are saved under:
  - `<batch-output-root>/csv`
  - `<batch-output-root>/figures`
