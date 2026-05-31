# AI_MANIFESTO_2026

Pipeline for image animation preprocessing plus incremental Hugging Face transformer training.

## Install

```bash
pip install -r requirements.txt
```

## CLI

```bash
python AI_MANIFESTO_transformer.py --data "images/*.jpg" --out batch_1_test
```

### Behavior

- Each input image is processed exactly once per CLI run.
- Input images are first normalized to a common working resolution so effects are applied consistently across mixed source sizes/aspect ratios.
- Intermediate step frames are written under `tmp/steps/<run_timestamp>/...`.
- Step frames are used as training data for the Hugging Face transformer training stage.
- Final outputs are kept at:
  - `output_folder/final_images/<source_name>_processed.jpg`
- For source images that are not already `3840x2160`, the processed output is composited on a black `3840x2160` canvas and scaled to fit within 90% of the frame while preserving aspect ratio.
- Temporary step frames are deleted after training unless `--keep-training-artifacts` is provided.
- Model/checkpoint cache is persisted in `output_folder/model_cache/` (or `--cache-dir`) so later runs continue training from the same saved state.

## Useful options

- `-n/--num-runs`: compatibility flag that must be `1`.
- `--tmp-steps-dir`: change temporary training-frame location (default `tmp/steps`).
- `--cache-dir`: override persistent model/checkpoint cache location.
- `--train-epochs`: epochs to train this run (incremental over cached checkpoints).
- `--skip-training`: run preprocessing only.
- `--keep-training-artifacts`: keep temporary step frames instead of deleting them.
