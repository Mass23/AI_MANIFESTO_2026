# AI_MANIFESTO_2026

Pipeline for image animation using a lightweight batch CLI around `process_image.py`.

## Usage

```bash
python AI_MANIFESTO_transformer.py --data "images/*.jpg" --out batch_1_test -n 5
```

## What it does

- loads all images matching the `--data` glob(s)
- processes each image `-n/--num-runs` times with different random seeds
- saves intermediate frames under `steps/`
- saves a final stylized output image for each run

## Output structure

```text
batch_1_test/
  image_name/
    run_000/
      steps/
        step_00.png
        ...
      output.png
      run_info.txt
```

## Install

```bash
pip install -r requirements.txt
```
