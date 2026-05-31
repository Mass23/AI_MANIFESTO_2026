from __future__ import annotations

import argparse
import glob
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from process_image import ProcessingConfig, process_image_once


@dataclass(frozen=True)
class StepFrameRecord:
    frame_path: Path
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental cached training workflow for AI_MANIFESTO image animation."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help='Input image globs, e.g. --data "images/*.jpg" "images/*.png"',
    )
    parser.add_argument("--out", required=True, help="Output folder for run artifacts")
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        help="Compatibility flag; must remain 1 because each image is processed once per run.",
    )
    parser.add_argument("--steps", type=int, default=12, help="Intermediate step frames per image")
    parser.add_argument(
        "--effects-per-step",
        type=int,
        default=64,
        help="Patch effects applied per intermediate step",
    )
    parser.add_argument("--noise-stddev", type=float, default=8.0, help="Gaussian noise standard deviation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic runs")
    parser.add_argument(
        "--tmp-steps-dir",
        default="tmp/steps",
        help="Temporary step frame directory used as training data",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Persistent cache/checkpoint directory (default: <out>/model_cache)",
    )
    parser.add_argument(
        "--keep-training-artifacts",
        action="store_true",
        help="Preserve temporary step frames after training",
    )
    parser.add_argument("--model-name", default="google/vit-base-patch16-224", help="HF model to fine-tune")
    parser.add_argument("--train-epochs", type=float, default=1.0, help="Epochs to train this run")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Trainer batch size")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Hugging Face training stage (useful for quick preprocessing checks)",
    )
    return parser.parse_args()


def expand_input_files(patterns: list[str]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            match_path = Path(match)
            if match_path.is_file():
                matches.add(match_path.resolve())
    return sorted(matches)


def train_incremental(
    step_records: list[StepFrameRecord],
    cache_dir: Path,
    model_name: str,
    train_epochs: float,
    train_batch_size: int,
) -> dict:
    try:
        import torch
        from PIL import Image
        from torch.utils.data import Dataset
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except Exception as exc:  # pragma: no cover - import-time dependency gate
        raise RuntimeError(
            "Training dependencies are unavailable. Install requirements or run with --skip-training."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cache_dir / "model"
    checkpoints_dir = cache_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({record.label for record in step_records})
    num_labels = len(labels)
    label_lookup = {label: index for index, label in enumerate(labels)}

    processor = AutoImageProcessor.from_pretrained(model_name)

    class StepFramesDataset(Dataset):
        def __init__(self, records: list[StepFrameRecord]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            record = self.records[index]
            with Image.open(record.frame_path).convert("RGB") as image:
                encoded = processor(images=image, return_tensors="pt")
            return {
                "pixel_values": encoded["pixel_values"].squeeze(0),
                "labels": torch.tensor(label_lookup[record.label], dtype=torch.long),
            }

    dataset = StepFramesDataset(step_records)

    if model_dir.exists():
        model = AutoModelForImageClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    last_checkpoint = get_last_checkpoint(str(checkpoints_dir))

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=train_batch_size,
        num_train_epochs=train_epochs,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(str(model_dir))
    processor.save_pretrained(model_dir)

    checkpoint_after_run = get_last_checkpoint(str(checkpoints_dir))
    state = {
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "num_frames": len(step_records),
        "num_source_images": num_labels,
        "last_checkpoint": checkpoint_after_run,
        "model_dir": str(model_dir),
        "global_step": trainer.state.global_step,
    }
    (cache_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    args = parse_args()

    if args.num_runs != 1:
        raise SystemExit("--num-runs/-n must be 1; each source image is processed exactly once per CLI run.")

    input_files = expand_input_files(args.data)
    if not input_files:
        raise SystemExit("No input images matched the provided --data globs.")

    output_dir = Path(args.out).resolve()
    final_images_dir = output_dir / "final_images"
    final_images_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    tmp_steps_root = Path(args.tmp_steps_dir).resolve()
    steps_root = tmp_steps_root / run_tag
    steps_root.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (output_dir / "model_cache")

    process_config = ProcessingConfig(
        num_steps=args.steps,
        effects_per_step=args.effects_per_step,
        noise_stddev=args.noise_stddev,
    )

    step_records: list[StepFrameRecord] = []
    for image_index, image_path in enumerate(input_files):
        image_steps_dir = steps_root / image_path.stem
        final_path = final_images_dir / f"{image_path.stem}_processed.jpg"
        image_seed = args.seed + image_index

        step_paths = process_image_once(
            input_path=image_path,
            steps_dir=image_steps_dir,
            final_output_path=final_path,
            seed=image_seed,
            config=process_config,
        )

        step_records.extend(StepFrameRecord(frame_path=step_path, label=image_index) for step_path in step_paths)

    training_state = None
    if not args.skip_training:
        training_state = train_incremental(
            step_records=step_records,
            cache_dir=cache_dir,
            model_name=args.model_name,
            train_epochs=args.train_epochs,
            train_batch_size=args.train_batch_size,
        )

    if not args.keep_training_artifacts:
        shutil.rmtree(steps_root, ignore_errors=True)
        if tmp_steps_root.exists() and not any(tmp_steps_root.iterdir()):
            tmp_steps_root.rmdir()

    summary = {
        "processed_images": len(input_files),
        "final_images_dir": str(final_images_dir),
        "tmp_steps_dir": str(steps_root),
        "training_cache_dir": str(cache_dir),
        "training_state": training_state,
        "preserved_tmp_steps": bool(args.keep_training_artifacts),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
