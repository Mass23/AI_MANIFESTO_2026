import argparse
import glob
import random
from pathlib import Path

from process_image import process_image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def expand_inputs(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process images into animated frame sequences."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help='Input image glob(s), e.g. --data "images/*.jpg" "images/*.png"',
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for processed batches",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        help="Number of times to process each image with different seeds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of saved processing steps per run",
    )
    parser.add_argument(
        "--effects-per-step",
        type=int,
        default=64,
        help="Number of patch effects applied per step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; if omitted, random seeds are generated",
    )

    args = parser.parse_args()

    input_files = expand_inputs(args.data)
    if not input_files:
        raise SystemExit("No input images matched the provided --data pattern(s).")

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    print(f"[info] Found {len(input_files)} image(s)")
    print(f"[info] Writing outputs to: {out_dir}")

    for image_path in input_files:
        stem = Path(image_path).stem
        image_out_root = out_dir / stem
        ensure_dir(image_out_root)

        for run_idx in range(args.num_runs):
            seed = args.seed + run_idx if args.seed is not None else random.randint(0, 10**9)
            run_out_dir = image_out_root / f"run_{run_idx:03d}"
            print(f"[info] Processing {image_path} -> {run_out_dir} (seed={seed})")

            process_image(
                input_path=image_path,
                output_dir=run_out_dir,
                seed=seed,
                num_steps=args.steps,
                effects_per_step=args.effects_per_step,
            )

    print("[done] Batch processing complete.")


if __name__ == "__main__":
    main()
