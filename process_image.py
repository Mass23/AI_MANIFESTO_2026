from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image

NORMALIZED_WORKING_SIZE = (1280, 720)
FINAL_CANVAS_SIZE = (3840, 2160)
FINAL_CANVAS_FIT_RATIO = 0.9


@dataclass(frozen=True)
class ProcessingConfig:
    num_steps: int = 12
    effects_per_step: int = 64
    noise_stddev: float = 8.0


def save_image_rgb(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=95)


def place_on_black_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int],
    fit_ratio: float = 1.0,
) -> Image.Image:
    canvas_width, canvas_height = canvas_size
    max_width = max(1, int(round(canvas_width * fit_ratio)))
    max_height = max(1, int(round(canvas_height * fit_ratio)))
    source_width, source_height = image.size

    scale = min(max_width / source_width, max_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, color=(0, 0, 0))
    paste_x = (canvas_width - resized_width) // 2
    paste_y = (canvas_height - resized_height) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def get_random_rect(height: int, width: int, rng: random.Random, max_tries: int = 100) -> tuple[int, int, int, int] | None:
    area = height * width
    for _ in range(max_tries):
        target_area = area * rng.uniform(0.01, 0.1)
        aspect_ratio = rng.uniform(0.2, 5.0)

        rect_w = int(np.sqrt(target_area * aspect_ratio))
        rect_h = int(target_area / max(rect_w, 1))
        if rect_w <= 0 or rect_h <= 0:
            continue
        if rect_w >= width or rect_h >= height:
            continue

        x = rng.randint(0, width - rect_w)
        y = rng.randint(0, height - rect_h)
        return x, y, rect_w, rect_h

    return None


def effect_shuffle_rgb(patch: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
    return patch[:, :, np_rng.permutation(3)]


def effect_gaussian_noise(patch: np.ndarray, np_rng: np.random.Generator, stddev: float = 8.0) -> np.ndarray:
    noise = np_rng.normal(0, stddev, patch.shape).astype(np.int16)
    noisy = patch.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_effect(patch: np.ndarray, py_rng: random.Random, np_rng: np.random.Generator, noise_stddev: float) -> np.ndarray:
    effect_name = py_rng.choice(["effect_shuffle_rgb", "effect_gaussian_noise"])
    if effect_name == "effect_shuffle_rgb":
        return effect_shuffle_rgb(patch, np_rng=np_rng)
    return effect_gaussian_noise(patch, np_rng=np_rng, stddev=noise_stddev)


def process_image_once(
    input_path: str | Path,
    steps_dir: str | Path,
    final_output_path: str | Path,
    seed: int,
    config: ProcessingConfig,
) -> list[Path]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with Image.open(input_path).convert("RGB") as source_image:
        source_size = source_image.size
        normalized_image = place_on_black_canvas(
            source_image,
            canvas_size=NORMALIZED_WORKING_SIZE,
            fit_ratio=1.0,
        )

    working = np.asarray(normalized_image, dtype=np.uint8).copy()
    steps_dir = Path(steps_dir)
    steps_dir.mkdir(parents=True, exist_ok=True)

    step_paths: list[Path] = []

    for step in range(config.num_steps):
        height, width = working.shape[:2]
        for _ in range(config.effects_per_step):
            rect = get_random_rect(height, width, py_rng)
            if rect is None:
                continue

            x, y, rect_w, rect_h = rect
            patch = working[y : y + rect_h, x : x + rect_w]
            working[y : y + rect_h, x : x + rect_w] = apply_effect(
                patch,
                py_rng=py_rng,
                np_rng=np_rng,
                noise_stddev=config.noise_stddev,
            )

        step_path = steps_dir / f"step_{step:03d}.jpg"
        save_image_rgb(step_path, working)
        step_paths.append(step_path)

    final_image = Image.fromarray(working, mode="RGB")
    if source_size != FINAL_CANVAS_SIZE:
        final_image = place_on_black_canvas(
            final_image,
            canvas_size=FINAL_CANVAS_SIZE,
            fit_ratio=FINAL_CANVAS_FIT_RATIO,
        )
    else:
        final_image = final_image.resize(FINAL_CANVAS_SIZE, Image.Resampling.LANCZOS)

    save_image_rgb(final_output_path, np.asarray(final_image, dtype=np.uint8))
    return step_paths
