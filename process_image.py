import cv2
import numpy as np
import random
from pathlib import Path


NUM_STEPS = 12
EFFECTS_PER_STEP = 64


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image_rgb(path, image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def get_random_rect(img, max_tries=100):
    h, w, _ = img.shape
    area = h * w

    for _ in range(max_tries):
        target_area = area * random.uniform(0.01, 0.1)
        aspect_ratio = random.uniform(0.2, 5.0)

        rect_w = int(np.sqrt(target_area * aspect_ratio))
        rect_h = int(target_area / max(rect_w, 1))

        if rect_w <= 0 or rect_h <= 0:
            continue
        if rect_w >= w or rect_h >= h:
            continue

        x = random.randint(0, w - rect_w)
        y = random.randint(0, h - rect_h)
        return x, y, rect_w, rect_h

    return None


def effect_shuffle_rgb(patch):
    return patch[:, :, np.random.permutation(3)]


def effect_gaussian_noise(patch):
    noise = np.random.normal(0, 8, patch.shape).astype(np.int16)
    noisy = patch.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def cinematic_grade(img):
    img = img.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * 1.2 + 0.5, 0, 1)

    img[:, :, 0] *= 0.95
    img[:, :, 2] *= 1.05

    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) / (center_x**2 + center_y**2)
    vignette = 1 - 0.4 * mask

    img *= vignette[..., np.newaxis]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def process_image(
    input_path,
    output_dir,
    seed,
    num_steps=NUM_STEPS,
    effects_per_step=EFFECTS_PER_STEP,
):
    random.seed(seed)
    np.random.seed(seed)

    image = load_image_rgb(input_path)
    working = image.copy()

    output_dir = Path(output_dir)
    steps_dir = output_dir / "steps"
    ensure_dir(steps_dir)

    effects = [
        effect_shuffle_rgb,
        effect_gaussian_noise,
    ]

    for step in range(num_steps):
        for _ in range(effects_per_step):
            rect = get_random_rect(working)
            if rect is None:
                continue

            x, y, rw, rh = rect
            patch = working[y:y + rh, x:x + rw]

            effect = random.choice(effects)
            new_patch = effect(patch)
            working[y:y + rh, x:x + rw] = new_patch

        working = cv2.resize(
            working,
            None,
            fx=1.02,
            fy=1.02,
            interpolation=cv2.INTER_LINEAR,
        )

        save_image_rgb(steps_dir / f"step_{step:02d}.png", working)

    final = cinematic_grade(working)
    save_image_rgb(output_dir / "output.png", final)

    (output_dir / "run_info.txt").write_text(
        f"input={input_path}\nseed={seed}\nnum_steps={num_steps}\n"
        f"effects_per_step={effects_per_step}\n",
        encoding="utf-8",
    )

    return output_dir / "output.png"
