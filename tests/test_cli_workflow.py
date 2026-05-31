from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "AI_MANIFESTO_transformer.py"


class CliWorkflowTests(unittest.TestCase):
    def _create_image(
        self,
        path: Path,
        color: tuple[int, int, int],
        size: tuple[int, int] = (24, 24),
    ) -> None:
        Image.new("RGB", size, color=color).save(path, format="JPEG")

    def test_generates_final_images_and_cleans_tmp_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            out_dir = root / "out"
            steps_root = root / "tmp_steps"
            images_dir.mkdir(parents=True)

            self._create_image(images_dir / "a.jpg", (255, 0, 0))
            self._create_image(images_dir / "b.jpg", (0, 255, 0))

            cmd = [
                sys.executable,
                str(CLI_PATH),
                "--data",
                str(images_dir / "*.jpg"),
                "--out",
                str(out_dir),
                "--tmp-steps-dir",
                str(steps_root),
                "--steps",
                "2",
                "--effects-per-step",
                "2",
                "--skip-training",
            ]
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)

            summary = json.loads(result.stdout)
            self.assertEqual(summary["processed_images"], 2)

            final_images_dir = out_dir / "final_images"
            final_a = final_images_dir / "a_processed.jpg"
            final_b = final_images_dir / "b_processed.jpg"
            self.assertTrue(final_a.exists())
            self.assertTrue(final_b.exists())

            for final_image in (final_a, final_b):
                with Image.open(final_image) as image:
                    self.assertEqual(image.size, (3840, 2160))
                    corner = image.getpixel((0, 0))
                    self.assertTrue(all(channel <= 10 for channel in corner))

            self.assertFalse(steps_root.exists())

    def test_rejects_num_runs_other_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            out_dir = root / "out"
            images_dir.mkdir(parents=True)
            self._create_image(images_dir / "a.jpg", (255, 0, 0))

            cmd = [
                sys.executable,
                str(CLI_PATH),
                "--data",
                str(images_dir / "*.jpg"),
                "--out",
                str(out_dir),
                "-n",
                "2",
                "--skip-training",
            ]
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
