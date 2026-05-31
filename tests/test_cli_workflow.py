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
    def _create_image(self, path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (24, 24), color=color).save(path, format="JPEG")

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
            self.assertTrue((final_images_dir / "a_processed.jpg").exists())
            self.assertTrue((final_images_dir / "b_processed.jpg").exists())

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
