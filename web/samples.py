"""Registry of the curated sample images available in the "try it" picker.

These live in web/static/samples/, committed to git -- copied there once by
scripts/prepare_samples.py from images/API/. The web app never reads from the
large gitignored dataset/ folder.
"""

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "static" / "samples"

SAMPLE_IDS = [
    "3000",
    "3005",
    "3010",
    "3015",
    "3020",
    "3025",
    "3030",
    "3040",
    "3045",
    "3050",
    "3055",
    "3060",
]


def get_sample_paths(sample_id: str) -> tuple[Path, Path]:
    """Returns (image_path, mask_path) for a curated sample id."""
    if sample_id not in SAMPLE_IDS:
        raise KeyError(f"Unknown sample_id: {sample_id!r}")
    image_path = SAMPLES_DIR / f"{sample_id}.png"
    mask_path = SAMPLES_DIR / f"{sample_id}_mask.png"
    return image_path, mask_path
