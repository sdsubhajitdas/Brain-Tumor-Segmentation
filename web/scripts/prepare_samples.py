"""One-off script: copy hand-picked sample image/mask pairs from images/API/
into web/static/samples/, so the web app never depends on the large gitignored
dataset/ folder. Run once locally; commit the output. Not run at Docker build
or request time.
"""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_IMAGE_DIR = REPO_ROOT / "images" / "API" / "Original Image"
SOURCE_MASK_DIR = REPO_ROOT / "images" / "API" / "Original Mask"
DEST_DIR = REPO_ROOT / "web" / "static" / "samples"

# Hand-picked for a spread of tumor sizes/positions (see web-app branch history).
CURATED_IDS = [3000, 3005, 3010, 3015, 3020, 3025, 3030, 3040, 3045, 3050, 3055, 3060]


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for idx in CURATED_IDS:
        image_src = SOURCE_IMAGE_DIR / f"{idx}.png"
        mask_src = SOURCE_MASK_DIR / f"{idx}_mask.png"
        if not image_src.is_file() or not mask_src.is_file():
            raise FileNotFoundError(f"Missing source pair for idx {idx}")
        shutil.copy(image_src, DEST_DIR / f"{idx}.png")
        shutil.copy(mask_src, DEST_DIR / f"{idx}_mask.png")
        print(f"Copied sample {idx}")
    print(f"\nDone. {len(CURATED_IDS)} pairs copied to {DEST_DIR}")


if __name__ == "__main__":
    main()
