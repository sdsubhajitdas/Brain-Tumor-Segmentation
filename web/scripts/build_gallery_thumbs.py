"""One-off script: generate WebP gallery thumbnails/full-size images from the
88 top-level images/{dice_score}_{idx}.png composite result images. Run once
locally; commit the output under web/static/gallery/. Not run at Docker build
or container-startup time -- the source images never change without a code
change, so regenerating on every deploy would be pure waste.
"""

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "images"
THUMBS_DIR = REPO_ROOT / "web" / "static" / "gallery" / "thumbs"
FULL_DIR = REPO_ROOT / "web" / "static" / "gallery" / "full"

THUMB_LONG_EDGE = 300
FULL_LONG_EDGE_CAP = 1600
WEBP_QUALITY = 80


def resized(image: Image.Image, long_edge: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= long_edge:
        return image
    scale = long_edge / max(width, height)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def main():
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    source_files = sorted(SOURCE_DIR.glob("*.png"))
    if not source_files:
        raise FileNotFoundError(f"No source PNGs found in {SOURCE_DIR}")

    for source_path in source_files:
        gallery_id = source_path.stem  # e.g. "0.98010_423"
        with Image.open(source_path) as image:
            image = image.convert("RGB")

            thumb = resized(image, THUMB_LONG_EDGE)
            thumb.save(THUMBS_DIR / f"{gallery_id}.webp", "WEBP", quality=WEBP_QUALITY)

            full = resized(image, FULL_LONG_EDGE_CAP)
            full.save(FULL_DIR / f"{gallery_id}.webp", "WEBP", quality=WEBP_QUALITY)

        print(f"Processed {gallery_id}")

    print(f"\nDone. {len(source_files)} images -> {THUMBS_DIR} and {FULL_DIR}")


if __name__ == "__main__":
    main()
