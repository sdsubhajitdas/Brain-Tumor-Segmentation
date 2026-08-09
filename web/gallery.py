"""Registry of gallery result images, built by scripts/build_gallery_thumbs.py
into web/static/gallery/{thumbs,full}/. Scanned dynamically rather than
hardcoded so the manifest can't drift from what's actually on disk.
"""

from pathlib import Path

GALLERY_DIR = Path(__file__).resolve().parent / "static" / "gallery"
THUMBS_DIR = GALLERY_DIR / "thumbs"
FULL_DIR = GALLERY_DIR / "full"


def list_gallery_items() -> list[dict]:
    """Each item: {'id': '0.98010_423', 'dice_label': '0.98010'}."""
    items = []
    for path in sorted(THUMBS_DIR.glob("*.webp")):
        gallery_id = path.stem
        dice_label = gallery_id.split("_", 1)[0]
        items.append({"id": gallery_id, "dice_label": dice_label})
    return items
