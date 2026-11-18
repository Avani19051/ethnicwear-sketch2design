"""Build a BicycleGAN-ready dataset of (sketch | photo) pairs from ethnic-wear photos.

Two ways to point it at photos:

    # any folder of garment photos (searched recursively)
    python prepare_data.py --photos my_photos/ --out datasets/ethnicwear

    # a dataset with a JSON-lines index, e.g. IndoFashion on Kaggle
    python prepare_data.py --index /kaggle/input/indo-fashion-dataset/train_data.json \
        --root /kaggle/input/indo-fashion-dataset --classes saree women_kurta lehenga \
        --limit 8000 --out datasets/ethnicwear

Output layout matches what BicycleGAN's aligned dataset loader expects:
    <out>/train/00001.jpg   (512x256: sketch on the left, photo on the right)
    <out>/val/00001.jpg
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from sketch2design.sketch import is_usable, make_pair

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def from_folder(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


def from_index(index: Path, root: Path, classes: list[str] | None) -> list[Path]:
    """Read a JSON-lines file with {"image_path": ..., "class_label": ...} per line."""
    paths = []
    with open(index) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if classes and row.get("class_label") not in classes:
                continue
            paths.append(root / row["image_path"])
    return paths


def build(paths: list[Path], out: Path, val_frac: float, method: str, sigma: float, seed: int) -> dict:
    random.Random(seed).shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac)) if len(paths) > 1 else 0
    split = {"val": paths[:n_val], "train": paths[n_val:]}
    stats = {"kept": 0, "too_sparse_or_busy": 0, "unreadable": 0}
    for phase, items in split.items():
        (out / phase).mkdir(parents=True, exist_ok=True)
        i = 0
        for p in items:
            try:
                photo = Image.open(p)
                photo.load()
            except (OSError, UnidentifiedImageError):
                stats["unreadable"] += 1
                continue
            pair, sk = make_pair(photo, method, sigma)
            if not is_usable(sk):
                stats["too_sparse_or_busy"] += 1
                continue
            i += 1
            pair.save(out / phase / f"{i:05d}.jpg", quality=95)
            stats["kept"] += 1
        stats[phase] = i
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--photos", type=Path, help="folder of garment photos")
    src.add_argument("--index", type=Path, help="JSON-lines index with image_path and class_label")
    ap.add_argument("--root", type=Path, default=Path("."), help="base folder for image_path in --index")
    ap.add_argument("--classes", nargs="*", help="keep only these class_label values")
    ap.add_argument("--limit", type=int, default=0, help="use at most this many photos (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("datasets/ethnicwear"))
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--method", choices=["xdog", "canny"], default="xdog")
    ap.add_argument("--sigma", type=float, default=0.8, help="XDoG detail: higher = fewer, bolder lines")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = from_folder(a.photos) if a.photos else from_index(a.index, a.root, a.classes)
    if a.limit:
        random.Random(a.seed).shuffle(paths)
        paths = paths[: a.limit]
    if not paths:
        raise SystemExit("No photos found.")
    print(f"{len(paths)} photos -> {a.out}")
    print(build(paths, a.out, a.val_frac, a.method, a.sigma, a.seed))


if __name__ == "__main__":
    main()
