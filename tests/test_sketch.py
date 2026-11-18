"""Tests for the data side (no GPU, no model, no BicycleGAN checkout needed)."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_data  # noqa: E402
from sketch2design.sketch import SIZE, clean_user_sketch, fit_square, ink_ratio, is_usable, make_pair  # noqa: E402


def garment(w=300, h=600, colour=(180, 30, 60)) -> Image.Image:
    """A tall synthetic 'kurta': a coloured trapezoid with a border, on a grey background."""
    img = Image.new("RGB", (w, h), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.polygon([(w * 0.3, h * 0.1), (w * 0.7, h * 0.1), (w * 0.9, h * 0.9), (w * 0.1, h * 0.9)], fill=colour)
    d.line([(w * 0.1, h * 0.85), (w * 0.9, h * 0.85)], fill=(240, 200, 40), width=8)
    return img


def test_fit_square_keeps_whole_garment():
    sq = fit_square(garment(300, 600))
    assert sq.size == (SIZE, SIZE)
    a = np.asarray(sq)
    assert (a[:, :20] == 255).all()  # tall image -> white padding left and right, nothing cropped
    assert (a[5, SIZE // 2] != 255).any()  # top of the garment area is still there


def test_make_pair_layout():
    pair, sk = make_pair(garment())
    assert pair.size == (2 * SIZE, SIZE)
    assert sk.mode == "L"
    left = np.asarray(pair.crop((0, 0, SIZE, SIZE)))
    assert set(np.unique(left)) <= {0, 255}  # binary sketch
    right = np.asarray(pair.crop((SIZE, 0, 2 * SIZE, SIZE)))
    assert (right[..., 0] > right[..., 2] + 50).any()  # photo half keeps the red garment


def test_sketch_has_outline_but_is_mostly_paper():
    _, sk = make_pair(garment())
    r = ink_ratio(sk)
    assert 0.005 < r < 0.3
    assert is_usable(sk)


def test_blank_photo_is_rejected():
    _, sk = make_pair(Image.new("RGB", (256, 256), (230, 230, 230)))
    assert not is_usable(sk)


def test_canny_option_works():
    _, sk = make_pair(garment(), method="canny")
    assert 0 < ink_ratio(sk) < 0.3


def test_clean_user_sketch_flattens_uneven_lighting():
    # Pencil lines on paper with a dark vignette, like a phone photo.
    img = Image.new("L", (400, 400), 0)
    a = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[:400, :400]
    a = 230 - 0.35 * np.hypot(yy - 200, xx - 200)
    paper = Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGB")
    ImageDraw.Draw(paper).rectangle([120, 60, 280, 340], outline=(40, 40, 40), width=4)
    out = np.asarray(clean_user_sketch(paper))
    assert set(np.unique(out)) <= {0, 255}
    assert (out == 255).mean() > 0.85  # dark corners became white paper


def test_prepare_data_from_index(tmp_path):
    root = tmp_path / "ds"
    (root / "imgs").mkdir(parents=True)
    rows = []
    for i in range(6):
        cls = "saree" if i % 2 else "dhoti_pants"
        garment(colour=(30 * i, 80, 200)).save(root / "imgs" / f"{i}.jpg")
        rows.append({"image_path": f"imgs/{i}.jpg", "class_label": cls})
    (root / "train.json").write_text("\n".join(json.dumps(r) for r in rows))

    paths = prepare_data.from_index(root / "train.json", root, ["saree"])
    assert len(paths) == 3
    stats = prepare_data.build(paths, tmp_path / "out", 0.34, "xdog", 0.8, 0)
    assert stats["kept"] == 3 and stats["val"] == 1 and stats["train"] == 2
    assert Image.open(next((tmp_path / "out" / "train").glob("*.jpg"))).size == (2 * SIZE, SIZE)


def test_photo_border_is_not_drawn_as_a_line():
    # A plain tall photo gets white padding; the photo's edge must not become ink.
    _, sk = make_pair(Image.new("RGB", (200, 500), (120, 120, 120)))
    assert ink_ratio(sk) < 0.002
