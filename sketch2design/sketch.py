"""Turn garment photos into sketch-style line drawings, and build BicycleGAN training pairs.

BicycleGAN learns sketch -> photo from aligned pairs. Public ethnic-wear
datasets only have photos, so the sketch side is generated here. Plain Canny
edges (what edges2shoes used) pick up every fold and print as noise; on
heavily embroidered sarees and kurtas that gives a dense mess no designer
would draw. XDoG (eXtended Difference of Gaussians) instead gives clean,
pencil-like outlines with a controllable amount of detail, which is much
closer to what a person sketches.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

SIZE = 256


def _resize_long(img: Image.Image, size: int = SIZE) -> Image.Image:
    img = img.convert("RGB")
    scale = size / max(img.size)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)


def _pad(img: Image.Image, size: int = SIZE, fill=(255, 255, 255)) -> Image.Image:
    canvas = Image.new(img.mode, (size, size), fill if img.mode == "RGB" else 255)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def fit_square(img: Image.Image, size: int = SIZE) -> Image.Image:
    """Resize the longer side to `size` and pad to a square on a white background.

    Garment photos are usually tall (a full saree drape or a kurta on a
    model); cropping them square cuts off the hem or the neckline, which are
    exactly the design details that matter. Padding keeps the whole garment.
    """
    return _pad(_resize_long(img, size), size)


def xdog(gray: np.ndarray, sigma: float = 0.8, k: float = 1.6, gamma: float = 0.98,
         phi: float = 200.0, rel: float = 0.3) -> np.ndarray:
    """XDoG line drawing (Winnemöller et al., 2012). Input/output uint8, lines dark on white.

    The ink threshold adapts to the image's own edge strength, so a
    pastel kurta on a pale wall and a dark saree on white both get outlines.
    Raise `sigma` for fewer, bolder lines (rough sketch); lower it for more
    detail (embroidery and borders show up).
    """
    g = gray.astype(np.float32) / 255.0
    d = cv2.GaussianBlur(g, (0, 0), sigma) - gamma * cv2.GaussianBlur(g, (0, 0), sigma * k)
    neg = -d[d < 0]
    eps = min(-0.003, -rel * float(np.percentile(neg, 99))) if neg.size else -0.003
    out = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def canny(gray: np.ndarray) -> np.ndarray:
    """Canny edges with thresholds set from the image median, lines dark on white."""
    v = float(np.median(gray))
    lo, hi = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), lo, hi)
    return 255 - edges


def to_sketch(img: Image.Image, method: str = "xdog", sigma: float = 0.8) -> Image.Image:
    """Photo -> single-channel sketch image of the same size."""
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)  # smooth fabric texture, keep outlines
    lines = xdog(gray, sigma=sigma) if method == "xdog" else canny(gray)
    # Binarise: a hand sketch is ink or paper, not grey shading.
    _, binary = cv2.threshold(lines, 200, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary, mode="L")


def ink_ratio(sketch: Image.Image) -> float:
    """Fraction of pixels that are lines. Used to drop useless pairs."""
    a = np.asarray(sketch.convert("L"))
    return float((a < 128).mean())


def is_usable(sketch: Image.Image, lo: float = 0.01, hi: float = 0.30) -> bool:
    """Reject near-blank sketches (blurry or plain photos) and noise-filled ones (busy backgrounds)."""
    return lo <= ink_ratio(sketch) <= hi


def make_pair(photo: Image.Image, method: str = "xdog", sigma: float = 0.8) -> tuple[Image.Image, Image.Image]:
    """Return (A|B pair image of size 2*SIZE x SIZE, sketch). A = sketch, B = photo.

    The sketch is drawn *before* padding, so the edge of the photo itself
    (where its background meets the white padding) is not drawn as a line.
    """
    small = _resize_long(photo)
    sk = _pad(to_sketch(small, method, sigma))
    sq = _pad(small)
    pair = Image.new("RGB", (SIZE * 2, SIZE))
    pair.paste(sk.convert("RGB"), (0, 0))
    pair.paste(sq, (SIZE, 0))
    return pair, sk


def clean_user_sketch(img: Image.Image) -> Image.Image:
    """Prepare a hand-drawn or phone-photographed sketch for the model.

    Phone photos of paper have uneven lighting and grey paper; adaptive
    thresholding turns them into clean black-on-white lines like the
    training sketches.
    """
    sq = fit_square(img)
    gray = cv2.cvtColor(np.asarray(sq), cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return Image.fromarray(binary, mode="L")
