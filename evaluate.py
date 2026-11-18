"""Score a trained model on the validation sketches and save a results sheet.

    python evaluate.py --data datasets/ethnicwear --n_samples 6 --num 50

Metrics (LPIPS: lower = more similar):
  * diversity:    mean LPIPS between random designs for the same sketch.
                  Higher means more varied colourways; near 0 means the
                  model ignores the random code ("mode collapse").
  * best_of_k:    LPIPS from the real photo to the closest of the k random
                  designs. Lower means the real garment is among the things
                  the model can produce.
  * encoded:      LPIPS between the real photo and the design made from its
                  own encoded style. Lower means the encoder captures the look.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sketch2design.generator import Designer, _to_tensor
from sketch2design.sketch import SIZE


def split_pair(path: Path) -> tuple[Image.Image, Image.Image]:
    ab = Image.open(path).convert("RGB")
    return ab.crop((0, 0, SIZE, SIZE)).convert("L"), ab.crop((SIZE, 0, 2 * SIZE, SIZE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("datasets/ethnicwear"))
    ap.add_argument("--name", default="ethnicwear_bicycle_gan")
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--num", type=int, default=50, help="validation pairs to score")
    ap.add_argument("--grid_rows", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    a = ap.parse_args()

    import lpips
    import torch

    d = Designer(name=a.name)
    metric = lpips.LPIPS(net="alex").to(d.device)

    def dist(x: Image.Image, y: Image.Image) -> float:
        with torch.no_grad():
            return float(metric(_to_tensor(x, 3).to(d.device), _to_tensor(y, 3).to(d.device)))

    pairs = sorted((a.data / "val").glob("*.jpg"))[: a.num]
    if not pairs:
        raise SystemExit(f"No validation pairs in {a.data / 'val'}. Run prepare_data.py first.")

    div, best, enc, rows = [], [], [], []
    for i, p in enumerate(pairs):
        sketch, real = split_pair(p)
        designs = d.random_designs(sketch, n=a.n_samples, seed=i, clean=False)
        encoded = d.match_reference(sketch, real, clean=False)
        div.append(np.mean([dist(x, y) for x, y in itertools.combinations(designs, 2)]))
        best.append(min(dist(real, x) for x in designs))
        enc.append(dist(real, encoded))
        if len(rows) < a.grid_rows:
            rows.append([sketch.convert("RGB"), real, encoded, *designs])

    summary = {
        "pairs": len(pairs),
        "samples_per_sketch": a.n_samples,
        "diversity": round(float(np.mean(div)), 4),
        "best_of_k": round(float(np.mean(best)), 4),
        "encoded": round(float(np.mean(enc)), 4),
    }
    a.out.mkdir(exist_ok=True)
    (a.out / "metrics.json").write_text(json.dumps(summary, indent=2))

    header = ["sketch", "real", "matched", *[f"random {k+1}" for k in range(a.n_samples)]]
    cols, pad, top = len(header), 4, 20
    sheet = Image.new("RGB", (cols * (SIZE + pad), top + len(rows) * (SIZE + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, h in enumerate(header):
        draw.text((c * (SIZE + pad) + 6, 4), h, fill="black")
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            sheet.paste(img, (c * (SIZE + pad), top + r * (SIZE + pad)))
    sheet.save(a.out / "results.png")
    print(json.dumps(summary, indent=2))
    print(f"saved {a.out / 'results.png'}")


if __name__ == "__main__":
    main()
