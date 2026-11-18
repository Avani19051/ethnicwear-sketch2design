# Ethnic-wear Sketch2Design

**Draw the outline of a saree, kurta or lehenga and get several finished design options, or your sketch in the colours and fabric of a reference photo you like.**

The project retrains [BicycleGAN](https://github.com/junyanz/BicycleGAN) (Zhu et al., NeurIPS 2017) on Indian ethnic wear instead of the original shoes and handbags. It adds a sketch-generation pipeline suited to embroidered garments, a reference-style control, evaluation metrics and a web demo. The idea came from sustainable-fashion work: a designer can try many colourways digitally before cutting any fabric.

> Built on **junyanz/BicycleGAN** (BSD licence) by Jun-Yan Zhu, Richard Zhang, Deepak Pathak et al. Their code is fetched at a pinned commit by `scripts/setup.sh` and is not copied here; its licence is in [LICENSE-BicycleGAN](LICENSE-BicycleGAN). Everything in this repo is my own work.

## How it works

```mermaid
flowchart LR
    A[Garment photos<br/>saree · kurta · lehenga] --> B[fit to square<br/>pad, don't crop]
    B --> C[XDoG line drawing<br/>adaptive threshold]
    C --> D{usable?<br/>1–30% ink}
    D -- yes --> E[sketch | photo pairs]
    E --> F[BicycleGAN training<br/>T4 GPU]
    F --> G[Random designs<br/>many colourways]
    F --> H[Match reference<br/>encoder E → style code]
    G --> I[Gradio demo · metrics]
    H --> I
```

## What I added on top of BicycleGAN

| Piece | Why it matters |
|---|---|
| **Sketch generation for garments** (`sketch2design/sketch.py`) | Ethnic-wear datasets only have photos, so the sketch half of each pair has to be generated. The original edges2shoes data uses edge detection, which turns zari borders and prints into dense noise. I use **XDoG** with a threshold that adapts to each photo's contrast, giving clean pencil-style outlines for both a pastel kurta on a pale wall and a dark saree on white. |
| **Pad, don't crop** | Garment photos are tall. Square-cropping cuts off the hem or neckline, which are the design details that matter. Photos are padded to a square, and the sketch is drawn *before* padding so the photo's own edge isn't drawn as a fake line (there's a test for this). |
| **Quality filter** | Pairs whose sketch is almost blank (blurry or plain photos) or full of noise (busy backgrounds) are dropped before training. |
| **Hand-drawn input cleanup** | `clean_user_sketch` uses adaptive thresholding, so a phone photo of a pencil sketch on paper, with uneven light and grey paper, looks like the training sketches. |
| **Match a reference** (`Designer.match_reference`) | BicycleGAN trains an encoder E but only uses it to rebuild the ground-truth photo of the *same* pair. Here E encodes *any* garment or fabric photo into a style code, so you can say "this sketch, in that saree's colours". |
| **Evaluation** (`evaluate.py`) | LPIPS **diversity** (are the random designs actually different?), **best-of-k** (can the model produce something close to the real garment?) and **encoded** fidelity, plus a results sheet image. |
| **Kaggle-ready scripts and a Gradio demo** | `scripts/train.sh` turns off the visdom server, which isn't available on Kaggle or Colab. `app.py` shares a public demo link. |

## Run it (Kaggle, free T4 GPU)

Turn on **Settings → Accelerator → GPU T4** and **Internet**. Add a garment-photo dataset as input, for example IndoFashion (check its licence and terms before use).

```bash
git clone https://github.com/Avani19051/ethnicwear-sketch2design && cd ethnicwear-sketch2design
bash scripts/setup.sh                       # fetches BicycleGAN at a pinned commit + installs deps

# 1. build sketch|photo pairs (about 8k photos is a good size for a T4)
python prepare_data.py \
  --index /kaggle/input/indo-fashion-dataset/train_data.json \
  --root  /kaggle/input/indo-fashion-dataset \
  --classes saree women_kurta lehenga --limit 8000

# 2. train (EPOCHS=30 → 60 epochs total, roughly 3–4 h; EPOCHS=2 for a quick check)
bash scripts/train.sh

# 3. metrics + results sheet
python evaluate.py                          # outputs/metrics.json, outputs/results.png

# 4. demo
python app.py                               # prints a public gradio.live link
```

With your own photos, use `python prepare_data.py --photos my_folder/` instead. Check the field names in the dataset's index file and adjust `--classes` to match.

The data-side tests run without a GPU or model:

```bash
pip install numpy pillow opencv-python-headless pytest && pytest -q
```

## Results

Not trained yet. After the first full Kaggle run, this section will show `outputs/results.png` (sketch, real photo, matched style and 6 random designs per row) and the numbers from `outputs/metrics.json`.

## Project structure

```
sketch2design/
  sketch.py      photo → sketch (XDoG/Canny), padding, quality filter, user-sketch cleanup
  generator.py   load trained BicycleGAN; random_designs() and match_reference()
prepare_data.py  build the aligned sketch|photo dataset (folder or JSON index)
evaluate.py      LPIPS diversity / best-of-k / encoded fidelity + results sheet
app.py           Gradio demo
scripts/         setup.sh (fetch BicycleGAN), train.sh
tests/           data pipeline tests (CPU only)
```

## Licence

This repo's code: BSD 2-Clause ([LICENSE](LICENSE)). BicycleGAN: BSD, © 2017 Jun-Yan Zhu, Richard Zhang, Deepak Pathak ([LICENSE-BicycleGAN](LICENSE-BicycleGAN)). Datasets you train on have their own terms.
