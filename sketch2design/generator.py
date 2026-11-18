"""Load a trained BicycleGAN and generate designs from a sketch.

Two ways to choose the look of the output:

* ``random_designs``: sample random latent codes, giving several different
  colourways and fabrics for the same silhouette. This is the multimodal
  output BicycleGAN was built for.
* ``match_reference``: encode a *reference photo* (a saree or fabric you
  like) with BicycleGAN's encoder E and use that code for your sketch, so the
  new design borrows its colours and texture. The original repo only uses E
  to reconstruct the ground-truth photo of the same pair; here it is used as
  a style-transfer control across different garments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .sketch import SIZE, clean_user_sketch, fit_square

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "BicycleGAN"


def _to_tensor(img: Image.Image, channels: int):
    import torch

    a = np.asarray(img.convert("L" if channels == 1 else "RGB"), dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(a)
    t = t.unsqueeze(0) if channels == 1 else t.permute(2, 0, 1)
    return t.unsqueeze(0)


def _to_image(t) -> Image.Image:
    a = ((t[0].detach().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


class Designer:
    def __init__(self, checkpoints_dir: str = "checkpoints", name: str = "ethnicwear_bicycle_gan",
                 epoch: str = "latest", nz: int = 8, gpu: bool | None = None):
        import torch

        if not UPSTREAM.exists():
            raise SystemExit("BicycleGAN code not found. Run: bash scripts/setup.sh")
        sys.path.insert(0, str(UPSTREAM))
        from models import create_model  # noqa: E402  (upstream package)
        from options.test_options import TestOptions  # noqa: E402

        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        argv = [
            "x", "--dataroot", ".", "--name", name, "--checkpoints_dir", checkpoints_dir,
            "--model", "bicycle_gan", "--input_nc", "1", "--nz", str(nz),
            "--load_size", str(SIZE), "--crop_size", str(SIZE), "--epoch", epoch,
            "--use_dropout", "--gpu_ids", "0" if use_gpu else "-1",
        ]
        saved, sys.argv = sys.argv, argv
        try:
            opt = TestOptions().parse()
        finally:
            sys.argv = saved
        opt.num_threads, opt.batch_size, opt.serial_batches, opt.no_flip = 0, 1, True, True
        self.opt = opt
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        self.device = self.model.device
        self.nz = nz

    def _run(self, sketch: Image.Image, z):
        import torch

        with torch.no_grad():
            return _to_image(self.model.netG(_to_tensor(sketch, 1).to(self.device), z))

    def random_designs(self, sketch: Image.Image, n: int = 6, seed: int = 0, clean: bool = True) -> list[Image.Image]:
        import torch

        sk = clean_user_sketch(sketch) if clean else fit_square(sketch).convert("L")
        g = torch.Generator().manual_seed(seed)
        zs = torch.randn(n, self.nz, generator=g).to(self.device)
        return [self._run(sk, zs[[i]]) for i in range(n)]

    def match_reference(self, sketch: Image.Image, reference: Image.Image, clean: bool = True) -> Image.Image:
        import torch

        sk = clean_user_sketch(sketch) if clean else fit_square(sketch).convert("L")
        ref = _to_tensor(fit_square(reference), 3).to(self.device)
        with torch.no_grad():
            mu, _ = self.model.netE(ref)  # use the mean code: deterministic and cleanest
        return self._run(sk, mu)
