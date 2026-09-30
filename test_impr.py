"""Generate deterministic BicycleGAN fingerprint impressions.

This version adds two wrapper-only command-line options without requiring
changes to BicycleGAN's options/test_options.py:

    --latent-seed INT   Seed used for BicycleGAN latent sampling.
    --source-only       Ignore filenames that already look like impressions.

The custom options are removed from sys.argv before TestOptions parses the
normal BicycleGAN options.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMPRESSION_STEM_PATTERN = re.compile(r"^.+_\d+_cls\d+$", re.IGNORECASE)
CLASS_SUFFIX_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?P<class_suffix>_cls\d+)$",
    re.IGNORECASE,
)


def extract_custom_arguments(argv: Sequence[str]) -> Tuple[Optional[int], bool]:
    """Read local options and remove them before BicycleGAN parses argv."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--latent-seed", type=int, default=None)
    parser.add_argument("--source-only", action="store_true")

    custom, remaining = parser.parse_known_args(list(argv[1:]))
    sys.argv = [argv[0], *remaining]

    latent_seed = custom.latent_seed
    if latent_seed is None:
        environment_seed = os.environ.get("BICYCLEGAN_LATENT_SEED")
        if environment_seed not in (None, ""):
            try:
                latent_seed = int(environment_seed)
            except ValueError as exc:
                raise ValueError(
                    "BICYCLEGAN_LATENT_SEED must be an integer, got "
                    f"{environment_seed!r}"
                ) from exc

    return latent_seed, bool(custom.source_only)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and CUDA Torch RNGs."""

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_name(input_stem: str, impression_number: int) -> str:
    """Insert the impression number before a trailing ``_clsN`` suffix.

    Examples:
        seed0002_cls1 -> seed0002_1_cls1.png
        seed0002      -> seed0002_1.png
    """

    match = CLASS_SUFFIX_PATTERN.match(input_stem)
    if match:
        return (
            f"{match.group('prefix')}_{impression_number}"
            f"{match.group('class_suffix')}.png"
        )

    return f"{input_stem}_{impression_number}.png"


def main() -> None:
    latent_seed, source_only = extract_custom_arguments(sys.argv)

    # Seed before model creation/setup in case any model component uses RNG.
    if latent_seed is not None:
        seed_everything(latent_seed)
        print(f"Using BicycleGAN latent seed: {latent_seed}")
    else:
        print("No --latent-seed supplied; BicycleGAN sampling is non-deterministic.")

    # ----------------------------
    # Standard BicycleGAN options
    # ----------------------------
    opt = TestOptions().parse()

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # ----------------------------
    # Input / output folders
    # ----------------------------
    input_dir = Path(opt.dataroot).expanduser().resolve()
    output_dir = Path(opt.results_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Model
    # ----------------------------
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    print("Loading model %s" % opt.model)
    print("Reading images from:", input_dir)
    print("Saving impressions to:", output_dir)

    # ----------------------------
    # Transform
    # ----------------------------
    transform_A = get_transform(
        opt,
        grayscale=(opt.input_nc == 1),
    )

    # ----------------------------
    # Image list
    # ----------------------------
    A_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not (source_only and IMPRESSION_STEM_PATTERN.match(path.stem))
    )

    if not A_paths:
        raise RuntimeError(f"No input images found in {input_dir}")

    print(f"Found {len(A_paths)} input image(s)")

    # ----------------------------
    # Generate impressions
    # ----------------------------
    processed_total = min(len(A_paths), opt.num_test)

    for image_index, A_path in enumerate(islice(A_paths, opt.num_test)):
        print(
            "process input image %3.3d/%3.3d: %s"
            % (image_index + 1, processed_total, A_path.name)
        )

        A_img = Image.open(A_path).convert("RGB")
        A_tensor = transform_A(A_img).unsqueeze(0)

        # Dummy B for model compatibility.
        if opt.output_nc == 3 and A_tensor.shape[1] == 1:
            B_tensor = A_tensor.repeat(1, 3, 1, 1)
        else:
            B_tensor = A_tensor.clone()

        data = {
            "A": A_tensor,
            "B": B_tensor,
            "A_paths": [str(A_path)],
            "B_paths": [str(A_path)],
        }

        model.set_input(data)

        # Re-seed immediately before latent generation. This prevents model
        # setup or the previous input image from shifting the requested stream.
        if latent_seed is not None:
            per_image_seed = latent_seed + image_index * 1_000_003
            seed_everything(per_image_seed)
            print(f"  latent stream seed: {per_image_seed}")

        z_samples = model.get_z_random(opt.n_samples, opt.nz)

        for sample_index in range(opt.n_samples):
            _real_A, fake_B, _real_B = model.test(
                z_samples[[sample_index]],
                encode=False,
            )

            image = fake_B[0].detach().cpu()
            image = image.permute(1, 2, 0).numpy()
            image = (image + 1.0) * 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)

            save_name = make_output_name(
                A_path.stem,
                sample_index + 1,
            )
            save_path = output_dir / save_name

            Image.fromarray(image).save(save_path)
            print("saved:", save_name)

    print("Done.")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
