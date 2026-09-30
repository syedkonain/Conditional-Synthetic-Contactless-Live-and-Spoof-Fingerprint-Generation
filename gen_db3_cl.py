import os
import random
import csv
import subprocess
from typing import Optional, Union, List
import re
import click
import dnnlib
import numpy as np
import torch
import PIL.Image
import cv2
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import legacy


def parse_seeds(s: Union[str, List[int]]) -> List[int]:
    # If it's already a list of ints, just return it
    if isinstance(s, list):
        return s

    ranges: List[int] = []
    range_re = re.compile(r'^(\d+)-(\d+)$')

    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        m = range_re.match(part)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            ranges.extend(range(start, end + 1))
        else:
            # Single number
            ranges.append(int(part))

    return ranges


def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

@click.command()
@click.option('--network', 'network_pkl', help='Path to StyleGAN3 network pickle file', required=True)
@click.option('--outdir', help='Output directory for images', type=str, required=True)
@click.option('--class', 'class_idx', type=int, help='Class label for conditional models')
@click.option('--num-images', type=int, default=50, help='Number of unique images to generate')
@click.option('--num-impressions', type=int, default=3, help='Number of impressions per image')
@click.option('--name', type=str, required=False, help='Spoof type name (e.g., sil, gel). Runs test.py if provided.')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--seeds', type=parse_seeds, required=False, help="Seed list or ranges (e.g. '1,5,10-20')")
@click.option('--translate', type=str, default='0,0', help='XY translation for StyleGAN3 (e.g., "0.3,0.1")')
@click.option('--rotate', type=float, default=0, help='Rotation for StyleGAN3 in degrees')
@click.option('--truncation-psi', type=float, default=1.0, show_default=True, help='Truncation psi controls tradeoff between variety and fidelity')
def generate_images(network_pkl, outdir, class_idx, num_images,
                    num_impressions, name, noise_mode,
                    seeds, translate, rotate, truncation_psi):

    outdir = os.path.abspath(outdir)

    print(f'Loading network from: {network_pkl}')
    print(f'Output directory: {outdir}')

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify --class for conditional model.')
        label[:, class_idx] = 1

    seed_list = seeds if seeds else None
    translate = tuple(map(float, translate.split(',')))

    metadata_path = os.path.join(outdir, "metadata.csv")
    with open(metadata_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seed", "filename", "truncation_psi", "noise_mode"])


        generated = 0
        used_seeds = set()

        with tqdm(total=num_images, desc="Accepted images") as pbar:
            while generated < num_images:
                if seed_list:
                    if len(seed_list) == 0:
                        break
                    seed = seed_list.pop(0)
                else:
                    seed = random.randint(0, 100000)
                if seed in used_seeds:
                    continue
                used_seeds.add(seed)

                rnd = np.random.RandomState(seed)
                z = rnd.randn(1, G.z_dim)
                z_tensor = torch.from_numpy(z).to(device)

                if hasattr(G.synthesis, 'input'):
                    m = make_transform(translate, rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m).to(device))

                img = G(z_tensor, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                base_img = img[0].cpu().numpy()

                filename = f"seed{generated:04d}.png"
                img_path = os.path.join(outdir, filename)

                cv2.imwrite(
                    img_path,
                    cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
                )

                writer.writerow([
                    seed,
                    filename,
                    f"{truncation_psi:.3f}",
                    noise_mode
                ])

                generated += 1
                pbar.update(1)
    print(f"Running test_impr.py "
          f"(n_samples={num_impressions})...")

    print("Files generated by StyleGAN:")
    print(os.listdir(outdir))
    subprocess.run([
        "python",
        "/BicycleGAN/test_impr.py",

        "--dataroot", outdir,
        "--results_dir", outdir,

        "--checkpoints_dir",
        "/BicycleGAN/pretrained_models",

        "--name", "BicycleGAN_multiple_impression_weights",

        "--direction", "AtoB",

        "--load_size", "512",
        "--crop_size", "512",

        "--input_nc", "3",
        "--output_nc", "3",

        "--num_test", str(num_images),

        "--n_samples", str(num_impressions),

        "--no_flip",

        "--netG", "unet_512",
        "--netE", "resnet_512",

        "--netD", "basic_512_multi",
        "--netD2", "basic_512_multi",

        "--gpu_ids", "0"

    ], check=True,
        cwd="/BicycleGAN")
    if name and name.lower() != "live":
        print(f'Running test.py with name: {name}')
        print("Files sent to CycleGAN:")
        for f in sorted(os.listdir(outdir)):
            if f.endswith(".png"):
                print(f)
        subprocess.run(["python3", "test.py", "--dataroot", outdir, "--name", name, "--model", "test", "--crop_size", "512","--load_size", "512", "--no_dropout", "--results_dir", outdir])

if __name__ == "__main__":
    generate_images()
