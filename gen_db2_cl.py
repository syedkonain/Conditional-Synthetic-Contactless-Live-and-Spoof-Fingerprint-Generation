import os
import random
import re
import subprocess
import sys
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


DEFAULT_BICYCLEGAN_DIR = "Path of BicycleGAN"
DEFAULT_BICYCLEGAN_CHECKPOINTS_DIR = "path of BicycleGAN model weight"
DEFAULT_BICYCLEGAN_MODEL_NAME = "BicycleGAN_multiple_impression_weights"

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    """Parse comma-separated integers and inclusive ranges, e.g. ``1,5,10-20``."""

    values: List[int] = []
    range_re = re.compile(r'^(\d+)-(\d+)$')

    for part in s.split(','):
        part = part.strip()
        if not part:
            continue

        match = range_re.match(part)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start:
                raise click.BadParameter(f'Invalid descending seed range: {part}')
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    return values

#----------------------------------------------------------------------------

def run_bicyclegan(
    outdir: str,
    num_images: int,
    num_impressions: int,
    bicyclegan_dir: str,
    checkpoints_dir: str,
    model_name: str,
    gpu_ids: str,
) -> None:
    """Generate multiple impressions for the StyleGAN images using BicycleGAN."""

    if num_impressions == 0:
        print('Skipping BicycleGAN because --num-impressions=0.')
        return

    bicyclegan_dir = os.path.abspath(bicyclegan_dir)
    checkpoints_dir = os.path.abspath(checkpoints_dir)
    test_script = os.path.join(bicyclegan_dir, 'test_impr.py')

    if not os.path.isfile(test_script):
        raise click.ClickException(
            f'BicycleGAN test script was not found: {test_script}. '
            'Set the correct path with --bicyclegan-dir.'
        )
    if not os.path.isdir(checkpoints_dir):
        raise click.ClickException(
            f'BicycleGAN checkpoints directory was not found: {checkpoints_dir}. '
            'Set the correct path with --bicyclegan-checkpoints-dir.'
        )

    print(f'Running BicycleGAN test_impr.py with {num_impressions} impressions per image...')
    command = [
        sys.executable,
        test_script,
        '--dataroot', outdir,
        '--results_dir', outdir,
        '--checkpoints_dir', checkpoints_dir,
        '--name', model_name,
        '--direction', 'AtoB',
        '--load_size', '512',
        '--crop_size', '512',
        '--input_nc', '3',
        '--output_nc', '3',
        '--num_test', str(num_images),
        '--n_samples', str(num_impressions),
        '--no_flip',
        '--netG', 'unet_512',
        '--netE', 'resnet_512',
        '--netD', 'basic_512_multi',
        '--netD2', 'basic_512_multi',
        '--gpu_ids', gpu_ids,
    ]

    try:
        subprocess.run(command, check=True, cwd=bicyclegan_dir)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f'BicycleGAN failed with exit code {exc.returncode}.'
        ) from exc

#----------------------------------------------------------------------------

def run_cyclegan_spoof(
    outdir: str,
    spoof_name: Optional[str],
    cyclegan_dir: str,
) -> None:
    """Run CycleGAN spoof generation when a non-live spoof name is supplied."""

    if not spoof_name:
        print('Skipping CycleGAN spoof generation because --name was not provided.')
        return
    if spoof_name.lower() == 'live':
        print('Skipping CycleGAN spoof generation because --name=live.')
        return

    cyclegan_dir = os.path.abspath(cyclegan_dir)
    test_script = os.path.join(cyclegan_dir, 'test.py')
    if not os.path.isfile(test_script):
        raise click.ClickException(
            f'CycleGAN test script was not found: {test_script}. '
            'Set the correct path with --cyclegan-dir.'
        )

    print(f'Running CycleGAN spoof generation with model name: {spoof_name}')
    command = [
        sys.executable,
        test_script,
        '--dataroot', outdir,
        '--name', spoof_name,
        '--model', 'test',
        '--crop_size', '512',
        '--load_size', '512',
        '--no_dropout',
        '--results_dir', outdir,
    ]

    try:
        subprocess.run(command, check=True, cwd=cyclegan_dir)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f'CycleGAN spoof generation failed with exit code {exc.returncode}.'
        ) from exc

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='Optional seed list, e.g. 1,5,10-20')
@click.option('--num-images', type=click.IntRange(min=1), default=50, show_default=True,
              help='Number of unique StyleGAN images to generate; with --seeds, uses up to this many seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num-impressions', type=click.IntRange(min=0), default=3, show_default=True,
              help='Number of BicycleGAN impressions per generated image; use 0 to skip BicycleGAN')
@click.option('--name', 'spoof_name', type=str,
              help='CycleGAN spoof model name, e.g. sil or gel; live/omitted skips spoof generation')
@click.option('--bicyclegan-dir', type=click.Path(file_okay=False),
              default=DEFAULT_BICYCLEGAN_DIR, show_default=True,
              help='Directory containing BicycleGAN test_impr.py')
@click.option('--bicyclegan-checkpoints-dir', type=click.Path(file_okay=False),
              default=DEFAULT_BICYCLEGAN_CHECKPOINTS_DIR, show_default=True,
              help='BicycleGAN checkpoints directory')
@click.option('--bicyclegan-model-name', type=str,
              default=DEFAULT_BICYCLEGAN_MODEL_NAME, show_default=True,
              help='BicycleGAN checkpoint/model name')
@click.option('--cyclegan-dir', type=click.Path(file_okay=False), default='.', show_default=True,
              help='Directory containing CycleGAN test.py')
@click.option('--gpu-ids', type=str, default='0', show_default=True,
              help='GPU IDs passed to BicycleGAN')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    num_images: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    num_impressions: int,
    spoof_name: Optional[str],
    bicyclegan_dir: str,
    bicyclegan_checkpoints_dir: str,
    bicyclegan_model_name: str,
    cyclegan_dir: str,
    gpu_ids: str,
) -> None:

    outdir = os.path.abspath(outdir)
    print('Loading networks from "%s"...' % network_pkl)
    print(f'Output directory: {outdir}')

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    generated_count = 0

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)

        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            output_path = os.path.join(outdir, f'proj{idx:02d}.png')
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_path)
            generated_count += 1
    else:
        # Match gen_db3_cl.py's --num-images behavior. When --seeds is omitted,
        # create the requested number of unique random seeds. When it is supplied,
        # consume at most --num-images unique entries from that list.
        selected_seeds: List[int] = []
        used_seeds = set()

        if seeds is None:
            while len(selected_seeds) < num_images:
                seed = random.randint(0, 100000)
                if seed in used_seeds:
                    continue
                used_seeds.add(seed)
                selected_seeds.append(seed)
            print(f'No --seeds provided; generated {len(selected_seeds)} unique random seeds.')
        else:
            for seed in seeds:
                if seed in used_seeds:
                    print(f'warn: duplicate seed {seed} was skipped')
                    continue
                used_seeds.add(seed)
                selected_seeds.append(seed)
                if len(selected_seeds) >= num_images:
                    break

            if not selected_seeds:
                ctx.fail('The supplied --seeds value did not contain any usable seeds')
            if len(selected_seeds) < num_images:
                print(
                    f'warn: requested {num_images} image(s), but only '
                    f'{len(selected_seeds)} unique seed(s) were supplied'
                )
            elif len(seeds) > num_images:
                print(f'Using the first {num_images} unique seed(s) from --seeds.')

        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            if class_idx < 0 or class_idx >= G.c_dim:
                ctx.fail(f'--class must be between 0 and {G.c_dim - 1}')
            label[:, class_idx] = 1
        elif class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        for seed_idx, seed in enumerate(selected_seeds):
            print(
                'Generating image for seed %d (%d/%d) ...'
                % (seed, seed_idx + 1, len(selected_seeds))
            )
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            output_path = os.path.join(outdir, f'seed{seed:04d}.png')
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_path)
            generated_count += 1

    if generated_count == 0:
        raise click.ClickException('No StyleGAN images were generated; post-processing was not started.')

    print(f'StyleGAN2-ADA generated {generated_count} image(s).')
    run_bicyclegan(
        outdir=outdir,
        num_images=generated_count,
        num_impressions=num_impressions,
        bicyclegan_dir=bicyclegan_dir,
        checkpoints_dir=bicyclegan_checkpoints_dir,
        model_name=bicyclegan_model_name,
        gpu_ids=gpu_ids,
    )
    run_cyclegan_spoof(
        outdir=outdir,
        spoof_name=spoof_name,
        cyclegan_dir=cyclegan_dir,
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
