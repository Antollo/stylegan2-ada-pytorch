"""Qualitative analysis: EMA vs Active generator comparison.
Generates image pairs from G (EMA) and G_base (active generator) using
identical latents, ranks them by SD Loss (LPIPS), then uses the
discriminator to select the most/least realistic images from the
most similar and most dissimilar pairs.  Saves 40 diagnostic images.
"""

import os
import copy
import numpy as np
import torch
import PIL.Image
import lpips
from scipy import stats

import dnnlib
from . import metric_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mbstd_group_size(D):
    """Extract mbstd_group_size from the discriminator, defaulting to 1."""
    for module in D.modules():
        if hasattr(module, 'group_size'):
            return module.group_size
    return 1


def _run_D_safe(D, images, c, mbstd_gs):
    """Run discriminator, padding the batch to a multiple of mbstd_gs."""
    B = images.shape[0]
    remainder = B % mbstd_gs
    if remainder != 0:
        pad_size = mbstd_gs - remainder
        images = torch.cat([images, images[:pad_size]], dim=0)
        c = torch.cat([c, c[:pad_size]], dim=0)
    logits = D(images, c)
    return logits[:B]


def _tensor_to_pil(img_tensor):
    """Convert a single image tensor [-1, 1] → PIL.Image (RGB uint8)."""
    img = (img_tensor.clamp(-1, 1) + 1) * 127.5
    img = img.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return PIL.Image.fromarray(img, 'RGB')

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_ema_qualitative(
    opts,
    num_pairs=1000,
    num_similar=100,
    num_dissimilar=100,
    num_top=10,
    batch_size=8,
    checkpoint_name='ckpt',
    output_dir=None,
):
    """Generate and save qualitative comparison images.

    Args:
        opts:             MetricOptions (must have G, G_base, D).
        num_pairs:        Total image pairs to generate (default 1000).
        num_similar:      How many lowest-SD-loss pairs to keep (default 100).
        num_dissimilar:   How many highest-SD-loss pairs to keep (default 100).
        num_top:          How many most-real / most-fake to save (default 10).
        batch_size:       Generation batch size.
        checkpoint_name:  Prefix for saved filenames.
        output_dir:       Directory to save images.  If None, uses run_dir or cwd.

    Returns:
        dict with summary statistics (mean/std of SD losses, etc.).
    """
    # --- Only rank 0 does real work ---
    if opts.rank != 0:
        return dict(
            sd_loss_mean=float('nan'),
            sd_loss_std=float('nan'),
            d_score_mean=float('nan'),
            d_score_std=float('nan'),
            spearman_corr=float('nan'),
            spearman_pvalue=float('nan'),
            pearson_corr=float('nan'),
            pearson_pvalue=float('nan'),
            num_images_saved=float('nan'),
        )

    assert opts.G_base is not None, 'G_base (active generator) must be provided'
    assert opts.D is not None, 'D (discriminator) must be provided'

    device = opts.device

    # Prepare models
    G_ema = copy.deepcopy(opts.G).eval().requires_grad_(False).to(device)
    G_base = copy.deepcopy(opts.G_base).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(device)
    mbstd_gs = _get_mbstd_group_size(D)

    # Prepare LPIPS (SD Loss)
    lpips_fn = lpips.LPIPS(net='vgg').eval().to(device)

    # Prepare dataset for labels
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'ema_qualitative')
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Generate pairs and compute per-sample SD Loss
    # ------------------------------------------------------------------
    print(f'  Generating {num_pairs} image pairs ...')

    all_sd_losses = []      # list of 1-D tensors
    all_imgs_ema = []        # list of image tensors  [C, H, W]
    all_imgs_base = []
    all_labels = []          # list of label tensors  [c_dim]

    generated = 0
    while generated < num_pairs:
        cur_batch = min(batch_size, num_pairs - generated)

        z = torch.randn([cur_batch, G_ema.z_dim], device=device)
        c = [dataset.get_label(np.random.randint(len(dataset))) for _ in range(cur_batch)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(device)

        with torch.no_grad():
            img_ema = G_ema(z=z, c=c, **opts.G_kwargs)    # [-1, 1]
            img_base = G_base(z=z, c=c, **opts.G_kwargs)  # [-1, 1]

            # Per-sample LPIPS (returns [B, 1, 1, 1])
            sd = lpips_fn(img_ema, img_base)  # [B, 1, 1, 1]
            sd = sd.view(-1)                   # [B]

        all_sd_losses.append(sd.cpu())
        all_imgs_ema.append(img_ema.cpu())
        all_imgs_base.append(img_base.cpu())
        all_labels.append(c.cpu())

        generated += cur_batch
        if generated % 100 == 0 or generated >= num_pairs:
            print(f'    {generated}/{num_pairs}')

    # Concatenate
    all_sd_losses = torch.cat(all_sd_losses, dim=0)       # [N]
    all_imgs_ema = torch.cat(all_imgs_ema, dim=0)          # [N, C, H, W]
    all_imgs_base = torch.cat(all_imgs_base, dim=0)
    all_labels = torch.cat(all_labels, dim=0)               # [N, c_dim]

    sd_mean = float(all_sd_losses.mean())
    sd_std = float(all_sd_losses.std())
    print(f'  SD Loss — mean: {sd_mean:.4f}, std: {sd_std:.4f}')

    # ------------------------------------------------------------------
    # Phase 2: Score ALL generations with discriminator
    # ------------------------------------------------------------------
    print(f'  Scoring all {num_pairs} images with D ...')
    all_d_scores = []
    for start in range(0, all_imgs_ema.shape[0], batch_size):
        end = min(start + batch_size, all_imgs_ema.shape[0])
        batch_imgs = all_imgs_ema[start:end].to(device)
        batch_c = all_labels[start:end].to(device)
        if G_ema.c_dim == 0:
            batch_c = torch.zeros([batch_imgs.shape[0], 0], device=device)
        with torch.no_grad():
            logits = _run_D_safe(D, batch_imgs, batch_c, mbstd_gs)
        all_d_scores.append(logits.cpu().view(-1))
        if (end % 1000 == 0) or end >= all_imgs_ema.shape[0]:
            print(f'    {end}/{all_imgs_ema.shape[0]}')
    all_d_scores = torch.cat(all_d_scores, dim=0)  # [N]

    d_mean = float(all_d_scores.mean())
    d_std = float(all_d_scores.std())
    print(f'  D scores — mean: {d_mean:.4f}, std: {d_std:.4f}')

    # ------------------------------------------------------------------
    # Phase 3: Spearman correlation between SD Loss and D scores
    # ------------------------------------------------------------------
    sd_np = all_sd_losses.numpy()
    d_np = all_d_scores.numpy()
    spearman_corr, spearman_pvalue = stats.spearmanr(sd_np, d_np)
    spearman_corr = float(spearman_corr)
    spearman_pvalue = float(spearman_pvalue)
    pearson_corr, pearson_pvalue = stats.pearsonr(sd_np, d_np)
    pearson_corr = float(pearson_corr)
    pearson_pvalue = float(pearson_pvalue)
    print(f'  Spearman correlation (SD vs D): r={spearman_corr:.4f}, p={spearman_pvalue:.2e}')
    print(f'  Pearson  correlation (SD vs D): r={pearson_corr:.4f}, p={pearson_pvalue:.2e}')

    # ------------------------------------------------------------------
    # Phase 4: Primary filtering — most similar / most dissimilar
    # ------------------------------------------------------------------
    sorted_indices = torch.argsort(all_sd_losses)
    similar_idx = sorted_indices[:num_similar]          # lowest SD loss
    dissimilar_idx = sorted_indices[-num_dissimilar:]   # highest SD loss

    # ------------------------------------------------------------------
    # Phase 5: Secondary filtering — pick top real/fake from each group
    # ------------------------------------------------------------------
    def _pick_top_bottom(group_idx, k):
        """From group_idx, pick k most-real and k most-fake by D score."""
        group_d = all_d_scores[group_idx]
        sorted_order = torch.argsort(group_d)
        fake_local = sorted_order[:k]       # lowest logits → most fake
        real_local = sorted_order[-k:]       # highest logits → most real
        real_local = real_local.flip(0)      # most-real first
        return group_idx[real_local], group_d[real_local], group_idx[fake_local], group_d[fake_local]

    sim_real_idx, sim_real_logits, sim_fake_idx, sim_fake_logits = _pick_top_bottom(similar_idx, num_top)
    dis_real_idx, dis_real_logits, dis_fake_idx, dis_fake_logits = _pick_top_bottom(dissimilar_idx, num_top)

    # ------------------------------------------------------------------
    # Phase 6: Save images
    # ------------------------------------------------------------------
    def _save_group(indices, logits, category):
        """Save images for a group and return count."""
        count = 0
        for i in range(len(indices)):
            idx = int(indices[i])
            sd_val = float(all_sd_losses[idx])
            d_val = float(logits[i])
            fname = f'{checkpoint_name}_{category}_SD{sd_val:.4f}_D{d_val:.4f}.png'
            fpath = os.path.join(output_dir, fname)
            pil_img = _tensor_to_pil(all_imgs_ema[idx])
            pil_img.save(fpath)
            count += 1
        return count

    print(f'  Saving images to {output_dir} ...')
    n = 0
    n += _save_group(sim_real_idx, sim_real_logits, 'similar_real')
    n += _save_group(sim_fake_idx, sim_fake_logits, 'similar_fake')
    n += _save_group(dis_real_idx, dis_real_logits, 'dissimilar_real')
    n += _save_group(dis_fake_idx, dis_fake_logits, 'dissimilar_fake')
    print(f'  Saved {n} images.')

    return dict(
        sd_loss_mean=sd_mean,
        sd_loss_std=sd_std,
        d_score_mean=d_mean,
        d_score_std=d_std,
        spearman_corr=spearman_corr,
        spearman_pvalue=spearman_pvalue,
        pearson_corr=pearson_corr,
        pearson_pvalue=pearson_pvalue,
        num_images_saved=float(n),
    )

# ---------------------------------------------------------------------------
