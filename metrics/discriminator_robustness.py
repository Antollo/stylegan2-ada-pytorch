import copy
import numpy as np
import torch
import torch.nn as nn
import kornia.filters as KF
import dnnlib

from . import metric_utils


def _get_mbstd_group_size(D):
    """Extract mbstd_group_size from the discriminator, defaulting to 1."""
    for module in D.modules():
        if hasattr(module, 'group_size'):
            return module.group_size
    return 1


def _run_D_safe(D, images, c, mbstd_gs):
    """Run discriminator, padding the batch to a multiple of mbstd_gs if needed."""
    B = images.shape[0]
    remainder = B % mbstd_gs
    if remainder != 0:
        pad_size = mbstd_gs - remainder
        images = torch.cat([images, images[:pad_size]], dim=0)
        c = torch.cat([c, c[:pad_size]], dim=0)
    logits = D(images, c)
    return logits[:B]

# ---------------------------------------------------------------------------

class FeatureSuppressionEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

    def suppress_texture_gaussian(self, x, kernel_size=(13, 13), sigma=(1.5, 1.5)):
        """Low-pass filter: destroys texture, blurs shape."""
        return KF.gaussian_blur2d(x, kernel_size, sigma)

    def suppress_texture_bilateral(self, x, kernel_size=(13, 13), sigma_color=0.5, sigma_space=(8.0, 8.0)):
        """Edge-preserving low-pass filter: destroys texture, PRESERVES shape."""
        return KF.bilateral_blur(x, kernel_size, sigma_color, sigma_space)

    def suppress_shape_patch_shuffle(self, x, grid_size=2):
        """Spatial disruption: destroys global/local shape, PRESERVES texture."""
        B, C, H, W = x.shape

        patch_h = H // grid_size
        patch_w = W // grid_size

        reshaped = x.view(B, C, grid_size, patch_h, grid_size, patch_w)
        reshaped = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
        num_patches = grid_size * grid_size
        patches = reshaped.view(B, C, num_patches, patch_h, patch_w)

        shuffled = torch.zeros_like(patches)
        for b in range(B):
            idx = torch.randperm(num_patches, device=x.device)
            shuffled[b] = patches[b, :, idx, :, :]

        shuffled = shuffled.view(B, C, grid_size, grid_size, patch_h, patch_w)
        shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
        output = shuffled.view(B, C, H, W)

        return output

# ---------------------------------------------------------------------------

def _collect_logits_real(D, opts, num_items, batch_size, perturbation_fn=None):
    """Collect discriminator logits on real dataset images.

    Dataset images arrive as uint8 [0, 255].  We convert to float32 [-1, 1]
    for the discriminator.  Perturbation functions receive [0, 1] images.
    """
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    num_items = min(num_items, len(dataset))
    mbstd_gs = _get_mbstd_group_size(D)

    item_subset = [(i * opts.num_gpus + opts.rank) % num_items
                   for i in range((num_items - 1) // opts.num_gpus + 1)]

    all_logits = []
    count = 0
    for images, labels in torch.utils.data.DataLoader(
        dataset=dataset, sampler=item_subset, batch_size=batch_size,
        pin_memory=True, num_workers=3, prefetch_factor=2,
    ):
        if count >= num_items:
            break
        # uint8 [0, 255] → float [0, 1]
        images = images.to(opts.device).to(torch.float32) / 255.0

        if perturbation_fn is not None:
            images = perturbation_fn(images)

        # [0, 1] → [-1, 1] for discriminator
        images = images * 2.0 - 1.0

        c = labels.to(opts.device) if opts.G.c_dim > 0 else torch.zeros([images.shape[0], 0], device=opts.device)
        with torch.no_grad():
            logits = _run_D_safe(D, images, c, mbstd_gs)
        all_logits.append(logits.cpu())
        count += images.shape[0]

    return torch.cat(all_logits, dim=0)[:num_items]

# ---------------------------------------------------------------------------

def _collect_logits_fake(D, G, opts, num_items, batch_size, batch_gen=None, perturbation_fn=None):
    """Collect discriminator logits on generated images.

    Generator output is float [-1, 1].  Perturbation functions receive [0, 1].
    """
    if batch_gen is None:
        batch_gen = batch_size
    assert batch_size % batch_gen == 0

    G = copy.deepcopy(G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    all_logits = []
    count = 0
    while count < num_items:
        images = []
        c_list = []
        for _ in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            c = [dataset.get_label(np.random.randint(len(dataset))) for _ in range(batch_gen)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            img = G(z=z, c=c, **opts.G_kwargs)  # float, roughly [-1,1]
            images.append(img)
            c_list.append(c)
        images = torch.cat(images, dim=0)
        c_batch = torch.cat(c_list, dim=0)

        if perturbation_fn is not None:
            # [-1, 1] → [0, 1] for perturbation
            images = (images + 1) / 2.0
            images = perturbation_fn(images)
            # [0, 1] → [-1, 1]
            images = images * 2.0 - 1.0

        if G.c_dim == 0:
            c_batch = torch.zeros([images.shape[0], 0], device=opts.device)

        with torch.no_grad():
            logits = _run_D_safe(D, images, c_batch, _get_mbstd_group_size(D))
        all_logits.append(logits.cpu())
        count += images.shape[0]

    return torch.cat(all_logits, dim=0)[:num_items]

# ---------------------------------------------------------------------------

def _calibrate_threshold(real_logits, fake_logits):
    """Find the threshold that maximizes binary classification accuracy.

    Convention: logit > threshold → predicted real.
    """
    all_logits = torch.cat([real_logits.view(-1), fake_logits.view(-1)], dim=0).numpy()
    all_labels = np.concatenate([
        np.ones(real_logits.shape[0]),
        np.zeros(fake_logits.shape[0]),
    ])

    # Sweep thresholds at each unique logit value
    sorted_logits = np.sort(np.unique(all_logits))
    # Add sentinels
    candidates = np.concatenate([[sorted_logits[0] - 1], sorted_logits, [sorted_logits[-1] + 1]])

    best_acc = 0.0
    best_thr = 0.0
    for thr in candidates:
        preds = (all_logits > thr).astype(np.float32)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return float(best_thr), float(best_acc)


def _compute_accuracy(real_logits, fake_logits, threshold):
    """Compute accuracy at the given threshold."""
    all_logits = torch.cat([real_logits.view(-1), fake_logits.view(-1)], dim=0).numpy()
    all_labels = np.concatenate([
        np.ones(real_logits.shape[0]),
        np.zeros(fake_logits.shape[0]),
    ])
    preds = (all_logits > threshold).astype(np.float32)
    return float((preds == all_labels).mean())

# ---------------------------------------------------------------------------

def compute_discriminator_robustness(opts, num_real=10000, num_fake=10000, batch_size=64):
    """Evaluate discriminator robustness to bilateral filtering & patch shuffling.

    Returns dict with:
      - clean_acc:      accuracy on unperturbed data (with calibrated threshold)
      - bilateral_acc:  accuracy after bilateral filtering
      - bilateral_drop: absolute accuracy drop (clean − bilateral)
      - patchshuffle_acc:  accuracy after patch shuffling
      - patchshuffle_drop: absolute accuracy drop (clean − patchshuffle)
    """
    if opts.rank != 0:
        return dict(
            clean_acc=float('nan'),
            bilateral_acc=float('nan'), bilateral_drop=float('nan'),
            patchshuffle_acc=float('nan'), patchshuffle_drop=float('nan'),
        )


    assert opts.D is not None, 'Discriminator (D) must be provided in opts'

    D = opts.D.to(opts.device)
    evaluator = FeatureSuppressionEvaluator().to(opts.device)

    # ---- Phase 1: Clean data ----
    print('  Collecting clean real logits ...')
    real_logits_clean = _collect_logits_real(D, opts, num_real, batch_size)

    print('  Collecting clean fake logits ...')
    fake_logits_clean = _collect_logits_fake(D, opts.G_base, opts, num_fake, batch_size)

    threshold, clean_acc = _calibrate_threshold(real_logits_clean, fake_logits_clean)
    print(f'  Calibrated threshold = {threshold:.4f}, clean accuracy = {clean_acc:.4f}')

    # ---- Phase 2: Bilateral filtering ----
    print('  Collecting bilateral-filtered real logits ...')
    real_logits_bilateral = _collect_logits_real(
        D, opts, num_real, batch_size,
        perturbation_fn=evaluator.suppress_texture_bilateral,
    )
    print('  Collecting bilateral-filtered fake logits ...')
    fake_logits_bilateral = _collect_logits_fake(
        D, opts.G_base, opts, num_fake, batch_size,
        perturbation_fn=evaluator.suppress_texture_bilateral,
    )
    _, bilateral_acc = _calibrate_threshold(real_logits_bilateral, fake_logits_bilateral)

    print('  Collecting gaussian-filtered real logits ...')
    real_logits_gaussian = _collect_logits_real(
        D, opts, num_real, batch_size,
        perturbation_fn=evaluator.suppress_texture_gaussian,
    )
    print('  Collecting gaussian-filtered fake logits ...')
    fake_logits_gaussian = _collect_logits_fake(
        D, opts.G_base, opts, num_fake, batch_size,
        perturbation_fn=evaluator.suppress_texture_gaussian,
    )
    _, gaussian_acc = _calibrate_threshold(real_logits_gaussian, fake_logits_gaussian)

    # ---- Phase 3: Patch shuffling ----
    print('  Collecting patch-shuffled real logits ...')
    real_logits_patchshuffle = _collect_logits_real(
        D, opts, num_real, batch_size,
        perturbation_fn=evaluator.suppress_shape_patch_shuffle,
    )
    print('  Collecting patch-shuffled fake logits ...')
    fake_logits_patchshuffle = _collect_logits_fake(
        D, opts.G_base, opts, num_fake, batch_size,
        perturbation_fn=evaluator.suppress_shape_patch_shuffle,
    )
    _, patchshuffle_acc = _calibrate_threshold(real_logits_patchshuffle, fake_logits_patchshuffle)

    return dict(
        clean_acc=clean_acc,
        bilateral_acc=bilateral_acc,
        gaussian_acc=gaussian_acc,
        patchshuffle_acc=patchshuffle_acc,
    )

# ---------------------------------------------------------------------------
