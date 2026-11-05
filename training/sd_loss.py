import torch
import torch.nn.functional as F
import lpips
from kornia import augmentation
from kornia.constants import SamplePadding
from pytorch_msssim import ms_ssim


class DiscriminatorPerceptualLoss(torch.nn.Module):
    """LPIPS-style perceptual loss using the first N blocks of the discriminator.

    Extracts intermediate feature maps from DiscriminatorBlocks, channel-normalizes
    them, and computes a per-layer L1 distance (averaged spatially, summed across layers).
    All D parameters are frozen during the forward pass — gradients flow to G only.
    """

    def __init__(self, D, num_layers=3):
        super().__init__()
        # Unwrap DistributedDataParallel if present, so we can access
        # block_resolutions and the b{res} sub-modules directly.
        self.D = D.module if hasattr(D, 'module') else D
        self.num_layers = num_layers

    def _extract_features(self, img):
        """Run img through the first `num_layers` D blocks, collecting features."""
        features = []
        x = None
        block_resolutions = self.D.block_resolutions

        for i, res in enumerate(block_resolutions):
            if i >= self.num_layers:
                break
            block = getattr(self.D, f'b{res}')
            x, img = block(x, img)
            features.append(x.to(torch.float32))

        return features

    @staticmethod
    def _channel_normalize(feat, eps=1e-10):
        """Unit-normalize each feature map along the channel dimension (like LPIPS)."""
        return F.normalize(feat, p=2, dim=1, eps=eps)

    def forward(self, img_a, img_b):
        # img_a is from G (needs gradients), img_b is from G_ema (no gradients needed).
        # D's parameters already have requires_grad=False during G's training phase,
        # so gradients flow through D back to G but don't update D.
        feats_a = self._extract_features(img_a)
        with torch.no_grad():
            feats_b = self._extract_features(img_b)

        # Per-layer normalized distance
        loss = 0.0
        for fa, fb in zip(feats_a, feats_b):
            fa = self._channel_normalize(fa)
            fb = self._channel_normalize(fb)
            loss = loss + F.mse_loss(fa, fb)

        loss = loss / self.num_layers

        return loss


class SDLoss(torch.nn.Module):
    def __init__(self, device, use_aug=True, loss_type='lpips', D=None, dlpips_layers=3):
        super().__init__()
        self.device = device
        self.use_aug = use_aug
        self.loss_type = loss_type
        
        if self.loss_type == "lpips":
            self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        elif self.loss_type == "msssim":
            pass
        elif self.loss_type in ["l1", "l2"]:
            pass
        elif self.loss_type == "dlpips":
            if D is None:
                raise ValueError("D (discriminator) must be provided for loss_type='dlpips'")
            self.dlpips_fn = DiscriminatorPerceptualLoss(D, num_layers=dlpips_layers)
        else:
            raise ValueError(f"Unknown sd_loss_type: {self.loss_type}")
            
        if self.use_aug:
            self.aug = augmentation.AugmentationSequential(
                augmentation.RandomHorizontalFlip(p=0.5),
                augmentation.RandomAffine(degrees=5, translate=None, p=0.5, padding_mode=SamplePadding.REFLECTION),
                augmentation.RandomAffine(degrees=0, translate=(0.01, 0.01), p=0.5, padding_mode=SamplePadding.REFLECTION),
                random_apply=2,
                data_keys=["input", "input"]
            ).to(device)

    def forward(self, gen_img, gen_img_ema):
        if self.use_aug:
            gen_img, gen_img_ema = self.aug(gen_img, gen_img_ema)
        gen_img = torch.clamp(gen_img, min=-2.0, max=2.0)
        gen_img_ema = torch.clamp(gen_img_ema, min=-2.0, max=2.0)
            
        if self.loss_type == "lpips":
            loss = self.lpips_fn(gen_img, gen_img_ema).mean()
        elif self.loss_type == "msssim":
            win_size = 7 if min(*gen_img.shape[2:]) < 160 else 11
            gen_img_msssim = torch.clamp((gen_img + 1) / 2, 0.0, 1.0)
            gen_img_ema_msssim = torch.clamp((gen_img_ema + 1) / 2, 0.0, 1.0)
            loss = 1 - ms_ssim(gen_img_msssim, gen_img_ema_msssim, data_range=1, size_average=True, win_size=win_size)
        elif self.loss_type == "l1":
            loss = F.l1_loss(gen_img, gen_img_ema)
        elif self.loss_type == "l2":
            loss = F.mse_loss(gen_img, gen_img_ema)
        elif self.loss_type == "dlpips":
            loss = self.dlpips_fn(gen_img, gen_img_ema)
        return loss