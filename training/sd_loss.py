import torch
import lpips
from kornia import augmentation
from kornia.constants import SamplePadding
from pytorch_msssim import ms_ssim

class SDLoss(torch.nn.Module):
    def __init__(self, device, use_aug=True, loss_type='lpips'):
        super().__init__()
        self.device = device
        self.use_aug = use_aug
        self.loss_type = loss_type
        
        if self.loss_type == "lpips":
            self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        elif self.loss_type == "msssim":
            pass
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
            
        if self.loss_type == "lpips":
            loss = self.lpips_fn(gen_img, gen_img_ema).mean()
        elif self.loss_type == "msssim":
            win_size = 7 if min(*gen_img.shape[2:]) < 160 else 11
            gen_img = (gen_img + 1) / 2
            gen_img_ema = (gen_img_ema + 1) / 2
            loss = 1 - ms_ssim(gen_img, gen_img_ema, data_range=1, size_average=True, win_size=win_size)
        return loss