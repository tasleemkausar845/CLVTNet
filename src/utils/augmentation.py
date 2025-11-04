import torch
import numpy as np
import random
from typing import Tuple

class MixUpCutMixAugmentation:
    """Combined MixUp and CutMix augmentation for EEG topographic maps."""

    def __init__(self, aug_config):
        """
        Args:
            aug_config: AugmentationConfig object
        """
        self.use_mixup = aug_config.use_mixup
        self.use_cutmix = aug_config.use_cutmix
        self.mixup_alpha = aug_config.mixup_alpha
        self.cutmix_alpha = aug_config.cutmix_alpha
        self.mixup_prob = aug_config.mixup_prob
        self.cutmix_prob = aug_config.cutmix_prob

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply augmentation to batch."""
        # Check which augmentation to apply
        if self.use_mixup and self.use_cutmix:
            if random.random() < 0.5 and random.random() < self.mixup_prob:
                return self.mixup(x, y)
            elif random.random() < self.cutmix_prob:
                return self.cutmix(x, y)
        elif self.use_mixup and random.random() < self.mixup_prob:
            return self.mixup(x, y)
        elif self.use_cutmix and random.random() < self.cutmix_prob:
            return self.cutmix(x, y)

        return x, y, y, 1.0  # No augmentation

    def mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """MixUp augmentation."""
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0

        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """CutMix augmentation."""
        batch_size = x.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1.0

        index = torch.randperm(batch_size).to(x.device)
        _, _, H, W = x.size()
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam