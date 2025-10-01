import torch
import numpy as np
import random
from typing import Tuple

class MixUpCutMixAugmentation:
    """Combined MixUp and CutMix augmentation for EEG data."""
    
    def __init__(self, config):
        self.config = config
        self.use_mixup = config.use_mixup
        self.use_cutmix = config.use_cutmix
        self.mixup_alpha = config.mixup_alpha
        self.cutmix_alpha = config.cutmix_alpha
        self.augmentation_prob = config.augmentation_prob
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply augmentation to batch.
        
        Args:
            x: Input batch (B, C, H, W)
            y: Target labels (B,)
            
        Returns:
            Tuple of (augmented_x, y_a, y_b, lambda)
        """
        if random.random() > self.augmentation_prob:
            return x, y, y, 1.0
        
        if self.use_mixup and self.use_cutmix:
            if random.random() < 0.5:
                return self.mixup(x, y)
            else:
                return self.cutmix(x, y)
        elif self.use_mixup:
            return self.mixup(x, y)
        elif self.use_cutmix:
            return self.cutmix(x, y)
        else:
            return x, y, y, 1.0
    
    def mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        batch_size = x.size(0)
        
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        batch_size = x.size(0)
        
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0
        
        index = torch.randperm(batch_size).to(x.device)
        
        _, _, H, W = x.size()
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam