import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing scheduler with warm restarts and warmup."""
    
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1,
                 min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates."""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * 
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                 (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8,
                 eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                    self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                                 (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]

def get_scheduler(training_config, optimizer, steps_per_epoch):
    """Get learning rate scheduler based on configuration.
    
    Args:
        training_config: Training configuration
        optimizer: PyTorch optimizer
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Learning rate scheduler
    """
    scheduler_name = training_config.scheduler.lower()
    
    if scheduler_name == 'cosine_annealing':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(training_config.warmup_epochs),
            max_epochs=int(training_config.num_epochs),
            warmup_start_lr=float(training_config.warmup_lr),
            eta_min=float(training_config.min_lr)
        )
    elif scheduler_name == 'cosine_annealing_restarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(training_config.num_epochs) // 4,
            cycle_mult=1.0,
            max_lr=float(training_config.learning_rate),
            min_lr=float(training_config.min_lr),
            warmup_steps=int(training_config.warmup_epochs),
            gamma=0.9
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_config.num_epochs) // 3,
            gamma=0.1
        )
    elif scheduler_name == 'multistep':
        milestones = [int(training_config.num_epochs) // 3, 2 * int(training_config.num_epochs) // 3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
    elif scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler