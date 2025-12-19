from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Data processing configuration aligned with KUL/DTU dataset loaders."""
    
    # EEG processing
    sampling_rate: int = 128
    window_length: float = 2.0
    overlap: float = 0.5
    
    use_ssf_extraction: bool = True
    use_multiband_ssf: bool = False
    
    # Dataset selection
    dataset_type: str = 'KUL'  # 'KUL' or 'DTU'
    dataset_path: str = '/path/to/dataset'
    

    # EEG topomap settings
    
    image_size: int = 256
    interpolation_method: str = "cubic"
    normalization: str = "z_score"
    
    # Channels — loader will verify and warn
    n_channels: int = 64
    electrode_type: str = "standard"
    
    # Supported file formats for scanning dataset
    file_formats: List[str] = field(default_factory=lambda: ['.edf', '.fif', '.set', '.csv', '.txt'])


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    input_channels: int = 1
    num_classes: int = 2
    
    embed_dims: List[int] = field(default_factory=lambda: [32, 80, 160, 256])
    num_heads: List[int] = field(default_factory=lambda: [2, 5, 10, 16])
    num_blocks: List[int] = field(default_factory=lambda: [2, 2, 5, 2])
    filter_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    
    dropout_rate: float = 0.1
    drop_path_rate: float = 0.1
    
    use_checkpoint: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    batch_size: int = 64
    num_epochs: int = 300
    
    learning_rate: float = 0.0003
    weight_decay: float = 0.05
    warmup_epochs: int = 20
    
    optimizer: str = "adam"
    scheduler: str = "cosine_annealing"
    
    gradient_clip: float = 1.0
    
    early_stopping_patience: int = 20
    early_stopping_delta: float = 0.001
    
    use_amp: bool = False
    
    label_smoothing: float = 0.0


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    
    use_augmentation: bool = True
    
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    
    use_mixup: bool = True
    mixup_alpha: float = 1.0
    mixup_prob: float = 0.5
    
    use_random_erasing: bool = True
    random_erasing_prob: float = 0.25
    
    use_rand_augment: bool = False
    rand_augment_n: int = 2
    rand_augment_m: int = 10


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""
    
    name: str = "CLVTNet_EEG_Classification"
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    save_best_only: bool = True
    
    log_frequency: int = 10
    eval_frequency: int = 1
    
    use_wandb: bool = False
    wandb_project: str = "clvtnet"
    wandb_entity: Optional[str] = None
    
    deterministic: bool = True
    benchmark: bool = False


@dataclass
class Config:
    """Complete configuration for training & data pipeline."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def validate(self):
        """Validate configuration parameters."""
        if self.model.input_channels != len(self.data.frequency_bands) and not self.data.use_multiband_ssf:
            if self.model.input_channels != 1:
                raise ValueError(
                    f"input_channels ({self.model.input_channels}) should be 1 for single-band SSF"
                )
        
        if self.data.window_length <= 0:
            raise ValueError("window_length must be positive")
        
        if not 0 <= self.data.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__,
            'experiment': self.experiment.__dict__
        }


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
