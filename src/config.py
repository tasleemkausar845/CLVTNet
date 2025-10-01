import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""
    name: str = "CLVTNet_EEG_Classification"
    seed: int = 42
    dataset: str = "KUL_DTU"
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class DataConfig:
    """Data processing configuration parameters."""
    sampling_rate: int = 1024
    window_length: float = 2.0  # seconds
    overlap: float = 0.5  # overlap ratio
    frequency_bands: Dict[str, List[float]] = field(default_factory=lambda: {
        'alpha': [8.0, 13.0],
        'beta': [13.0, 30.0],
        'gamma': [30.0, 100.0]
    })
    electrode_montage: str = "standard_1020"
    image_size: int = 64
    interpolation_method: str = "cubic"
    normalization: str = "z_score"  # z_score, min_max, robust
    artifact_removal: bool = True
    filter_low: float = 0.5
    filter_high: float = 50.0
    notch_frequency: float = 50.0  # Hz

@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""
    name: str = "CLVTNet"
    input_channels: int = 1
    num_classes: int = 2
    image_size: int = 64
    
    # Hierarchical feature dimensions
    embed_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Multi-head attention parameters
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    
    # Convolutional filter sizes for different stages
    filter_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    
    # Number of transformer blocks per stage
    num_blocks: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    
    # Dropout and regularization
    dropout_rate: float = 0.1
    drop_path_rate: float = 0.1
    
    # Activation functions
    activation: str = "gelu"
    
    # Positional encoding
    use_positional_encoding: bool = True
    
    # Channel attention reduction ratio
    channel_attention_reduction: int = 16

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    num_epochs: int = 200
    
    # Optimizer settings
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    
    # Learning rate scheduler
    scheduler: str = "cosine_annealing"
    warmup_epochs: int = 10
    warmup_lr: float = 1e-6
    min_lr: float = 1e-6
    
    # Loss function
    criterion: str = "cross_entropy"
    label_smoothing: float = 0.1
    
    # Regularization techniques
    gradient_clipping: float = 1.0
    
    # Data augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    augmentation_prob: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_delta: float = 1e-4
    
    # Model checkpointing
    save_best_only: bool = True
    save_frequency: int = 10  # epochs
    
    # Validation
    validation_frequency: int = 1  # epochs

@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'auc_roc', 'auc_pr', 'confusion_matrix'
    ])
    
    # Cross-validation
    cv_folds: int = 5
    stratified_cv: bool = True
    
    # Statistical testing
    bootstrap_samples: int = 1000
    confidence_interval: float = 0.95
    
    # Visualization
    generate_plots: bool = True
    plot_confusion_matrix: bool = True
    plot_roc_curves: bool = True
    plot_feature_maps: bool = True
    plot_attention_weights: bool = True
    
    # Model interpretability
    compute_gradcam: bool = True
    compute_lime: bool = False  # computationally expensive
    
    # Per-class analysis
    per_class_metrics: bool = True

@dataclass
class Config:
    """Complete configuration object."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

class ConfigManager:
    """Configuration management utility."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Config:
        """Load configuration from file or create default."""
        if self.config_path and os.path.exists(self.config_path):
            return self._load_from_file()
        else:
            return Config()  # Use defaults
    
    def _load_from_file(self) -> Config:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return self._dict_to_config(config_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to configuration object."""
        experiment_config = ExperimentConfig(
            **config_dict.get('experiment', {})
        )
        
        data_config = DataConfig(
            **config_dict.get('data', {})
        )
        
        model_config = ModelConfig(
            **config_dict.get('model', {})
        )
        
        training_config = TrainingConfig(
            **config_dict.get('training', {})
        )
        
        evaluation_config = EvaluationConfig(
            **config_dict.get('evaluation', {})
        )
        
        return Config(
            experiment=experiment_config,
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config
        )
    
    def get_config(self) -> Config:
        """Get configuration object."""
        return self.config
    
    def save_config(self, save_path: str) -> None:
        """Save current configuration to file."""
        config_dict = {
            'experiment': self.config.experiment.__dict__,
            'data': self.config.data.__dict__,
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__,
            'evaluation': self.config.evaluation.__dict__
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for section, values in updates.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def validate_config(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        # Validate model configuration
        if len(self.config.model.embed_dims) != len(self.config.model.num_heads):
            errors.append("embed_dims and num_heads must have same length")
        
        if len(self.config.model.embed_dims) != len(self.config.model.num_blocks):
            errors.append("embed_dims and num_blocks must have same length")
        
        # Validate data configuration
        if self.config.data.overlap < 0 or self.config.data.overlap >= 1:
            errors.append("overlap must be in range [0, 1)")
        
        # Validate training configuration
        if self.config.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.config.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        return errors