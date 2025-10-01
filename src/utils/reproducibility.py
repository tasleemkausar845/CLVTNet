import os
import random
import numpy as np
import torch
import logging
from typing import Optional

def set_deterministic_mode(seed: int = 42, use_deterministic: bool = True):
    """Set deterministic mode for reproducible results.
    
    Args:
        seed: Random seed value
        use_deterministic: Whether to use deterministic algorithms (may slow down training)
    """
    logger = logging.getLogger(__name__)
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set PyTorch deterministic mode
    if use_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set environment variable for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.backends.cudnn.benchmark = True
    
    logger.info(f"Deterministic mode set with seed: {seed}")

def get_system_info() -> dict:
    """Get system information for reproducibility documentation."""
    import platform
    import sys
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info

class ReproducibilityManager:
    """Manages reproducibility settings and documentation."""
    
    def __init__(self, seed: int = 42, log_system_info: bool = True):
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        
        # Set deterministic mode
        set_deterministic_mode(seed)
        
        # Log system information
        if log_system_info:
            self.log_system_info()
    
    def log_system_info(self):
        """Log comprehensive system information."""
        info = get_system_info()
        
        self.logger.info("=== System Information ===")
        for key, value in info.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=== End System Information ===")
    
    def save_experiment_info(self, save_path: str, config: dict, additional_info: Optional[dict] = None):
        """Save experiment information for reproducibility.
        
        Args:
            save_path: Path to save experiment info
            config: Experiment configuration
            additional_info: Additional information to save
        """
        import json
        from datetime import datetime
        
        experiment_info = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'system_info': get_system_info(),
            'config': config
        }
        
        if additional_info:
            experiment_info.update(additional_info)
        
        with open(save_path, 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)
        
        self.logger.info(f"Experiment info saved to: {save_path}")

def verify_reproducibility(model_func, input_data, num_runs: int = 3):
    """Verify that a model produces consistent results across multiple runs.
    
    Args:
        model_func: Function that returns model output
        input_data: Input data for the model
        num_runs: Number of runs to compare
        
    Returns:
        True if results are consistent, False otherwise
    """
    results = []
    
    for run in range(num_runs):
        # Reset random states
        set_deterministic_mode(42)
        
        # Get model output
        with torch.no_grad():
            output = model_func(input_data)
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            results.append(output)
    
    # Check if all results are identical
    for i in range(1, num_runs):
        if not np.allclose(results[0], results[i], rtol=1e-5, atol=1e-8):
            return False
    
    return True