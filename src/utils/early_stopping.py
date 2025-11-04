import numpy as np

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy/metrics
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(self, current_score: float) -> bool:
        """Check if training should stop.
        
        Args:
            current_score: Current validation score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.wait = 0
        self.best_score = None
        self.should_stop = False