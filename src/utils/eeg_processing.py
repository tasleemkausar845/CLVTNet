import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import zscore
import logging

class EEGProcessor:
    """Simplified EEG signal processing utilities."""
    
    def __init__(self, config):
        """Initialize EEG processor with configuration.
        
        Args:
            config: Data configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def preprocess_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """Apply basic preprocessing to raw EEG signal.
        
        Args:
            raw_signal: Raw EEG data (time_points, channels)
            
        Returns:
            Preprocessed EEG signal
        """
        signal = raw_signal.copy()
        
        # Handle different input shapes
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]
        
        self.logger.debug(f"Preprocessing signal shape: {signal.shape}")
        
        # Simple normalization (z-score per channel)
        signal = self._normalize_signal(signal)
        
        return signal
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize EEG signal."""
        normalized_signal = np.zeros_like(signal)
        
        for ch in range(signal.shape[1]):
            channel_data = signal[:, ch]
            # Z-score normalization
            if np.std(channel_data) > 1e-6:  # Avoid division by zero
                normalized_signal[:, ch] = (channel_data - np.mean(channel_data)) / np.std(channel_data)
            else:
                normalized_signal[:, ch] = channel_data
        
        return normalized_signal
    
    def extract_windows(self, signal: np.ndarray, label: int) -> Tuple[List[np.ndarray], List[int]]:
        """Extract overlapping windows from continuous EEG signal.
        
        Args:
            signal: Preprocessed EEG signal (time_points, channels)
            label: Class label for this signal
            
        Returns:
            Tuple of (window_list, label_list)
        """
        window_length_samples = int(self.config.window_length * self.config.sampling_rate)
        overlap_samples = int(window_length_samples * self.config.overlap)
        step_size = window_length_samples - overlap_samples
        
        windows = []
        labels = []
        
        # Extract windows
        start = 0
        while start + window_length_samples <= signal.shape[0]:
            window = signal[start:start + window_length_samples, :]
            
            # Simple quality check
            if self._is_window_valid(window):
                windows.append(window)
                labels.append(label)
            
            start += step_size
        
        self.logger.debug(f"Extracted {len(windows)} valid windows from signal")
        
        return windows, labels
    
    def _is_window_valid(self, window: np.ndarray) -> bool:
        """Check if EEG window contains valid data."""
        # Check for flat signals
        if np.any(np.var(window, axis=0) < 1e-6):
            return False
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(window)):
            return False
        
        return True