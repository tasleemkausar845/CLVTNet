import numpy as np
import logging
from typing import List, Tuple
from scipy.signal import butter, filtfilt
from mne.preprocessing import ICA
import mne

class EEGProcessor:
    """EEG signal processing utilities aligned with preprocessing described in the paper."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess_signal(self, raw_signal: np.ndarray, sfreq: int = None) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. ICA for artifact removal
        2. Baseline correction
        3. Deionization filter (remove low freq drift)
        4. Channel normalization (z-score)
        5. Downsampling to target sampling rate if needed
        """
        
        signal = raw_signal.copy()

        # Handle shape
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]
        
        sfreq = sfreq or self.config.sampling_rate

        # ----- Step 1: ICA removal of eye/muscle artifacts (requires mne Raw object) -----
        try:
            raw_mne = mne.io.RawArray(signal.T, mne.create_info(
                ch_names=[f"Ch{i}" for i in range(signal.shape[1])],
                sfreq=sfreq,
                ch_types='eeg'
            ))

            ica = ICA(n_components=min(20, signal.shape[1]), random_state=97)
            ica.fit(raw_mne)
            # Here you could manually select components; as placeholder remove None
            raw_mne = ica.apply(raw_mne)
            signal = raw_mne.get_data().T
            self.logger.debug("ICA artifact removal applied")
        except Exception as e:
            self.logger.warning(f"ICA step skipped: {e}")

        # ----- Step 2: Baseline correction -----
        signal = self._baseline_correction(signal)

        # ----- Step 3: Deionization filter (remove low freq drift < 0.5 Hz) -----
        signal = self._highpass_filter(signal, sfreq, cutoff=0.5)

        # ----- Step 4: Normalize per channel -----
        signal = self._normalize_signal(signal)

        # ----- Step 5: Downsample if needed -----
        if sfreq != self.config.sampling_rate:
            self.logger.debug(f"Resampling from {sfreq} Hz to {self.config.sampling_rate} Hz")
            signal = mne.filter.resample(signal.T, up=self.config.sampling_rate, down=sfreq).T

        return signal

    def _baseline_correction(self, signal: np.ndarray) -> np.ndarray:
        """Remove mean value from each channel (baseline drift correction)."""
        return signal - np.mean(signal, axis=0)

    def _highpass_filter(self, signal: np.ndarray, sfreq: float, cutoff: float = 0.5) -> np.ndarray:
        """Remove very low frequency drift via highpass Butterworth filter."""
        b, a = butter(N=2, Wn=cutoff/(sfreq/2), btype='high')
        return filtfilt(b, a, signal, axis=0)

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalization for each channel."""
        normalized = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            channel_data = signal[:, ch]
            std = np.std(channel_data)
            if std > 1e-6:
                normalized[:, ch] = (channel_data - np.mean(channel_data)) / std
            else:
                normalized[:, ch] = channel_data
        return normalized

    def extract_windows(self, signal: np.ndarray, label: int) -> Tuple[List[np.ndarray], List[int]]:
        """Same logic as before for overlapping window extraction."""
        window_len = int(self.config.window_length * self.config.sampling_rate)
        overlap_len = int(window_len * self.config.overlap)
        step_size = window_len - overlap_len

        windows, labels = [], []
        start = 0
        while start + window_len <= signal.shape[0]:
            window = signal[start:start + window_len, :]
            if self._is_window_valid(window):
                windows.append(window)
                labels.append(label)
            start += step_size
        return windows, labels

    def _is_window_valid(self, window: np.ndarray) -> bool:
        """Reject windows with flat signals or NaNs."""
        if np.any(np.var(window, axis=0) < 1e-6):
            return False
        if np.any(~np.isfinite(window)):
            return False
        return True