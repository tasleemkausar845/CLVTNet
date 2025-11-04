import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import logging
from typing import Tuple, Dict


class SSFExtractor:
    """
    Extracts spectro-spatial features from EEG data using FFT/PSD.
    """

    def __init__(self, config):
        """
        Args:
            config: DataConfig with sampling_rate, frequency_bands, psd_method
        """
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.frequency_bands = config.frequency_bands
        self.psd_method = config.psd_method
        self.logger = logging.getLogger(__name__)

        # Only check alpha if single-band extraction
        if not config.use_multiband_ssf and 'alpha' not in self.frequency_bands:
            raise ValueError("Alpha band must be configured in frequency_bands for single-band SSF")

        if 'alpha' in self.frequency_bands:
            self.alpha_band = self.frequency_bands['alpha']
            self.logger.info(f"Alpha band: {self.alpha_band[0]}-{self.alpha_band[1]} Hz")

    def extract_psd(self, eeg_segment: np.ndarray, method: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract PSD from EEG segment.

        Args:
            eeg_segment: (time_points, n_channels)
            method: 'welch' or 'fft'
        """
        method = method or self.psd_method

        if method == 'welch':
            nperseg = min(eeg_segment.shape[0], int(2 * self.sampling_rate))
            freqs, psd = signal.welch(eeg_segment, fs=self.sampling_rate,
                                      nperseg=nperseg, axis=0)
        elif method == 'fft':
            n = eeg_segment.shape[0]
            fft_vals = fft(eeg_segment, axis=0)
            freqs = fftfreq(n, 1 / self.sampling_rate)
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            psd = np.abs(fft_vals[pos_mask, :]) ** 2 / n
        else:
            raise ValueError(f"Unknown PSD method: {method}")

        return freqs, psd

    def extract_band_power(self, freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """Extract power from given band (low, high)."""
        low, high = band
        idx = (freqs >= low) & (freqs <= high)
        if not np.any(idx):
            self.logger.warning(f"No frequencies in {low}-{high} Hz range")
            return np.zeros(psd.shape[1])
        return np.trapz(psd[idx, :], freqs[idx], axis=0)

    def extract_multiband_power(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract power for multiple frequency bands."""
        return {band_name: self.extract_band_power(freqs, psd, tuple(band))
                for band_name, band in self.frequency_bands.items()}

    def eeg_segment_to_ssf(self, eeg_segment: np.ndarray, topographic_mapper, use_multiband: bool = False) -> np.ndarray:
        """
        Convert EEG segment to SSF map.

        Returns:
            (image_size, image_size) for single-band,
            or (n_bands, image_size, image_size) for multi-band
        """
        freqs, psd = self.extract_psd(eeg_segment)

        if use_multiband:
            band_powers = self.extract_multiband_power(freqs, psd)
            ssf_maps = [topographic_mapper.eeg_to_topographic_map(band_powers[band_name])
                        for band_name in self.frequency_bands.keys()]
            ssf_map = np.stack(ssf_maps, axis=0)
            self.logger.debug(f"Multi-band SSF map shape={ssf_map.shape}")
        else:
            alpha_power = self.extract_band_power(freqs, psd, tuple(self.alpha_band))
            ssf_map = topographic_mapper.eeg_to_topographic_map(alpha_power)
            self.logger.debug(f"Alpha-band SSF map shape={ssf_map.shape}")

        return ssf_map