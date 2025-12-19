import numpy as np
import pandas as pd
from pathlib import Path
import logging
import re

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logging.warning("MNE not available. Install with: pip install mne")


class KULDatasetLoader:
    """Loader for KUL Auditory Attention Detection Dataset."""

    def __init__(self, data_root: str, config):
        self.data_root = Path(data_root)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # dataset-specific constants
        self.n_subjects = 16
        self.n_channels = 64
        self.expected_channels = self._kul_channel_names()

    def _kul_channel_names(self):
        """Return expected 64-channel names for KUL dataset."""
        return [
            'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
            'T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz',
            'P4','P8','PO9','O1','Oz','O2','PO10','AF7','AF3','AF4','AF8',
            'F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10','C5','C1',
            'C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7',
            'PO3','POz','PO4','PO8','FCz','Iz'
        ]

    def _parse_filename(self, filepath: Path):
        """Extract metadata from KUL filename/path."""
        fp_str = str(filepath).lower()
        md = {
            'filepath': str(filepath),
            'subject_id': None,
            'trial_id': None,
            'story_id': None,
            'label': None
        }

        subj = re.search(r'subject[_\s]*(\d+)', fp_str)
        trial = re.search(r'trial[_\s]*(\d+)', fp_str)
        story = re.search(r'story[_\s]*(\d+)', fp_str)

        if subj: md['subject_id'] = int(subj.group(1))
        if trial: md['trial_id'] = int(trial.group(1))
        if story: md['story_id'] = int(story.group(1))

        if 'left' in fp_str or 'attend_left' in fp_str:
            md['label'] = 0
        elif 'right' in fp_str or 'attend_right' in fp_str:
            md['label'] = 1

        # defaults if not found
        if md['subject_id'] is None:
            md['subject_id'] = 1
        if md['trial_id'] is None:
            md['trial_id'] = 1
        if md['story_id'] is None:
            # derive story from trial pattern
            md['story_id'] = ((md['trial_id'] - 1) % 4) + 1
        if md['label'] is None:
            md['label'] = 0

        return md

    def load_eeg_file(self, filepath: Path):
        """Load EEG data from various formats using MNE or NumPy."""
        metadata = self._parse_filename(filepath)
        ext = filepath.suffix.lower()

        if not MNE_AVAILABLE and ext in ('.edf', '.fif', '.set'):
            raise ImportError("MNE required for EDF/FIF/SET loading")

        if ext == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')
        elif ext == '.fif':
            raw = mne.io.read_raw_fif(filepath, preload=True, verbose='ERROR')
        elif ext == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose='ERROR')
        elif ext in ('.csv', '.txt'):
            data = np.loadtxt(filepath, delimiter=',')
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data, metadata
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if raw.info['sfreq'] != self.config.sampling_rate:
            raw.resample(self.config.sampling_rate, verbose='ERROR')

        data = raw.get_data().T
        if data.shape[1] != self.n_channels:
            self.logger.warning(f"{filepath.name} channels mismatch: got {data.shape[1]}, expected {self.n_channels}")

        return data, metadata

    def scan_dataset(self):
        """Scan KUL dataset directory for files."""
        exts = ['.edf', '.fif', '.set', '.csv', '.txt']
        files = []
        for ext in exts:
            files.extend(self.data_root.rglob(f'*{ext}'))

        recs = [self._parse_filename(f) for f in files]
        return pd.DataFrame(recs)

    def create_leakage_proof_splits(self, df: pd.DataFrame):
        """TLHO-V split: per subject, 80% train / 10% val / 10% test."""
        splits = []
        for subj in df['subject_id'].unique():
            subj_df = df[df['subject_id'] == subj]
            trials = subj_df['trial_id'].unique()
            np.random.shuffle(trials)
            n_train = int(0.8 * len(trials))
            n_val = int(0.1 * len(trials))
            train_trials = trials[:n_train]
            val_trials = trials[n_train:n_train+n_val]
            test_trials = trials[n_train+n_val:]
            splits.append({
                'train_idx': subj_df[subj_df['trial_id'].isin(train_trials)].index.tolist(),
                'val_idx': subj_df[subj_df['trial_id'].isin(val_trials)].index.tolist(),
                'test_idx': subj_df[subj_df['trial_id'].isin(test_trials)].index.tolist()
            })
        self.logger.info(f"Created {len(splits)} TLHO-V splits for KUL")
        return splits


class DTUDatasetLoader:
    """Loader for DTU Auditory Attention Detection Dataset."""

    def __init__(self, data_root: str, config):
        self.data_root = Path(data_root)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.n_subjects = 18  # dataset-specific

    def _parse_filename(self, filepath: Path):
        """Extract metadata from DTU filename/path."""
        fp_str = str(filepath).lower()
        md = {
            'filepath': str(filepath),
            'subject_id': None,
            'trial_id': None,
            'label': None
        }

        subj = re.search(r'subject[_\s]*(\d+)', fp_str)
        trial = re.search(r'trial[_\s]*(\d+)', fp_str)
        if subj: md['subject_id'] = int(subj.group(1))
        if trial: md['trial_id'] = int(trial.group(1))

        if 'speaker1' in fp_str or 'spk1' in fp_str or 'attend_1' in fp_str:
            md['label'] = 0
        elif 'speaker2' in fp_str or 'spk2' in fp_str or 'attend_2' in fp_str:
            md['label'] = 1

        if md['subject_id'] is None: md['subject_id'] = 1
        if md['trial_id'] is None: md['trial_id'] = 1
        if md['label'] is None: md['label'] = 0

        return md

    def load_eeg_file(self, filepath: Path):
        """Load EEG data using MNE or NumPy."""
        metadata = self._parse_filename(filepath)
        ext = filepath.suffix.lower()

        if not MNE_AVAILABLE and ext in ('.edf', '.fif', '.set'):
            raise ImportError("MNE required for EDF/FIF/SET loading")

        if ext == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')
        elif ext == '.fif':
            raw = mne.io.read_raw_fif(filepath, preload=True, verbose='ERROR')
        elif ext == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose='ERROR')
        elif ext in ('.csv', '.txt'):
            data = np.loadtxt(filepath, delimiter=',')
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data, metadata
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if raw.info['sfreq'] != self.config.sampling_rate:
            raw.resample(self.config.sampling_rate, verbose='ERROR')

        data = raw.get_data().T
        return data, metadata

    def scan_dataset(self):
        """Scan DTU dataset directory for files."""
        exts = ['.edf', '.fif', '.set', '.csv', '.txt']
        files = []
        for ext in exts:
            files.extend(self.data_root.rglob(f'*{ext}'))

        recs = [self._parse_filename(f) for f in files]
        return pd.DataFrame(recs)

    def create_leakage_proof_splits(self, df: pd.DataFrame):
        """TLHO-V split same as KUL."""
        splits = []
        for subj in df['subject_id'].unique():
            subj_df = df[df['subject_id'] == subj]
            trials = subj_df['trial_id'].unique()
            np.random.shuffle(trials)
            n_train = int(0.8 * len(trials))
            n_val = int(0.1 * len(trials))
            train_trials = trials[:n_train]
            val_trials = trials[n_train:n_train+n_val]
            test_trials = trials[n_train+n_val:]
            splits.append({
                'train_idx': subj_df[subj_df['trial_id'].isin(train_trials)].index.tolist(),
                'val_idx': subj_df[subj_df['trial_id'].isin(val_trials)].index.tolist(),
                'test_idx': subj_df[subj_df['trial_id'].isin(test_trials)].index.tolist()
            })
        self.logger.info(f"Created {len(splits)} TLHO-V splits for DTU")
        return splits

import scipy.io

class AVGCDatasetLoader:
    
    def __init__(self, data_root: str, config, n_channels: int):
        self.data_root = Path(data_root)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Critical: Set YOUR actual channel count here (e.g., 32, 64, etc.)
        self.n_channels = n_channels  # User-defined channel count
    
    def _parse_filename(self, filepath: Path):
        """Extract metadata from filename/path (unchanged, as it doesn't use electrode names)."""
        fp_str = str(filepath).lower()
        md = {
            'filepath': str(filepath),
            'subject_id': None,
            'trial_id': None,
            'story_id': None,
            'label': None
        }
        
        # Extract subject ID (adjust regex if your filenames use different patterns)
        subj = re.search(r'(?:subject|subj|s)(?:[_\s]*)(\d+)', fp_str)
        if subj:
            md['subject_id'] = int(subj.group(1))
        
        # Extract trial ID (adjust regex if needed)
        trial = re.search(r'(?:trial|run|session)(?:[_\s]*)(\d+)', fp_str)
        if trial:
            md['trial_id'] = int(trial.group(1))
        
        # Extract story ID (adjust regex if needed)
        story = re.search(r'(?:story|audio|stimulus)(?:[_\s]*)(\d+)', fp_str)
        if story:
            md['story_id'] = int(story.group(1))
        
        # Extract attention label (0: left, 1: right; adjust based on YOUR filenames)
        if 'left' in fp_str or 'attend_left' in fp_str:
            md['label'] = 0
        elif 'right' in fp_str or 'attend_right' in fp_str:
            md['label'] = 1
        
        # Set defaults if metadata not found (adjust as needed)
        md['subject_id'] = md['subject_id'] or 1
        md['trial_id'] = md['trial_id'] or 1
        md['story_id'] = md['story_id'] or ((md['trial_id'] - 1) % 4) + 1
        md['label'] = md['label'] if md['label'] is not None else 0
            
        return md
    
    def load_eeg_file(self, filepath: Path):
        metadata = self._parse_filename(filepath)
        
        # Load .mat file and extract EEG array
        mat_data = scipy.io.loadmat(filepath)
        eeg_data = None
        
        # Look for common variable names in YOUR .mat files (add/remove keys as needed)
        possible_keys = ['eeg', 'data', 'signal', 'X', 'raw_data']
        for key in possible_keys:
            if key in mat_data and isinstance(mat_data[key], np.ndarray):
                eeg_data = mat_data[key]
                break
        
        if eeg_data is None:
            raise ValueError(f"Could not find EEG array in {filepath}. Check variable names.")
        
        # Ensure shape: (time × channels). Transpose if needed.
        if eeg_data.shape[0] < eeg_data.shape[1]:  # Assume (channels × time) → transpose
            eeg_data = eeg_data.T
        
        # Validate channel count against YOUR data (critical!)
        if eeg_data.shape[1] != self.n_channels:
            self.logger.warning(
                f"Channel count mismatch in {filepath.name}: "
                f"Expected {self.n_channels}, got {eeg_data.shape[1]}"
            )
        
        return eeg_data, metadata
    
    def scan_dataset(self):
       
        files = list(self.data_root.rglob('*.mat'))
        if not files:
            self.logger.warning("No .mat files found in data root.")
            return pd.DataFrame()
        
        records = [self._parse_filename(f) for f in files]
        return pd.DataFrame(records)
    
    def create_leakage_proof_splits(self, df: pd.DataFrame):
        
        splits = []
        for subj in df['subject_id'].unique():
            subj_df = df[df['subject_id'] == subj]
            trials = subj_df['trial_id'].unique()
            np.random.shuffle(trials)
            
            n_train = int(0.8 * len(trials))
            n_val = int(0.1 * len(trials))
            train_trials = trials[:n_train]
            val_trials = trials[n_train:n_train+n_val]
            test_trials = trials[n_train+n_val:]
            
            splits.append({
                'train_idx': subj_df[subj_df['trial_id'].isin(train_trials)].index.tolist(),
                'val_idx': subj_df[subj_df['trial_id'].isin(val_trials)].index.tolist(),
                'test_idx': subj_df[subj_df['trial_id'].isin(test_trials)].index.tolist()
            })
        
        self.logger.info(f"Created {len(splits)} subject-specific splits.")
        return splits
