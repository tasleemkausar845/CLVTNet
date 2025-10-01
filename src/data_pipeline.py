import os
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.config import Config
from src.utils.eeg_processing import EEGProcessor
from src.utils.topographic_mapping import TopographicMapper

class EEGDataset(Dataset):
    """PyTorch dataset for EEG topographic maps."""
    
    def __init__(self, 
                 data_path: str, 
                 labels_path: str, 
                 transform=None,
                 subject_ids_path: Optional[str] = None):
        """Initialize EEG dataset."""
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels_path)
        self.transform = transform
        
        if subject_ids_path and os.path.exists(subject_ids_path):
            self.subject_ids = np.load(subject_ids_path)
        else:
            self.subject_ids = None
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded dataset: {self.data.shape[0]} samples, "
                        f"{len(np.unique(self.labels))} classes")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if sample.ndim == 2:
            sample = sample[np.newaxis, ...]
        
        return torch.FloatTensor(sample), torch.LongTensor([label]).squeeze()

class EEGDataPipeline:
    """Complete EEG data processing pipeline."""
    
    def __init__(self, config: Config):
        """Initialize data pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.eeg_processor = EEGProcessor(config.data)
        self.topo_mapper = TopographicMapper(config.data)
        self.scaler = self._get_scaler()
        
    def _get_scaler(self):
        """Get appropriate scaler based on configuration."""
        if self.config.data.normalization == "z_score":
            return StandardScaler()
        elif self.config.data.normalization == "min_max":
            return MinMaxScaler()
        elif self.config.data.normalization == "robust":
            return RobustScaler()
        else:
            return None
    
    def process_raw_data(self, raw_data_dir: str, output_dir: str) -> str:
        """Process raw EEG data files."""
        self.logger.info("Starting raw EEG data processing")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = []
        labels = []
        subject_ids = []
        
        raw_files = self._scan_raw_data_directory(raw_data_dir)
        
        for file_info in raw_files:
            self.logger.info(f"Processing file: {file_info['path']}")
            eeg_data, file_labels = self._process_single_file(file_info)
            
            processed_data.extend(eeg_data)
            labels.extend(file_labels)
            subject_ids.extend([file_info['subject_id']] * len(eeg_data))
        
        processed_data = np.array(processed_data)
        labels = np.array(labels)
        subject_ids = np.array(subject_ids)
        
        np.save(os.path.join(output_dir, 'processed_eeg_data.npy'), processed_data)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
        np.save(os.path.join(output_dir, 'subject_ids.npy'), subject_ids)
        
        metadata = {
            'num_samples': len(processed_data),
            'num_channels': processed_data.shape[-1] if processed_data.ndim > 1 else 1,
            'sampling_rate': self.config.data.sampling_rate,
            'window_length': self.config.data.window_length,
            'overlap': self.config.data.overlap,
            'num_subjects': len(np.unique(subject_ids)),
            'num_classes': len(np.unique(labels))
        }
        
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Processed {len(processed_data)} samples from "
                        f"{metadata['num_subjects']} subjects")
        
        return output_dir
    
    def _scan_raw_data_directory(self, data_dir: str) -> List[Dict]:
        """Scan directory for raw EEG data files."""
        raw_files = []
        data_path = Path(data_dir)
        
        extensions = ['.csv', '.txt', '.edf', '.set', '.fif']
        
        for ext in extensions:
            for file_path in data_path.rglob(f'*{ext}'):
                file_info = self._extract_file_metadata(file_path)
                if file_info:
                    raw_files.append(file_info)
        
        if not raw_files:
            raise FileNotFoundError(f"No EEG data files found in {data_dir}")
        
        return raw_files
    
    def _extract_file_metadata(self, file_path: Path) -> Optional[Dict]:
        """Extract metadata from filename and directory structure."""
        parts = file_path.parts
        filename = file_path.stem
        
        metadata = {
            'path': str(file_path),
            'filename': filename,
            'subject_id': 1,
            'condition': 'unknown',
            'trial': 1,
            'label': 0
        }
        
        for part in parts:
            if 'subject' in part.lower() or 'sub' in part.lower():
                try:
                    metadata['subject_id'] = int(''.join(filter(str.isdigit, part)))
                except ValueError:
                    pass
        
        if 'attend_left' in filename.lower() or 'left' in filename.lower():
            metadata['label'] = 0
            metadata['condition'] = 'attend_left'
        elif 'attend_right' in filename.lower() or 'right' in filename.lower():
            metadata['label'] = 1
            metadata['condition'] = 'attend_right'
        
        for part in parts:
            if 'attend_left' in part.lower() or part.lower() == 'left':
                metadata['label'] = 0
                metadata['condition'] = 'attend_left'
            elif 'attend_right' in part.lower() or part.lower() == 'right':
                metadata['label'] = 1
                metadata['condition'] = 'attend_right'
        
        if 'trial' in filename.lower() or 'tra' in filename.lower():
            try:
                trial_part = filename.lower().split('trial')[-1] if 'trial' in filename.lower() else filename.lower().split('tra')[-1]
                metadata['trial'] = int(''.join(filter(str.isdigit, trial_part)))
            except (ValueError, IndexError):
                pass
        
        return metadata
    
    def _process_single_file(self, file_info: Dict) -> Tuple[List[np.ndarray], List[int]]:
        """Process a single EEG data file."""
        file_path = file_info['path']
        
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, header=None)
            eeg_data = data.values
        elif file_path.endswith('.txt'):
            eeg_data = np.loadtxt(file_path, delimiter=',')
        else:
            try:
                data = pd.read_csv(file_path, header=None)
                eeg_data = data.values
            except:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        eeg_data = self.eeg_processor.preprocess_signal(eeg_data)
        windows, window_labels = self.eeg_processor.extract_windows(
            eeg_data, file_info['label']
        )
        
        return windows, window_labels
    
    def generate_topographic_maps(self, processed_data_path: str, output_dir: str) -> str:
        """Generate topographic maps from processed EEG data."""
        self.logger.info("Generating EEG topographic maps")
        os.makedirs(output_dir, exist_ok=True)
        
        eeg_data = np.load(os.path.join(processed_data_path, 'processed_eeg_data.npy'))
        labels = np.load(os.path.join(processed_data_path, 'labels.npy'))
        
        subject_ids_path = os.path.join(processed_data_path, 'subject_ids.npy')
        if os.path.exists(subject_ids_path):
            subject_ids = np.load(subject_ids_path)
        else:
            subject_ids = np.ones(len(labels))
        
        topo_maps = []
        
        for i, eeg_sample in enumerate(eeg_data):
            if i % 1000 == 0:
                self.logger.info(f"Processing sample {i+1}/{len(eeg_data)}")
            
            topo_map = self.topo_mapper.eeg_to_topographic_map(eeg_sample)
            topo_maps.append(topo_map)
        
        topo_maps = np.array(topo_maps)
        
        if self.scaler:
            original_shape = topo_maps.shape
            topo_maps_flat = topo_maps.reshape(original_shape[0], -1)
            topo_maps_flat = self.scaler.fit_transform(topo_maps_flat)
            topo_maps = topo_maps_flat.reshape(original_shape)
        
        np.save(os.path.join(output_dir, 'topographic_maps.npy'), topo_maps)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
        np.save(os.path.join(output_dir, 'subject_ids.npy'), subject_ids)
        
        if self.scaler:
            with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        self.logger.info(f"Generated {len(topo_maps)} topographic maps")
        
        return output_dir
    
    def create_data_loaders(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders for training, validation, and testing."""
        self.logger.info("Creating data loaders")
        
        maps_path = os.path.join(data_path, 'topographic_maps.npy')
        labels_path = os.path.join(data_path, 'labels.npy')
        subjects_path = os.path.join(data_path, 'subject_ids.npy')
        
        labels = np.load(labels_path)
        
        if os.path.exists(subjects_path):
            subject_ids = np.load(subjects_path)
            train_idx, temp_idx = self._subject_based_split(subject_ids, test_size=0.4, random_state=self.config.experiment.seed)
            val_idx, test_idx = self._subject_based_split(subject_ids[temp_idx], test_size=0.5, random_state=self.config.experiment.seed)
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]
        else:
            indices = np.arange(len(labels))
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.4, stratify=labels, random_state=self.config.experiment.seed
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=self.config.experiment.seed
            )
        
        full_dataset = EEGDataset(maps_path, labels_path, subjects_path if os.path.exists(subjects_path) else None)
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.experiment.num_workers,
            pin_memory=self.config.experiment.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            pin_memory=self.config.experiment.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            pin_memory=self.config.experiment.pin_memory
        )
        
        self.logger.info(f"Created data loaders - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _subject_based_split(self, subject_ids: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split data based on subjects to prevent data leakage."""
        unique_subjects = np.unique(subject_ids)
        
        train_subjects, test_subjects = train_test_split(
            unique_subjects, test_size=test_size, random_state=random_state
        )
        
        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
        test_idx = np.where(np.isin(subject_ids, test_subjects))[0]
        
        return train_idx, test_idx