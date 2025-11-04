import os
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


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
        self.logger.info(
            f"Loaded dataset: {self.data.shape[0]} samples, "
            f"{len(np.unique(self.labels))} classes"
        )

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
    """
    Complete EEG data processing pipeline.
    - Works ONLY with KULDatasetLoader or DTUDatasetLoader
    - Creates leakage-proof CV splits using loader’s create_leakage_proof_splits()
    """

    def __init__(self, config, dataset_loader, ssf_extractor=None):
        """
        Args:
            config: Configuration object
            dataset_loader: Must be KULDatasetLoader or DTUDatasetLoader instance
            ssf_extractor: Optional SSFExtractor for spectro-spatial features
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        from eeg_processing import EEGProcessor
        from topographic_mapping import TopographicMapper

        self.eeg_processor = EEGProcessor(config.data)
        self.topo_mapper = TopographicMapper(config.data)
        self.scaler = self._get_scaler()

        self.dataset_loader = dataset_loader
        self.ssf_extractor = ssf_extractor

        self.use_ssf = getattr(config.data, 'use_ssf_extraction', False)
        self.use_multiband = getattr(config.data, 'use_multiband_ssf', False)

        if self.use_ssf and self.ssf_extractor is None:
            from ssf_extraction import SSFExtractor
            self.ssf_extractor = SSFExtractor(config.data)
            self.logger.info("Initialized SSF extractor")

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

    def process_raw_data(self, output_dir: str) -> str:
        """
        End-to-end processing:
        1. Scan dataset using loader
        2. Load each EEG file and metadata
        3. Preprocess and segment into windows
        4. Extract SSF / topographic maps
        5. Save arrays and metadata
        6. Create leakage-proof splits
        """
        self.logger.info("Starting EEG data processing pipeline")
        os.makedirs(output_dir, exist_ok=True)

        if self.dataset_loader is None:
            raise ValueError("dataset_loader must be provided (KUL/DTU loader)")

        df = self.dataset_loader.scan_dataset()
        self.logger.info(f"Found {len(df)} EEG files")

        all_ssf_maps = []
        all_labels = []
        all_metadata = []

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                self.logger.info(f"Processing file {idx+1}/{len(df)}")

            eeg_data, metadata = self.dataset_loader.load_eeg_file(Path(row['filepath']))
            eeg_data = self.eeg_processor.preprocess_signal(eeg_data)

            windows, window_labels = self.eeg_processor.extract_windows(
                eeg_data,
                metadata['label']
            )

            for window, label in zip(windows, window_labels):
                if self.use_ssf:
                    ssf_map = self.ssf_extractor.eeg_segment_to_ssf(
                        window,
                        self.topo_mapper,
                        use_multiband=self.use_multiband
                    )
                else:
                    channel_means = np.mean(window, axis=0)
                    ssf_map = self.topo_mapper.eeg_to_topographic_map(channel_means)

                all_ssf_maps.append(ssf_map)
                all_labels.append(label)

                win_md = metadata.copy()
                win_md['window_idx'] = len(all_ssf_maps) - 1
                all_metadata.append(win_md)

        # Convert to arrays and dataframe
        all_ssf_maps = np.array(all_ssf_maps)
        all_labels = np.array(all_labels)
        metadata_df = pd.DataFrame(all_metadata)

        self.logger.info(f"Generated {len(all_ssf_maps)} maps with shape {all_ssf_maps.shape}")

        # Save data
        np.save(os.path.join(output_dir, 'topographic_maps.npy'), all_ssf_maps)
        np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
        np.save(os.path.join(output_dir, 'subject_ids.npy'), metadata_df['subject_id'].values)

        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata_df, f)

        # CV splits
        cv_splits = self.dataset_loader.create_leakage_proof_splits(df)
        with open(os.path.join(output_dir, 'cv_splits.pkl'), 'wb') as f:
            pickle.dump(cv_splits, f)
        self.logger.info(f"Created {len(cv_splits)} CV splits")

        self.logger.info(f"Data processing complete: {output_dir}")
        return output_dir

    def create_data_loaders(self, data_path: str, fold: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test DataLoaders based on CV splits."""
        maps_path = os.path.join(data_path, 'topographic_maps.npy')
        labels_path = os.path.join(data_path, 'labels.npy')
        subjects_path = os.path.join(data_path, 'subject_ids.npy')
        splits_path = os.path.join(data_path, 'cv_splits.pkl')

        labels = np.load(labels_path)

        if os.path.exists(splits_path):
            with open(splits_path, 'rb') as f:
                cv_splits = pickle.load(f)
            if fold >= len(cv_splits):
                raise ValueError(f"Fold {fold} >= number of CV splits {len(cv_splits)}")

            split = cv_splits[fold]
            train_idx = split['train_idx']
            val_idx = split['val_idx']
            test_idx = split['test_idx']
        else:
            raise FileNotFoundError("cv_splits.pkl not found in data_path")

        full_dataset = EEGDataset(
            maps_path, 
            labels_path, 
            subjects_path if os.path.exists(subjects_path) else None
        )

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

        self.logger.info(
            f"DataLoaders created - Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        return train_loader, val_loader, test_loader