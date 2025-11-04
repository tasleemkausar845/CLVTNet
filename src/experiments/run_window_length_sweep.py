import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from src.config import Config
from ablation_configs import AblationConfig
from dataset_loaders import KULDatasetLoader, DTUDatasetLoader
from ssf_extraction import SSFExtractor
from data_pipeline_updated import EEGDataPipeline


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'window_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def run_window_length_experiment(config: Config, window_length: float, 
                                 dataset: str, output_dir: str, device: str):
    """
    Run experiment with specific window length.
    
    Args:
        config: Base configuration
        window_length: Window length in seconds
        dataset: Dataset name
        output_dir: Output directory
        device: Device to use
    
    Returns:
        Dictionary with results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment with window length: {window_length}s")
    
    config.data.window_length = window_length
    
    if dataset == 'KUL':
        dataset_loader = KULDatasetLoader(config.data.dataset_path, config.data)
    elif dataset == 'DTU':
        dataset_loader = DTUDatasetLoader(config.data.dataset_path, config.data)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    ssf_extractor = SSFExtractor(config.data)
    
    pipeline = EEGDataPipeline(
        config,
        dataset_loader=dataset_loader,
        ssf_extractor=ssf_extractor
    )
    
    processed_data_dir = Path(output_dir) / 'processed_data' / f'window_{window_length}s'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    if not (processed_data_dir / 'topographic_maps.npy').exists():
        logger.info("Processing data with SSF extraction")
        pipeline.process_raw_data(
            output_dir=str(processed_data_dir),
            cv_mode=config.data.cv_mode,
            n_folds=config.data.cv_folds
        )
    else:
        logger.info("Using existing processed data")
    
    results = []
    cv_splits_path = processed_data_dir / 'cv_splits.pkl'
    
    ablation_config = AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        use_se_in_fusion=True,
        use_conv_module=True
    )
    
    for fold in range(config.data.cv_folds):
        logger.info(f"Training fold {fold + 1}/{config.data.cv_folds}")
        
        train_loader, val_loader, test_loader = pipeline.create_data_loaders(
            data_path=str(processed_data_dir),
            cv_splits_path=str(cv_splits_path) if cv_splits_path.exists() else None,
            fold=fold
        )
        
        from clvtnet import CLVTNet
        model = CLVTNet(config, ablation_config=ablation_config)
        model = model.to(device)
        
        from training import Trainer
        trainer = Trainer(model, config, device=device, fold=fold)
        
        fold_results = trainer.train(train_loader, val_loader, test_loader)
        results.append(fold_results)
        
        logger.info(f"Fold {fold} - Test Accuracy: {fold_results['test_accuracy']:.4f}")
    
    mean_accuracy = np.mean([r['test_accuracy'] for r in results])
    std_accuracy = np.std([r['test_accuracy'] for r in results])
    
    logger.info(f"Window {window_length}s - Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return {
        'window_length': window_length,
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'fold_results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Run window length sweep experiments')
    parser.add_argument('--dataset', type=str, default='KUL', choices=['KUL', 'DTU'],
                       help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/window_sweep',
                       help='Output directory')
    parser.add_argument('--window_lengths', type=float, nargs='+',
                       default=[0.1, 1.0, 2.0, 5.0, 10.0],
                       help='Window lengths to test (in seconds)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cv_mode', type=str, default='subject',
                       choices=['subject', 'trial', 'story'],
                       help='Cross-validation mode')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_dir)
    logger.info("Starting window length sweep experiments")
    logger.info(f"Arguments: {args}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    config = Config()
    config.data.dataset_name = args.dataset
    config.data.dataset_path = args.dataset_path
    config.data.cv_mode = args.cv_mode
    config.data.cv_folds = args.cv_folds
    config.experiment.seed = args.seed
    
    logger.info(f"Testing window lengths: {args.window_lengths}")
    
    all_results = {}
    for window_length in args.window_lengths:
        try:
            results = run_window_length_experiment(
                config, window_length, args.dataset, args.output_dir, args.device
            )
            all_results[f'{window_length}s'] = results
        except Exception as e:
            logger.error(f"Error with window length {window_length}s: {str(e)}", exc_info=True)
    
    summary_file = Path(args.output_dir) / 'window_sweep_results.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All experiments complete. Results saved to {summary_file}")
    
    logger.info("\nWindow Length Sweep Results:")
    logger.info("-" * 80)
    for window, results in all_results.items():
        logger.info(f"{window}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")


if __name__ == '__main__':
    main()