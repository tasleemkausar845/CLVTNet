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
from ablation_configs import ABLATION_CONFIGS, get_ablation_config
from dataset_loaders import KULDatasetLoader, DTUDatasetLoader
from ssf_extraction import SSFExtractor
from data_pipeline_updated import EEGDataPipeline


def setup_logging(output_dir: str, experiment_name: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def run_ablation_study(config: Config, ablation_name: str, dataset: str, 
                      output_dir: str, device: str):
    """
    Run a single ablation study experiment.
    
    Args:
        config: Base configuration
        ablation_name: Name of ablation configuration
        dataset: Dataset name ('KUL' or 'DTU')
        output_dir: Output directory
        device: Device to use
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting ablation study: {ablation_name}")
    
    ablation_config = get_ablation_config(ablation_name)
    logger.info(f"Ablation configuration: {ablation_config}")
    
    config.data.use_multiband_ssf = ablation_config.use_multiband_ssf
    
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
    
    processed_data_dir = Path(output_dir) / 'processed_data' / ablation_name
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    if not (processed_data_dir / 'topographic_maps.npy').exists():
        logger.info("Processing raw data with SSF extraction")
        pipeline.process_raw_data(
            output_dir=str(processed_data_dir),
            cv_mode=config.data.cv_mode,
            n_folds=config.data.cv_folds
        )
    else:
        logger.info("Using existing processed data")
    
    results = []
    cv_splits_path = processed_data_dir / 'cv_splits.pkl'
    
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
    
    logger.info(f"Ablation {ablation_name} - Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    results_summary = {
        'ablation_name': ablation_name,
        'ablation_config': ablation_config.__dict__,
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'fold_results': results
    }
    
    results_file = Path(output_dir) / 'results' / f'{ablation_name}_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description='Run CLVTNet ablation studies')
    parser.add_argument('--dataset', type=str, default='KUL', choices=['KUL', 'DTU'],
                       help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/ablation_studies',
                       help='Output directory')
    parser.add_argument('--ablations', type=str, nargs='+', default=None,
                       help='Specific ablations to run (default: all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cv_mode', type=str, default='subject',
                       choices=['subject', 'trial', 'story'],
                       help='Cross-validation mode')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_dir, 'ablation_study')
    logger.info("Starting ablation studies")
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
    
    ablations_to_run = args.ablations if args.ablations else list(ABLATION_CONFIGS.keys())
    
    logger.info(f"Running {len(ablations_to_run)} ablation studies: {ablations_to_run}")
    
    all_results = {}
    for ablation_name in ablations_to_run:
        try:
            results = run_ablation_study(
                config, ablation_name, args.dataset, args.output_dir, args.device
            )
            all_results[ablation_name] = results
        except Exception as e:
            logger.error(f"Error in ablation {ablation_name}: {str(e)}", exc_info=True)
    
    summary_file = Path(args.output_dir) / 'ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All ablation studies complete. Summary saved to {summary_file}")
    
    logger.info("\nAblation Study Results Summary:")
    logger.info("-" * 80)
    for name, results in all_results.items():
        logger.info(f"{name}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")


if __name__ == '__main__':
    main()