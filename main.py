#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path

from src.config import ConfigManager
from src.data_pipeline import EEGDataPipeline
from src.models.clvtnet import CLVTNet
from src.training import TrainingManager
from src.evaluation import EvaluationManager
from src.utils.reproducibility import set_deterministic_mode

def setup_logging(log_level='INFO'):
    """Configure logging for the entire pipeline."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler('clvtnet_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CLVTNet: CNN-based Spatial Channel Enhanced Vision Transformer for EEG'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'evaluate', 'pipeline'],
        default='pipeline',
        help='Execution mode'
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True,
        help='Path to raw EEG data directory'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model-path', 
        type=str,
        help='Path to pretrained model (for evaluation mode)'
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()

def run_complete_pipeline(config_path, data_dir, output_dir):
    """Execute complete EEG processing and training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting CLVTNet pipeline")
    
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    set_deterministic_mode(config.experiment.seed)
    
    data_pipeline = EEGDataPipeline(config)
    
    logger.info("Processing raw EEG data")
    processed_data_path = data_pipeline.process_raw_data(
        raw_data_dir=data_dir,
        output_dir=os.path.join(output_dir, 'processed_data')
    )
    
    logger.info("Generating topographic maps")
    topo_maps_path = data_pipeline.generate_topographic_maps(
        processed_data_path=processed_data_path,
        output_dir=os.path.join(output_dir, 'topographic_maps')
    )
    
    logger.info("Creating data loaders")
    train_loader, val_loader, test_loader = data_pipeline.create_data_loaders(
        data_path=topo_maps_path
    )
    
    model = CLVTNet(config)
    
    logger.info("Starting model training")
    trainer = TrainingManager(config, output_dir)
    trained_model_path = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    logger.info("Evaluating trained model")
    evaluator = EvaluationManager(config, os.path.join(output_dir, 'evaluation'))
    evaluation_results = evaluator.comprehensive_evaluation(
        model_path=trained_model_path,
        test_loader=test_loader,
        val_loader=val_loader
    )
    
    logger.info("Pipeline completed successfully")
    logger.info(f"Results saved to: {output_dir}")
    
    return evaluation_results

def run_training_only(config_path, data_dir, output_dir):
    """Execute training phase only."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training mode")
    
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    set_deterministic_mode(config.experiment.seed)
    
    data_pipeline = EEGDataPipeline(config)
    train_loader, val_loader, _ = data_pipeline.create_data_loaders(data_dir)
    
    model = CLVTNet(config)
    trainer = TrainingManager(config, output_dir)
    
    trained_model_path = trainer.train(model, train_loader, val_loader)
    logger.info(f"Training completed. Model saved to: {trained_model_path}")
    
    return trained_model_path

def run_evaluation_only(config_path, model_path, data_dir, output_dir):
    """Execute evaluation phase only."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation mode")
    
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    data_pipeline = EEGDataPipeline(config)
    _, val_loader, test_loader = data_pipeline.create_data_loaders(data_dir)
    
    evaluator = EvaluationManager(config, output_dir)
    results = evaluator.comprehensive_evaluation(model_path, test_loader, val_loader)
    
    logger.info("Evaluation completed")
    return results

def main():
    """Main execution function."""
    args = parse_arguments()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.mode == 'pipeline':
            results = run_complete_pipeline(args.config, args.data_dir, args.output_dir)
        elif args.mode == 'train':
            results = run_training_only(args.config, args.data_dir, args.output_dir)
        elif args.mode == 'evaluate':
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")
            results = run_evaluation_only(
                args.config, args.model_path, args.data_dir, args.output_dir
            )
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path

from src.config import ConfigManager
from src.data_pipeline import EEGDataPipeline
from src.models.clvtnet import CLVTNet
from src.training import TrainingManager
from src.evaluation import EvaluationManager
from src.utils.reproducibility import set_deterministic_mode

def setup_logging(log_level='INFO'):
    """Configure logging for the entire pipeline."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CLVTNet: Convolutional-LSTM Vision Transformer for EEG Classification'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'evaluate', 'pipeline'],
        default='pipeline',
        help='Execution mode'
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True,
        help='Path to raw EEG data directory'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model-path', 
        type=str,
        help='Path to pretrained model (for evaluation mode)'
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()

def run_full_pipeline(config_path, data_dir, output_dir):
    """Execute complete EEG processing and training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete EEG processing pipeline")
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # Set deterministic behavior
    set_deterministic_mode(config.experiment.seed)
    
    # Initialize data pipeline
    data_pipeline = EEGDataPipeline(config)
    
    # Process raw EEG data
    logger.info("Processing raw EEG data")
    processed_data_path = data_pipeline.process_raw_data(
        raw_data_dir=data_dir,
        output_dir=os.path.join(output_dir, 'processed_data')
    )
    
    # Generate topographic maps
    logger.info("Generating EEG topographic representations")
    topo_maps_path = data_pipeline.generate_topographic_maps(
        processed_data_path=processed_data_path,
        output_dir=os.path.join(output_dir, 'topographic_maps')
    )
    
    # Prepare datasets
    logger.info("Preparing training and validation datasets")
    train_loader, val_loader, test_loader = data_pipeline.create_data_loaders(
        data_path=topo_maps_path
    )
    
    # Initialize model
    model = CLVTNet(config)
    
    # Training phase
    logger.info("Starting model training")
    trainer = TrainingManager(config, output_dir)
    trained_model_path = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Evaluation phase
    logger.info("Evaluating trained model")
    evaluator = EvaluationManager(config, output_dir)
    evaluation_results = evaluator.comprehensive_evaluation(
        model_path=trained_model_path,
        test_loader=test_loader,
        val_loader=val_loader
    )
    
    logger.info("Pipeline completed successfully")
    logger.info(f"Results saved to: {output_dir}")
    
    return evaluation_results

def run_training_only(config_path, data_dir, output_dir):
    """Execute training phase only."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training-only mode")
    
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    set_deterministic_mode(config.experiment.seed)
    
    # Assume data is already processed
    data_pipeline = EEGDataPipeline(config)
    train_loader, val_loader, _ = data_pipeline.create_data_loaders(data_dir)
    
    model = CLVTNet(config)
    trainer = TrainingManager(config, output_dir)
    
    trained_model_path = trainer.train(model, train_loader, val_loader)
    logger.info(f"Training completed. Model saved to: {trained_model_path}")
    
    return trained_model_path

def run_evaluation_only(config_path, model_path, data_dir, output_dir):
    """Execute evaluation phase only."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation-only mode")
    
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    data_pipeline = EEGDataPipeline(config)
    _, val_loader, test_loader = data_pipeline.create_data_loaders(data_dir)
    
    evaluator = EvaluationManager(config, output_dir)
    results = evaluator.comprehensive_evaluation(model_path, test_loader, val_loader)
    
    logger.info("Evaluation completed")
    return results

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.mode == 'pipeline':
            results = run_full_pipeline(args.config, args.data_dir, args.output_dir)
        elif args.mode == 'train':
            results = run_training_only(args.config, args.data_dir, args.output_dir)
        elif args.mode == 'evaluate':
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")
            results = run_evaluation_only(
                args.config, args.model_path, args.data_dir, args.output_dir
            )
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()