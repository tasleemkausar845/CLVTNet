import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.models.clvtnet import CLVTNet
from src.utils.eeg_visualization import EEGVisualizer

class EvaluationManager:
    """Comprehensive model evaluation system with visualization capabilities."""
    
    def __init__(self, config: Config, output_dir: str):
        """Initialize evaluation manager."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize visualizer
        self.visualizer = EEGVisualizer(config, str(self.output_dir))
        
    def comprehensive_evaluation(self, model_path: str, test_loader: DataLoader,
                               val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Execute comprehensive model evaluation with visualizations."""
        self.logger.info("Starting comprehensive model evaluation")
        
        model = self._load_model(model_path)
        
        test_results = self._evaluate_metrics(model, test_loader, "test")
        
        if val_loader is not None:
            val_results = self._evaluate_metrics(model, val_loader, "validation")
        else:
            val_results = {}
        
        # Generate visualizations if enabled
        if self.config.evaluation.generate_visualizations:
            self._generate_visualizations(model, test_loader, test_results)
        
        evaluation_results = {
            'test_performance': test_results,
            'validation_performance': val_results,
            'model_info': self._get_model_info(model)
        }
        
        self._save_results(evaluation_results)
        
        self.logger.info("Evaluation completed")
        return evaluation_results
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = CLVTNet(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _evaluate_metrics(self, model: nn.Module, data_loader: DataLoader,
                         split_name: str) -> Dict[str, Any]:
        """Evaluate core classification metrics."""
        self.logger.info(f"Evaluating metrics on {split_name} set")
        
        all_predictions = []
        all_targets = []
        all_features = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Extract features for visualization
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(inputs)
                    all_features.append(features.cpu().numpy())
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        
        results = {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'num_samples': len(y_true),
            'predictions': y_pred.tolist(),
            'targets': y_true.tolist()
        }
        
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        if self.config.model.num_classes > 2:
            results['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
            results['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
            results['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        if all_features:
            results['features'] = np.concatenate(all_features, axis=0)
        
        self.logger.info(f"{split_name.capitalize()} Results:")
        self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"  Precision: {results['precision']:.4f}")
        self.logger.info(f"  Recall: {results['recall']:.4f}")
        self.logger.info(f"  F1-Score: {results['f1_score']:.4f}")
        
        return results
    
    def _generate_visualizations(self, model: nn.Module, test_loader: DataLoader, 
                               test_results: Dict[str, Any]):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating evaluation visualizations")
        
        # Confusion matrix
        cm = np.array(test_results['confusion_matrix'])
        self.visualizer.plot_confusion_matrix(
            cm, 
            save_path=str(self.output_dir / "confusion_matrix.png"),
            class_names=['Attend Left', 'Attend Right']
        )
        
        # Model predictions visualization
        self.visualizer.visualize_model_predictions(
            model, 
            test_loader,
            save_path=str(self.output_dir / "model_predictions.png"),
            num_samples=8
        )
        
        # Feature maps visualization
        self._visualize_feature_maps(model, test_loader)
        
        # Attention weights visualization
        self._visualize_attention_weights(model, test_loader)
        
        # Training history visualization
        self._visualize_training_history()
        
        # Sample topographic maps
        self._visualize_sample_topographic_maps(test_loader)
    
    def _visualize_feature_maps(self, model: nn.Module, test_loader: DataLoader):
        """Visualize intermediate feature maps."""
        sample_batch = next(iter(test_loader))
        sample_input = sample_batch[0][:1].to(self.device)
        
        with torch.no_grad():
            features = model.conv_stem(sample_input)
            self.visualizer.plot_feature_maps(
                features,
                title="ConvModule Feature Maps",
                save_path=str(self.output_dir / "feature_maps_conv.png")
            )
            
            for i, stage in enumerate(model.stages):
                features = stage(features)
                self.visualizer.plot_feature_maps(
                    features,
                    title=f"Stage {i+1} Feature Maps",
                    save_path=str(self.output_dir / f"feature_maps_stage_{i+1}.png")
                )
    
    def _visualize_attention_weights(self, model: nn.Module, test_loader: DataLoader):
        """Visualize attention weights from SPEN blocks."""
        sample_batch = next(iter(test_loader))
        sample_input = sample_batch[0][:1].to(self.device)
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'global_branch'):
                attention_weights.append(output)
        
        handles = []
        for stage in model.stages:
            for block in stage:
                if hasattr(block, 'scsa'):
                    handle = block.scsa.register_forward_hook(attention_hook)
                    handles.append(handle)
        
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Visualize captured attention weights
        for i, attn_weights in enumerate(attention_weights):
            if attn_weights is not None:
                self.visualizer.plot_attention_weights(
                    attn_weights,
                    input_image=sample_input,
                    save_path=str(self.output_dir / f"attention_weights_stage_{i+1}.png")
                )
    
    def _visualize_training_history(self):
        """Visualize training history if available."""
        model_dir = self.output_dir.parent
        checkpoint_path = model_dir / "best_model.pth"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'training_history' in checkpoint:
                self.visualizer.plot_training_curves(
                    checkpoint['training_history'],
                    save_path=str(self.output_dir / "training_history.png")
                )
    
    def _visualize_sample_topographic_maps(self, test_loader: DataLoader):
        """Visualize sample topographic maps."""
        sample_batch = next(iter(test_loader))
        sample_inputs = sample_batch[0][:4]
        sample_targets = sample_batch[1][:4]
        
        for i, (input_map, target) in enumerate(zip(sample_inputs, sample_targets)):
            input_map = input_map.squeeze().numpy()
            label = "Attend Left" if target.item() == 0 else "Attend Right"
            
            self.visualizer.plot_topographic_map(
                input_map,
                title=f"Sample {i+1}: {label}",
                save_path=str(self.output_dir / f"topographic_sample_{i+1}.png")
            )
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information and complexity."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }
        
        # Add model complexity if available
        if hasattr(model, 'compute_model_complexity'):
            complexity = model.compute_model_complexity()
            model_info.update(complexity)
        
        return model_info
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        
        # Convert numpy arrays to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj
        
        serializable_results = make_serializable(results)
        
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self._save_csv_summary(results)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _save_csv_summary(self, results: Dict[str, Any]):
        """Save results summary as CSV."""
        
        if 'test_performance' in results:
            test_data = []
            test_perf = results['test_performance']
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'loss']
            for metric in metrics:
                if metric in test_perf:
                    test_data.append({
                        'metric': metric,
                        'value': test_perf[metric],
                        'split': 'test'
                    })
            
            if results.get('validation_performance'):
                val_perf = results['validation_performance']
                for metric in metrics:
                    if metric in val_perf:
                        test_data.append({
                            'metric': metric,
                            'value': val_perf[metric],
                            'split': 'validation'
                        })
            
            df = pd.DataFrame(test_data)
            df.to_csv(self.output_dir / "metrics_summary.csv", index=False)
    
    def quick_evaluate(self, model_path: str, test_loader: DataLoader) -> Dict[str, float]:
        """Quick evaluation returning only core metrics as floats."""
        model = self._load_model(model_path)
        results = self._evaluate_metrics(model, test_loader, "test")
        
        return {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        }
    
    def compare_models(self, model_paths: List[str], test_loader: DataLoader,
                      model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple models on core metrics."""
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
        
        self.logger.info(f"Comparing {len(model_paths)} models")
        
        comparison_results = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            self.logger.info(f"Evaluating {model_name}")
            metrics = self.quick_evaluate(model_path, test_loader)
            comparison_results[model_name] = metrics
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        best_models = {}
        
        for metric in metrics:
            best_model = max(comparison_results.keys(), 
                           key=lambda k: comparison_results[k][metric])
            best_value = comparison_results[best_model][metric]
            best_models[metric] = {
                'model': best_model,
                'value': best_value
            }
        
        final_results = {
            'model_results': comparison_results,
            'best_models': best_models
        }
        
        comparison_path = self.output_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        comparison_data = []
        for model_name, metrics_dict in comparison_results.items():
            for metric, value in metrics_dict.items():
                comparison_data.append({
                    'model': model_name,
                    'metric': metric,
                    'value': value
                })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(self.output_dir / "model_comparison.csv", index=False)
        
        return final_results