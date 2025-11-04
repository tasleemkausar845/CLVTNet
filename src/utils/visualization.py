import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

class VisualizationManager:
    """Comprehensive visualization utilities for model evaluation."""
    
    def __init__(self, config, output_dir: str):
        """Initialize visualization manager.
        
        Args:
            config: Configuration object
            output_dir: Directory to save visualizations
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str, 
                            class_names: Optional[List[str]] = None):
        """Plot confusion matrix with proper formatting."""
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also plot raw counts
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        raw_save_path = save_path.replace('.png', '_raw.png')
        plt.savefig(raw_save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, test_results: Dict[str, Any], save_path: str):
        """Plot ROC curve for binary classification."""
        from sklearn.metrics import roc_curve, auc
        
        y_true = np.array(test_results['targets'])
        y_prob = np.array(test_results['probabilities'])
        
        if len(np.unique(y_true)) != 2:
            print("ROC curve only available for binary classification")
            return
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, test_results: Dict[str, Any], save_path: str):
        """Plot Precision-Recall curve for binary classification."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        y_true = np.array(test_results['targets'])
        y_prob = np.array(test_results['probabilities'])
        
        if len(np.unique(y_true)) != 2:
            print("PR curve only available for binary classification")
            return
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_maps(self, features: torch.Tensor, title: str = "Feature Maps", 
                         save_path: str = None, max_channels: int = 16):
        """Plot feature maps from intermediate layers."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
        
        # Handle different tensor shapes
        if features.ndim == 4:  # (B, C, H, W)
            features = features[0]  # Take first batch
        
        num_channels = min(features.shape[0], max_channels)
        cols = 4
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_channels):
            row = i // cols
            col = i % cols
            
            im = axes[row, col].imshow(features[i], cmap='viridis')
            axes[row, col].set_title(f'Channel {i}', fontsize=10)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.6)
        
        # Hide empty subplots
        for i in range(num_channels, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_training_history(self, history: Dict[str, List], save_path: str):
        """Plot training history curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training summary statistics
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        summary_text = f"""Training Summary:
        Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})
        Final Training Loss: {final_train_loss:.4f}
        Final Validation Loss: {final_val_loss:.4f}
        Total Epochs: {len(epochs)}"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].axis('off')
        
        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, comparison_results: Dict[str, Dict], save_path: str):
        """Plot comparison between multiple models."""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model_names = list(comparison_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[name][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, values, alpha=0.7, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(model_names) > 3:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_weights(self, attention_weights: torch.Tensor, save_path: str,
                             input_image: Optional[torch.Tensor] = None):
        """Visualize attention weights."""
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().detach().numpy()
        
        if attention_weights.ndim == 4:  # (B, H, W, W) - multi-head attention
            attention_weights = attention_weights[0].mean(axis=0)  # Average over heads
        elif attention_weights.ndim == 3:  # (H, W, W)
            attention_weights = attention_weights.mean(axis=0)
        
        fig, axes = plt.subplots(1, 2 if input_image is not None else 1, 
                                figsize=(12 if input_image is not None else 6, 5))
        
        if input_image is not None:
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().detach().numpy()
            
            if input_image.ndim == 4:
                input_image = input_image[0, 0]  # Take first batch, first channel
            elif input_image.ndim == 3:
                input_image = input_image[0]
            
            axes[0].imshow(input_image, cmap='gray')
            axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            im = axes[1].imshow(attention_weights, cmap='hot', interpolation='bilinear')
            axes[1].set_title('Attention Weights', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], shrink=0.6)
        else:
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            im = axes[0].imshow(attention_weights, cmap='hot', interpolation='bilinear')
            axes[0].set_title('Attention Weights', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            plt.colorbar(im, ax=axes[0], shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution(self, y_true: np.ndarray, save_path: str,
                              class_names: Optional[List[str]] = None):
        """Plot class distribution in the dataset."""
        unique_classes, counts = np.unique(y_true, return_counts=True)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in unique_classes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, counts, alpha=0.7, color=plt.cm.Set3(np.arange(len(unique_classes))))
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create percentage plot
        percentages = counts / np.sum(counts) * 100
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, percentages, alpha=0.7, color=plt.cm.Set3(np.arange(len(unique_classes))))
        
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(percentages) * 0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Percentage of Samples')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        percentage_save_path = save_path.replace('.png', '_percentage.png')
        plt.savefig(percentage_save_path, dpi=300, bbox_inches='tight')
        plt.close()