import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

class EEGVisualizer:
    """EEG data and model visualization utilities."""
    
    def __init__(self, config, output_dir: str):
        """Initialize EEG visualizer."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_topographic_map(self, topo_map: np.ndarray, title: str = "EEG Topographic Map",
                           save_path: Optional[str] = None, show_electrodes: bool = True):
        """Plot EEG topographic map."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        im = ax.imshow(topo_map, cmap='RdBu_r', interpolation='bilinear')
        
        if show_electrodes:
            electrode_positions = self._get_electrode_positions()
            ax.scatter(electrode_positions[:, 1], electrode_positions[:, 0], 
                      c='black', s=20, alpha=0.7, marker='o')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6, label='Amplitude (μV)')
        ax.axis('off')
        
        self._add_head_outline(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectral_analysis(self, eeg_data: np.ndarray, save_path: Optional[str] = None):
        """Plot power spectral density analysis."""
        from scipy.signal import welch
        
        frequencies, psd = welch(
            eeg_data, 
            fs=self.config.data.sampling_rate, 
            axis=0, 
            nperseg=min(eeg_data.shape[0], 4 * self.config.data.sampling_rate)
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall PSD
        mean_psd = np.mean(psd, axis=1)
        axes[0, 0].semilogy(frequencies, mean_psd, 'b-', linewidth=2)
        axes[0, 0].set_title('Power Spectral Density', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Power (μV²/Hz)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, 50)
        
        # Alpha band analysis
        alpha_low, alpha_high = self.config.data.frequency_bands['alpha']
        alpha_mask = (frequencies >= alpha_low) & (frequencies <= alpha_high)
        alpha_psd = np.mean(psd[alpha_mask, :], axis=0)
        
        axes[0, 1].bar(range(len(alpha_psd)), alpha_psd, alpha=0.7)
        axes[0, 1].set_title('Alpha Band Power Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Alpha Power')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency band comparison
        bands = self.config.data.frequency_bands
        band_powers = []
        band_names = []
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            band_power = np.mean(psd[band_mask, :])
            band_powers.append(band_power)
            band_names.append(band_name)
        
        axes[1, 0].bar(band_names, band_powers, alpha=0.7)
        axes[1, 0].set_title('Frequency Band Powers', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Average Power')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Topographic power map
        channel_powers = np.mean(psd, axis=0)
        im = axes[1, 1].imshow(channel_powers.reshape(8, 8), cmap='viridis', interpolation='bilinear')
        axes[1, 1].set_title('Channel Power Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], shrink=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_model_predictions(self, model, data_loader, save_path: Optional[str] = None, 
                                  num_samples: int = 8):
        """Visualize model predictions on sample data."""
        model.eval()
        samples_visualized = 0
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                if samples_visualized >= num_samples:
                    break
                
                inputs = inputs.to(next(model.parameters()).device)
                outputs = model(inputs)
                predictions = torch.softmax(outputs, dim=1)
                
                for i in range(min(inputs.size(0), num_samples - samples_visualized)):
                    input_img = inputs[i, 0].cpu().numpy()
                    true_label = targets[i].item()
                    pred_probs = predictions[i].cpu().numpy()
                    pred_label = pred_probs.argmax()
                    confidence = pred_probs.max()
                    
                    axes[samples_visualized].imshow(input_img, cmap='RdBu_r')
                    
                    correct = "✓" if true_label == pred_label else "✗"
                    title = f"{correct} True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}"
                    axes[samples_visualized].set_title(title, fontsize=10)
                    axes[samples_visualized].axis('off')
                    
                    samples_visualized += 1
                    if samples_visualized >= num_samples:
                        break
        
        plt.suptitle('Model Predictions on Sample Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _get_electrode_positions(self) -> np.ndarray:
        """Get standard electrode positions for visualization."""
        positions = []
        
        # Standard 64-electrode positions in image coordinates
        for i in range(64):
            angle = 2 * np.pi * i / 64
            radius = 0.4 * self.config.data.image_size
            center = self.config.data.image_size / 2
            
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            positions.append([x, y])
        
        return np.array(positions)
    
    def _add_head_outline(self, ax):
        """Add head outline to topographic plot."""
        center = self.config.data.image_size / 2
        radius = self.config.data.image_size * 0.45
        
        theta = np.linspace(0, 2*np.pi, 100)
        head_x = center + radius * np.cos(theta)
        head_y = center + radius * np.sin(theta)
        
        ax.plot(head_x, head_y, 'k-', linewidth=2, alpha=0.8)
        
        # Nose
        nose_x = [center, center]
        nose_y = [center - radius, center - radius - 10]
        ax.plot(nose_y, nose_x, 'k-', linewidth=2, alpha=0.8)
        
        # Ears
        ear_offset = radius * 0.1
        ear_size = radius * 0.1
        
        ear_theta = np.linspace(np.pi/2, 3*np.pi/2, 20)
        left_ear_x = center - radius + ear_offset * np.cos(ear_theta)
        left_ear_y = center + ear_size * np.sin(ear_theta)
        ax.plot(left_ear_y, left_ear_x, 'k-', linewidth=2, alpha=0.8)
        
        right_ear_x = center + radius - ear_offset * np.cos(ear_theta)
        right_ear_y = center + ear_size * np.sin(ear_theta)
        ax.plot(right_ear_y, right_ear_x, 'k-', linewidth=2, alpha=0.8)
    
    def plot_feature_maps(self, features: torch.Tensor, title: str = "Feature Maps", 
                         save_path: Optional[str] = None, max_channels: int = 16):
        """Plot feature maps from intermediate layers."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
        
        if features.ndim == 4:
            features = features[0]
        
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
    
    def plot_attention_weights(self, attention_weights: torch.Tensor, 
                             input_image: Optional[torch.Tensor] = None,
                             save_path: Optional[str] = None):
        """Visualize attention weights."""
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().detach().numpy()
        
        if attention_weights.ndim == 4:
            attention_weights = attention_weights[0].mean(axis=0)
        elif attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)
        
        fig, axes = plt.subplots(1, 2 if input_image is not None else 1, 
                                figsize=(12 if input_image is not None else 6, 5))
        
        if input_image is not None:
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().detach().numpy()
            
            if input_image.ndim == 4:
                input_image = input_image[0, 0]
            elif input_image.ndim == 3:
                input_image = input_image[0]
            
            axes[0].imshow(input_image, cmap='gray')
            axes[0].set_title('Input Topographic Map', fontsize=12, fontweight='bold')
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str, 
                            class_names: Optional[List[str]] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = ['Attend Left', 'Attend Right']
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, history: Dict[str, List], save_path: str):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
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
    
    def plot_eeg_channels(self, eeg_data: np.ndarray, channel_names: Optional[List[str]] = None,
                         title: str = "EEG Channels", save_path: Optional[str] = None,
                         max_channels: int = 16):
        """Plot raw EEG channel data."""
        if eeg_data.ndim == 2:
            time_points, num_channels = eeg_data.shape
        else:
            raise ValueError("EEG data should be 2D (time_points, channels)")
        
        num_channels_plot = min(num_channels, max_channels)
        time = np.arange(time_points) / self.config.data.sampling_rate
        
        fig, axes = plt.subplots(num_channels_plot, 1, figsize=(12, 2 * num_channels_plot))
        if num_channels_plot == 1:
            axes = [axes]
        
        for i in range(num_channels_plot):
            axes[i].plot(time, eeg_data[:, i], 'b-', linewidth=0.8)
            
            if channel_names and i < len(channel_names):
                axes[i].set_ylabel(channel_names[i])
            else:
                axes[i].set_ylabel(f'Ch {i+1}')
            
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, time[-1])
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)