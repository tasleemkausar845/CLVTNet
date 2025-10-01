import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import pickle
import os
from scipy.interpolate import griddata
import logging

from src.config import DataConfig

class TopographicMapper:
    """Converts EEG channel data to 2D topographic maps."""
    
    def __init__(self, config: DataConfig):
        """Initialize topographic mapper.
        
        Args:
            config: Data configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load or create electrode positions
        self.electrode_positions = self._initialize_electrode_positions()
        self.grid_x, self.grid_y = self._create_interpolation_grid()
    
    def _initialize_electrode_positions(self) -> np.ndarray:
        """Initialize electrode positions for topographic mapping."""
        # Standard 10-20 electrode positions (64-channel layout)
        # These are projected to 2D using azimuthal equidistant projection
        
        electrode_positions = self._get_standard_1020_positions()
        
        # Project 3D positions to 2D using azimuthal equidistant projection
        positions_2d = []
        for pos_3d in electrode_positions:
            pos_2d = self._azimuthal_equidistant_projection(pos_3d)
            positions_2d.append(pos_2d)
        
        positions_2d = np.array(positions_2d)
        
        self.logger.info(f"Initialized {len(positions_2d)} electrode positions")
        return positions_2d
    
    def _get_standard_1020_positions(self) -> np.ndarray:
        """Get standard 10-20 electrode positions in 3D coordinates."""
        # Standard electrode positions on unit sphere
        # This is a simplified version - in practice, you would load from MNE or similar
        
        positions_3d = []
        
        # Generate electrode positions on unit sphere
        # Using spherical coordinates (phi, theta) then converting to Cartesian
        
        # Frontal electrodes
        positions_3d.extend([
            [0.8, 0.4, 0.4],   # Fp1
            [0.8, -0.4, 0.4],  # Fp2
            [0.6, 0.6, 0.5],   # F7
            [0.7, 0.3, 0.6],   # F3
            [0.8, 0.0, 0.6],   # Fz
            [0.7, -0.3, 0.6],  # F4
            [0.6, -0.6, 0.5],  # F8
        ])
        
        # Central electrodes
        positions_3d.extend([
            [0.3, 0.7, 0.6],   # C3
            [0.0, 0.0, 1.0],   # Cz
            [0.3, -0.7, 0.6],  # C4
        ])
        
        # Parietal electrodes
        positions_3d.extend([
            [-0.7, 0.3, 0.6],  # P3
            [-0.8, 0.0, 0.6],  # Pz
            [-0.7, -0.3, 0.6], # P4
        ])
        
        # Occipital electrodes
        positions_3d.extend([
            [-0.8, 0.4, 0.4],  # O1
            [-0.8, -0.4, 0.4], # O2
        ])
        
        # Temporal electrodes
        positions_3d.extend([
            [0.0, 0.8, 0.6],   # T7
            [0.0, -0.8, 0.6],  # T8
            [-0.3, 0.7, 0.6],  # P7
            [-0.3, -0.7, 0.6], # P8
        ])
        
        # Additional electrodes to reach 64 channels
        # Fill in intermediate positions
        for i in range(64 - len(positions_3d)):
            # Generate intermediate positions with some variation
            theta = 2 * np.pi * i / (64 - len(positions_3d))
            phi = np.pi/4 + 0.3 * np.sin(3 * theta)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            positions_3d.append([x, y, z])
        
        return np.array(positions_3d[:64])  # Ensure exactly 64 electrodes
    
    def _azimuthal_equidistant_projection(self, pos_3d: np.ndarray) -> np.ndarray:
        """Project 3D electrode position to 2D using azimuthal equidistant projection.
        
        Args:
            pos_3d: 3D position on unit sphere [x, y, z]
            
        Returns:
            2D projected position [x, y]
        """
        x, y, z = pos_3d
        
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        azimuth = np.arctan2(y, x)
        
        # Azimuthal equidistant projection
        rho = np.pi/2 - elevation
        
        # Convert back to Cartesian in 2D
        proj_x = rho * np.cos(azimuth)
        proj_y = rho * np.sin(azimuth)
        
        return np.array([proj_x, proj_y])
    
    def _create_interpolation_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create regular grid for interpolation."""
        # Create grid boundaries based on electrode positions
        min_x, max_x = np.min(self.electrode_positions[:, 0]), np.max(self.electrode_positions[:, 0])
        min_y, max_y = np.min(self.electrode_positions[:, 1]), np.max(self.electrode_positions[:, 1])
        
        # Add padding
        padding = 0.1
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        # Create regular grid
        grid_x, grid_y = np.mgrid[
            min_x:max_x:complex(0, self.config.image_size),
            min_y:max_y:complex(0, self.config.image_size)
        ]
        
        return grid_x, grid_y
    
    def eeg_to_topographic_map(self, eeg_data: np.ndarray) -> np.ndarray:
        """Convert EEG channel data to 2D topographic map.
        
        Args:
            eeg_data: EEG data for all channels (time_points, channels) or (channels,)
            
        Returns:
            2D topographic map (height, width)
        """
        # Handle different input shapes
        if eeg_data.ndim == 2:
            # Take mean across time if temporal data provided
            channel_values = np.mean(eeg_data, axis=0)
        else:
            channel_values = eeg_data
        
        # Ensure we have the right number of channels
        num_channels = min(len(channel_values), len(self.electrode_positions))
        channel_values = channel_values[:num_channels]
        electrode_pos = self.electrode_positions[:num_channels]
        
        # Interpolate to regular grid
        topographic_map = griddata(
            electrode_pos,
            channel_values,
            (self.grid_x, self.grid_y),
            method=self.config.interpolation_method,
            fill_value=0.0
        )
        
        # Apply circular mask (head shape)
        topographic_map = self._apply_head_mask(topographic_map)
        
        return topographic_map
    
    def _apply_head_mask(self, topo_map: np.ndarray) -> np.ndarray:
        """Apply circular mask to represent head boundary."""
        center_x, center_y = topo_map.shape[0] // 2, topo_map.shape[1] // 2
        radius = min(center_x, center_y) * 0.9
        
        # Create circular mask
        y, x = np.ogrid[:topo_map.shape[0], :topo_map.shape[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Apply mask
        masked_map = topo_map.copy()
        masked_map[~mask] = 0
        
        return masked_map
    
    def batch_eeg_to_topographic_maps(self, eeg_batch: np.ndarray) -> np.ndarray:
        """Convert batch of EEG data to topographic maps.
        
        Args:
            eeg_batch: Batch of EEG data (batch_size, time_points, channels)
            
        Returns:
            Batch of topographic maps (batch_size, height, width)
        """
        batch_size = eeg_batch.shape[0]
        topo_maps = np.zeros((batch_size, self.config.image_size, self.config.image_size))
        
        for i in range(batch_size):
            topo_maps[i] = self.eeg_to_topographic_map(eeg_batch[i])
        
        return topo_maps
    
    def visualize_topographic_map(self, topo_map: np.ndarray, 
                                save_path: Optional[str] = None,
                                title: str = "EEG Topographic Map") -> None:
        """Visualize topographic map.
        
        Args:
            topo_map: 2D topographic map
            save_path: Path to save visualization
            title: Plot title
        """
        plt.figure(figsize=(8, 8))
        
        # Plot topographic map
        im = plt.imshow(topo_map, cmap='RdBu_r', interpolation='bilinear')
        
        # Add electrode positions
        # Transform electrode positions to image coordinates
        electrode_img_coords = self._transform_positions_to_image_coords(self.electrode_positions)
        plt.scatter(electrode_img_coords[:, 1], electrode_img_coords[:, 0], 
                   c='black', s=20, alpha=0.7)
        
        # Styling
        plt.title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, shrink=0.6, label='Amplitude (μV)')
        plt.axis('off')
        
        # Add head outline
        self._add_head_outline()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _transform_positions_to_image_coords(self, positions: np.ndarray) -> np.ndarray:
        """Transform electrode positions to image coordinates."""
        # Transform from physical coordinates to image pixel coordinates
        min_x, max_x = np.min(self.grid_x), np.max(self.grid_x)
        min_y, max_y = np.min(self.grid_y), np.max(self.grid_y)
        
        img_coords = np.zeros_like(positions)
        img_coords[:, 0] = (positions[:, 0] - min_x) / (max_x - min_x) * (self.config.image_size - 1)
        img_coords[:, 1] = (positions[:, 1] - min_y) / (max_y - min_y) * (self.config.image_size - 1)
        
        return img_coords
    
    def _add_head_outline(self) -> None:
        """Add head outline to topographic plot."""
        # Draw head circle
        center = self.config.image_size / 2
        radius = self.config.image_size * 0.45
        
        theta = np.linspace(0, 2*np.pi, 100)
        head_x = center + radius * np.cos(theta)
        head_y = center + radius * np.sin(theta)
        
        plt.plot(head_x, head_y, 'k-', linewidth=2, alpha=0.8)
        
        # Draw nose
        nose_x = [center, center]
        nose_y = [center - radius, center - radius - 10]
        plt.plot(nose_y, nose_x, 'k-', linewidth=2, alpha=0.8)
        
        # Draw ears
        ear_offset = radius * 0.1
        ear_size = radius * 0.1
        
        # Left ear
        ear_theta = np.linspace(np.pi/2, 3*np.pi/2, 20)
        left_ear_x = center - radius + ear_offset * np.cos(ear_theta)
        left_ear_y = center + ear_size * np.sin(ear_theta)
        plt.plot(left_ear_y, left_ear_x, 'k-', linewidth=2, alpha=0.8)
        
        # Right ear
        right_ear_x = center + radius - ear_offset * np.cos(ear_theta)
        right_ear_y = center + ear_size * np.sin(ear_theta)
        plt.plot(right_ear_y, right_ear_x, 'k-', linewidth=2, alpha=0.8)
    
    def compute_spatial_features(self, topo_map: np.ndarray) -> Dict[str, float]:
        """Compute spatial features from topographic map.
        
        Args:
            topo_map: 2D topographic map
            
        Returns:
            Dictionary of spatial features
        """
        features = {}
        
        # Global statistics
        features['mean_activity'] = np.mean(topo_map)
        features['std_activity'] = np.std(topo_map)
        features['max_activity'] = np.max(topo_map)
        features['min_activity'] = np.min(topo_map)
        
        # Spatial moments
        features['spatial_variance'] = np.var(topo_map)
        features['spatial_skewness'] = self._compute_skewness(topo_map)
        features['spatial_kurtosis'] = self._compute_kurtosis(topo_map)
        
        # Center of mass
        center_of_mass = self._compute_center_of_mass(topo_map)
        features['center_of_mass_x'] = center_of_mass[0]
        features['center_of_mass_y'] = center_of_mass[1]
        
        # Asymmetry features
        features['left_right_asymmetry'] = self._compute_lr_asymmetry(topo_map)
        features['anterior_posterior_asymmetry'] = self._compute_ap_asymmetry(topo_map)
        
        # Spatial complexity
        features['spatial_complexity'] = self._compute_spatial_complexity(topo_map)
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        from scipy.stats import skew
        return skew(data.flatten())
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        from scipy.stats import kurtosis
        return kurtosis(data.flatten())
    
    def _compute_center_of_mass(self, topo_map: np.ndarray) -> Tuple[float, float]:
        """Compute center of mass of topographic map."""
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:topo_map.shape[0], 0:topo_map.shape[1]]
        
        # Compute center of mass
        total_mass = np.sum(np.abs(topo_map))
        if total_mass > 0:
            center_x = np.sum(x_coords * np.abs(topo_map)) / total_mass
            center_y = np.sum(y_coords * np.abs(topo_map)) / total_mass
        else:
            center_x = topo_map.shape[1] / 2
            center_y = topo_map.shape[0] / 2
        
        return center_x, center_y
    
    def _compute_lr_asymmetry(self, topo_map: np.ndarray) -> float:
        """Compute left-right asymmetry."""
        mid_point = topo_map.shape[1] // 2
        left_activity = np.mean(topo_map[:, :mid_point])
        right_activity = np.mean(topo_map[:, mid_point:])
        
        if abs(left_activity + right_activity) > 1e-10:
            asymmetry = (left_activity - right_activity) / (left_activity + right_activity)
        else:
            asymmetry = 0.0
        
        return asymmetry
    
    def _compute_ap_asymmetry(self, topo_map: np.ndarray) -> float:
        """Compute anterior-posterior asymmetry."""
        mid_point = topo_map.shape[0] // 2
        anterior_activity = np.mean(topo_map[:mid_point, :])
        posterior_activity = np.mean(topo_map[mid_point:, :])
        
        if abs(anterior_activity + posterior_activity) > 1e-10:
            asymmetry = (anterior_activity - posterior_activity) / (anterior_activity + posterior_activity)
        else:
            asymmetry = 0.0
        
        return asymmetry
    
    def _compute_spatial_complexity(self, topo_map: np.ndarray) -> float:
        """Compute spatial complexity using gradient magnitude."""
        # Compute gradients
        grad_x = np.gradient(topo_map, axis=1)
        grad_y = np.gradient(topo_map, axis=0)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return mean gradient magnitude as complexity measure
        return np.mean(gradient_magnitude)
    
    def save_electrode_positions(self, save_path: str) -> None:
        """Save electrode positions to file."""
        electrode_data = {
            'positions_2d': self.electrode_positions,
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'config': {
                'image_size': self.config.image_size,
                'interpolation_method': self.config.interpolation_method,
                'electrode_montage': self.config.electrode_montage
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(electrode_data, f)
        
        self.logger.info(f"Electrode positions saved to {save_path}")
    
    def load_electrode_positions(self, load_path: str) -> None:
        """Load electrode positions from file."""
        with open(load_path, 'rb') as f:
            electrode_data = pickle.load(f)
        
        self.electrode_positions = electrode_data['positions_2d']
        self.grid_x = electrode_data['grid_x']
        self.grid_y = electrode_data['grid_y']
        
        self.logger.info(f"Electrode positions loaded from {load_path}")
    
    def create_electrode_layout_visualization(self, save_path: Optional[str] = None) -> None:
        """Create visualization of electrode layout."""
        plt.figure(figsize=(10, 8))
        
        # Plot electrode positions
        plt.scatter(self.electrode_positions[:, 0], self.electrode_positions[:, 1], 
                   c='red', s=100, alpha=0.8, edgecolors='black')
        
        # Label electrodes
        for i, pos in enumerate(self.electrode_positions):
            plt.annotate(f'Ch{i+1}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
        
        # Add grid
        plt.scatter(self.grid_x.flatten(), self.grid_y.flatten(), 
                   c='lightblue', s=1, alpha=0.3)
        
        plt.title('EEG Electrode Layout and Interpolation Grid', fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()