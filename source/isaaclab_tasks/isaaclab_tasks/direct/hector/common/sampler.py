from typing import Tuple, List
from dataclasses import dataclass
import numpy as np


### Uniform ###

@dataclass
class UniformLineSampler:
    x_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert len(self.x_range) == 2
    
    def sample(self, num_samples: int) -> list[float]:
        points = np.random.uniform(self.x_range[0], self.x_range[1], num_samples).tolist()
        return points

@dataclass
class QuaternionSampler:
    x_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert len(self.x_range) == 2
    
    def sample(self, num_samples: int) -> list[tuple[float, float, float, float]]:
        points = np.random.uniform(self.x_range[0], self.x_range[1], num_samples)
        quat = np.stack([np.cos(points/2), np.zeros_like(points), np.zeros_like(points), np.sin(points/2)], axis=-1).tolist()
        return quat

@dataclass
class UniforPlaneSampler:
    x_range: List[float] | Tuple[float, float]
    y_range: List[float] | Tuple[float, float]
    
    
    def __post_init__(self):
        assert len(self.x_range) == 2
        assert len(self.y_range) == 2
    
    def sample(self, num_samples: int) -> list[tuple[float, float]]:
        x_vals = np.random.uniform(self.x_range[0], self.x_range[1], num_samples)
        y_vals = np.random.uniform(self.y_range[0], self.y_range[1], num_samples)
        
        points = np.stack((x_vals, y_vals), axis=-1).tolist()
        return points

@dataclass
class UniformCubicSampler:
    x_range: List[float] | Tuple[float, float]
    y_range: List[float] | Tuple[float, float]
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert len(self.x_range) == 2
        assert len(self.y_range) == 2
        assert len(self.z_range) == 2
    
    def sample(self, num_samples: int) -> list[tuple[float, float, float]]:
        x_vals = np.random.uniform(self.x_range[0], self.x_range[1], num_samples)
        y_vals = np.random.uniform(self.y_range[0], self.y_range[1], num_samples)
        z_vals = np.random.uniform(self.z_range[0], self.z_range[1], num_samples)
        
        points = np.stack((x_vals, y_vals, z_vals), axis=-1).tolist()
        return points

@dataclass
class CircularSampler:
    radius: float
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert self.radius > 0
        assert len(self.z_range) == 2
    
    def sample(self, center: np.ndarray, num_samples: int) -> list[tuple[float, float, float]]:
        # Evenly spaced angles in [0, 2π)
        theta = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        # Random global shift
        shift = np.random.uniform(0, 2*np.pi)
        theta = theta + shift

        x_vals = self.radius * np.cos(theta) + center[..., 0]
        y_vals = self.radius * np.sin(theta) + center[..., 1]
        z_vals = np.random.uniform(self.z_range[0], self.z_range[1], num_samples)

        points = np.stack((x_vals, y_vals, z_vals), axis=-1).tolist()
        return points

### Grid ###

@dataclass
class GridCubicSampler:
    x_range: List[float] | Tuple[float, float]
    y_range: List[float] | Tuple[float, float]
    z_range: List[float] | Tuple[float, float]
    x_resolution: float
    y_resolution: float
    z_resolution: float
    
    def __post_init__(self):
        assert len(self.x_range) == 2
        assert len(self.y_range) == 2
        assert len(self.z_range) == 2
    
    def sample(self) -> list[tuple[float, float, float]]:
        x_vals = np.arange(self.x_range[0], self.x_range[1], self.x_resolution)
        y_vals = np.arange(self.y_range[0], self.y_range[1], self.y_resolution)
        z_vals = np.arange(self.z_range[0], self.z_range[1], self.z_resolution)
        
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
        
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        
        points = np.stack((X_flat, Y_flat, Z_flat), axis=-1).tolist()
        return points