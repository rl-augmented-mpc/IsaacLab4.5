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

@dataclass
class InnerCircularSampler:
    radius_range: List[float] | Tuple[float, float]
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert len(self.radius_range) == 2
        assert len(self.z_range) == 2
    
    def sample(self, center: np.ndarray, num_samples: int) -> list[tuple[float, float, float]]:
        radius = np.random.uniform(self.radius_range[0], self.radius_range[1], num_samples)
        # Evenly spaced angles in [0, 2π)
        theta = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        # Random global shift
        shift = np.random.uniform(0, 2*np.pi)
        theta = theta + shift

        x_vals = radius * np.cos(theta) + center[..., 0]
        y_vals = radius * np.sin(theta) + center[..., 1]
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


### Shape sampler ###

@dataclass
class CircularTrajectorySampler:
    radius: float # radius of circle [m]
    omega: float # speed of rotation [rad/s]
    
    def sample(self, step:int, num_samples: int) -> list[tuple[float, float, float]]:
        vx = self.radius * self.omega * np.ones(num_samples)
        vy = np.zeros(num_samples)
        wz = self.omega * np.ones(num_samples)
        command = np.stack((vx, vy, wz), axis=1).tolist()
        return command


@dataclass
class InfinityTrajectorySampler:
    a: float         # scale factor for the infinity shape [m]
    dt: float        # time step between samples [s]
    total_s: float   # total length of time to complete the trajectory [s]
    
    def __post_init__(self):
        self.t = np.linspace(0, 2*np.pi, int(self.total_s / self.dt))
        self.traj_index = 0
        self.precompute_trajectory()
    
    def precompute_trajectory(self):
        # Gerono lemniscate parameterization (figure-eight shape)
        # x(t) = a * cos(t)
        # y(t) = a * sin(t) * cos(t)
        x = self.a * np.cos(self.t)
        y = self.a * np.sin(self.t) * np.cos(self.t)
        
        # Compute first derivatives: these represent the global velocity components
        dx_dt = np.gradient(x, self.dt)
        dy_dt = np.gradient(y, self.dt)
        
        # Compute the forward speed as the magnitude of the velocity vector
        v = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Compute second derivatives (needed for curvature)
        ddx_dt2 = np.gradient(dx_dt, self.dt)
        ddy_dt2 = np.gradient(dy_dt, self.dt)
        
        # Compute curvature kappa at each time step:
        #   kappa = (dx/dt * d²y/dt² - dy/dt * d²x/dt²) / (v**3)
        curvature = (dx_dt * ddy_dt2 - dy_dt * ddx_dt2) / (v**3 + 1e-8)
        
        # The yaw rate command is given by the curvature times the forward speed
        w_z = curvature * v
        
        # Return the commands as a list of (v_x, w_z) tuples in the local frame
        self.commands = np.stack((v, w_z), axis=1)

    def sample(self, step:int, num_samples: int) -> List[Tuple[float, float, float]]:
        v = self.commands[self.traj_index, 0] * np.ones(num_samples)
        w_z = self.commands[self.traj_index, 1] * np.ones(num_samples)
        vy = np.zeros(num_samples)
        commands = np.stack((v, vy, w_z), axis=1).tolist()
        self.traj_index += 1
        return commands


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dt = 0.01
    total_s = 20.0
    num_samples = int(total_s / dt)
    
    circular_command = np.zeros((num_samples, 2))
    infinity_command = np.zeros((num_samples, 2))
    
    circular_sampler = CircularTrajectorySampler(radius=2.0, omega=0.5)
    infinity_sampler = InfinityTrajectorySampler(a=2.0, dt=dt, total_s=total_s)
    
    # plot circular trajectory
    for t in range(num_samples):
        command = circular_sampler.sample(0,1)
        command = np.array(command)
        circular_command[t] = command[:, [0,2]]
        
        command = infinity_sampler.sample(0,1)
        command = np.array(command)
        infinity_command[t] = command[:, [0,2]]
        
    # plot circular trajectory
    vx = circular_command[:, 0]
    wz = circular_command[:, 1]
    angle = np.cumsum(wz*dt)
    velocity_world = np.zeros((num_samples, 2))
    velocity_world[:, 0] = vx * np.cos(angle)
    velocity_world[:, 1] = vx * np.sin(angle)
    position = np.cumsum(velocity_world * dt, axis=0)
    plt.quiver(position[:, 0], position[:, 1], velocity_world[:, 0], velocity_world[:, 1], color='b', label='circular')
    
    # plot inf trajectory
    vx = infinity_command[:, 0]
    wz = infinity_command[:, 1]
    angle = np.cumsum(wz*dt)
    velocity_world = np.zeros((num_samples, 2))
    velocity_world[:, 0] = vx * np.cos(angle)
    velocity_world[:, 1] = vx * np.sin(angle)
    position = np.cumsum(velocity_world * dt, axis=0)
    plt.quiver(position[:, 0], position[:, 1], velocity_world[:, 0], velocity_world[:, 1], color='r', label='infinity')
    
    plt.axis('equal')
    plt.grid()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.show()