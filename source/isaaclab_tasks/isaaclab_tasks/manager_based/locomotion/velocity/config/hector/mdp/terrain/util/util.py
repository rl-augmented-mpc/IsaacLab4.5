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
    
## Position sampler ##

@dataclass
class CircularSampler:
    radius: float
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert self.radius >= 0
        assert len(self.z_range) == 2
    
    def sample(self, center: np.ndarray, num_samples: int) -> list[tuple[float, float, float]]:
        theta = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        shift = np.random.uniform(0, 2*np.pi)
        theta = theta + shift

        x_vals = self.radius * np.cos(theta) + center[..., 0]
        y_vals = self.radius * np.sin(theta) + center[..., 1]
        z_vals = np.random.uniform(self.z_range[0], self.z_range[1], num_samples)

        points = np.stack((x_vals, y_vals, z_vals), axis=-1).tolist()
        return points

@dataclass
class CircularSamplerWithLimit:
    radius: float
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert self.radius >= 0
        assert len(self.z_range) == 2
    
    def sample(self, center: np.ndarray, num_samples: int) -> list[tuple[float, float, float]]:
        coord_mask = np.random.randint(0, 2, num_samples)
        theta = np.random.uniform(-np.pi/3, np.pi/3, num_samples)*coord_mask + np.random.uniform(np.pi - np.pi/3, np.pi + np.pi/3, num_samples)*(1-coord_mask)

        x_vals = self.radius * np.cos(theta) + center[..., 0]
        y_vals = self.radius * np.sin(theta) + center[..., 1]
        z_vals = np.random.uniform(self.z_range[0], self.z_range[1], num_samples)

        points = np.stack((x_vals, y_vals, z_vals), axis=-1).tolist()
        return points

@dataclass
class SquareSampler:
    radius: float
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert self.radius > 0, "Radius must be positive"
        assert len(self.z_range) == 2, "z_range must have exactly two elements"
    
    def sample(self, center: np.ndarray, num_samples: int) -> list[tuple[float, float, float]]:
        """
        Sample num_samples points along the square edge.
        The square is centered at 'center' with side length = 2*radius.
        Each point gets a z coordinate uniformly sampled from z_range.
        """
        perimeter = 8 * self.radius
        # Evenly spaced distances along the perimeter
        distances = np.linspace(0, perimeter, num_samples, endpoint=False).astype(np.float32)
        # Apply a random global shift to "rotate" the starting point along the perimeter
        shift = np.random.uniform(0, perimeter)
        distances = (distances + shift) % perimeter
        
        points = []
        for d in distances:
            # Bottom edge: from left to right
            if d < 2 * self.radius:
                x = center[0] - self.radius + d
                y = center[1] - self.radius
            # Right edge: from bottom to top
            elif d < 4 * self.radius:
                d_prime = d - 2 * self.radius
                x = center[0] + self.radius
                y = center[1] - self.radius + d_prime
            # Top edge: from right to left
            elif d < 6 * self.radius:
                d_prime = d - 4 * self.radius
                x = center[0] + self.radius - d_prime
                y = center[1] + self.radius
            # Left edge: from top to bottom
            else:
                d_prime = d - 6 * self.radius
                x = center[0] - self.radius
                y = center[1] + self.radius - d_prime
            
            z = np.random.uniform(self.z_range[0], self.z_range[1])
            points.append((x, y, z))
        
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

## orientation sampler ##

@dataclass
class EulertoQuaternionSampler:
    x_range: List[float] | Tuple[float, float]
    y_range: List[float] | Tuple[float, float]
    z_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert len(self.x_range) == 2
        assert len(self.y_range) == 2
        assert len(self.z_range) == 2
    
    def sample(self, num_samples: int) -> list[tuple[float, float, float, float]]:
        roll = np.random.uniform(self.x_range[0], self.x_range[1], num_samples)
        pitch = np.random.uniform(self.y_range[0], self.y_range[1], num_samples)
        yaw = np.random.uniform(self.z_range[0], self.z_range[1], num_samples)
        
        # Convert Euler angles to quaternion
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2) # type: ignore
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        quat = np.stack((qw, qx, qy, qz), axis=-1).tolist()
        return quat

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
class CircularOrientationSampler:
    x_range: List[float] | Tuple[float, float]
    
    def __post_init__(self):
        assert len(self.x_range) == 2
    
    def sample(self, positions: np.ndarray, num_samples: int) -> list[tuple[float, float, float, float]]:
        assert positions.shape[0] == num_samples
        
        yaw = np.arctan2(positions[:, 1], positions[:, 0])
        yaw_delta = np.random.uniform(self.x_range[0], self.x_range[1], num_samples)
        yaw = yaw + yaw_delta
        
        quat = np.stack([np.cos(yaw/2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw/2)], axis=-1).tolist()
        return quat

@dataclass
class BinaryOrientationSampler:
    def sample(self, positions: np.ndarray, num_samples: int) -> list[tuple[float, float, float, float]]:
        assert positions.shape[0] == num_samples
        
        yaw = np.zeros(positions.shape[0])
        yaw[positions[:, 0] < 0] = 0
        yaw[positions[:, 0] > 0] = np.pi
        
        quat = np.stack([np.cos(yaw/2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw/2)], axis=-1).tolist()
        return quat

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

# @dataclass
# class InfinityTrajectorySampler:
#     a: float         # scale factor for the infinity shape [m]
#     dt: float        # time step between samples [s]
#     total_s: float   # total length of time to complete the trajectory [s]
    
#     def __post_init__(self):
#         self.delta_t = 0.5
#         self.t = np.linspace(-self.delta_t, 2*np.pi, int(self.total_s / self.dt))
#         self.traj_index = 0
#         self.precompute_trajectory()
    
#     def precompute_trajectory(self):
#         t_1 = self.t[self.t < 0]
#         t_2 = self.t[self.t >= 0]
#         # Gerono lemniscate parameterization (figure-eight shape)
#         # x(t) = a * cos(t)
#         # y(t) = a * sin(t) * cos(t)
#         x = self.a * np.cos(t_2)
#         y = self.a * np.sin(t_2) * np.cos(t_2)
        
#         # Compute first derivatives: these represent the global velocity components
#         dx_dt = np.gradient(x, self.dt)
#         dy_dt = np.gradient(y, self.dt)
        
#         # Compute the forward speed as the magnitude of the velocity vector
#         v = np.sqrt(dx_dt**2 + dy_dt**2)
        
#         # Compute second derivatives (needed for curvature)
#         ddx_dt2 = np.gradient(dx_dt, self.dt)
#         ddy_dt2 = np.gradient(dy_dt, self.dt)
        
#         # Compute curvature kappa at each time step:
#         #   kappa = (dx/dt * d²y/dt² - dy/dt * d²x/dt²) / (v**3)
#         curvature = (dx_dt * ddy_dt2 - dy_dt * ddx_dt2) / (v**3 + 1e-8)
        
#         # The yaw rate command is given by the curvature times the forward speed
#         w_z = curvature * v
        
#         # Return the commands as a list of (v_x, w_z) tuples in the local frame
#         command2 = np.stack((v, w_z), axis=1)
        
#         v1 = v[0] * np.ones(len(t_1))
#         w_z1 = np.zeros(len(t_1))
#         command1 = np.stack((v1, w_z1), axis=1)
#         self.commands = np.concatenate((command1, command2), axis=0)
#         print(len(t_1))

#     def sample(self, step:int, num_samples: int) -> List[Tuple[float, float, float]]:
#         v = self.commands[self.traj_index, 0] * np.ones(num_samples)
#         w_z = self.commands[self.traj_index, 1] * np.ones(num_samples)
#         vy = np.zeros(num_samples)
#         commands = np.stack((v, vy, w_z), axis=1).tolist()
#         self.traj_index += 1
#         return commands


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