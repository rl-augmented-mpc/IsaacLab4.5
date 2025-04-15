import math
from inspect import isfunction
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch

def curriculum_linear_growth(
    step: int = 0, start: int = 0, end: int = 1000, **kwargs
) -> float:
    """
    Generates a curriculum with a linear growth rate.

    Args:
        step (int): Current step.
        start (int): Start step.
        end (int): End step.
        **kwargs: Additional arguments.

    Returns:
        float: Rate of growth.
    """

    if step < start:
        return 0.0

    if step > end:
        return 1.0

    current = step - start
    relative_end = end - start

    rate = current / (relative_end)

    return rate


def curriculum_sigmoid_growth(
    step: int = 0, start: int = 100, end: int = 1000, extent: float = 3, **kwargs
) -> float:
    """
    Generates a curriculum with a sigmoid growth rate.

    Args:
        step (int): Current step.
        start (int): Start step.
        end (int): End step.
        extent (float, optional): Extent of the sigmoid function.
        **kwargs: Additional arguments.

    Returns:
        float: Rate of growth.
    """

    if step < start:
        return 0.0

    if step > end:
        return 1.0

    current = step - start
    relative_end = end - start

    rate = (
        math.tanh(((extent * 2 * current / relative_end) - extent) / 2)
        - math.tanh(-extent / 2)
    ) / (math.tanh(extent / 2) - math.tanh(-extent / 2))

    return rate


def curriculum_pow_growth(
    step: int = 0, start: int = 0, end: int = 1000, alpha: float = 2.0, **kwargs
) -> float:
    """
    Generates a curriculum with a power growth rate.

    Args:
        step (int): Current step.
        start (int): Start step.
        end (int): End step.
        alpha (float, optional): Exponent of the power function.
        **kwargs: Additional arguments.

    Returns:
        float: Rate of growth.
    """

    if step < start:
        return 0.0

    if step > end:
        return 1.0

    current = step - start
    relative_end = end - start

    rate = (current / relative_end) ** alpha
    return rate


RateFunctionDict = {
    "none": lambda step, start, end, **kwargs: 1.0,
    "null": lambda step, start, end, **kwargs: 0.0,
    "linear": curriculum_linear_growth,
    "sigmoid": curriculum_sigmoid_growth,
    "pow": curriculum_pow_growth,
}

@dataclass
class CurriculumRateSampler:
    function: str
    start: int
    end: int
    extent: float = 3.0
    alpha: float = 2.0
    
    def __post_init__(self):
        assert self.start >= 0, "Start must be greater than 0"
        assert self.end > 0, "End must be greater than 0"
        assert self.start < self.end, "Start must be smaller than end"
        assert self.function in [
            "null",
            "none",
            "linear",
            "sigmoid",
            "pow",
        ], "Function must be linear, sigmoid or pow, none or null"
        assert self.extent > 0, "Extent must be greater than 0"
        assert self.alpha > 0, "Alpha must be greater than 0"
        self.rate_function = RateFunctionDict[self.function]
        self.kwargs = {
            key: value for key, value in self.__dict__.items() if not isfunction(value)
        }
    
    def sample(self, step:int)->float:
        return self.rate_function(step, **self.kwargs)

@dataclass
class CurriculumLineSampler:
    x_start: float
    x_end: float
    rate_sampler: CurriculumRateSampler
    
    def sample(self, step:int, num_samples: int) -> list[float]:
        rate = self.rate_sampler.sample(step)
        x = self.x_start + rate * (self.x_end - self.x_start)
        points = (np.ones(num_samples) * x).tolist()
        return points

@dataclass
class CurriculumUniformLineSampler:
    x_range_start: List[float] | Tuple[float, float]
    x_range_end: List[float] | Tuple[float, float]
    rate_sampler: CurriculumRateSampler
    
    
    def __post_init__(self):
        assert len(self.x_range_start) == 2
        assert len(self.x_range_end) == 2
    
    def sample(self, step:int, num_samples: int) -> list[float]:
        rate = self.rate_sampler.sample(step)
        x_range = [
            self.x_range_start[0] + rate * (self.x_range_end[0] - self.x_range_start[0]),
            self.x_range_start[1] + rate * (self.x_range_end[1] - self.x_range_start[1])
        ]
        points = np.random.uniform(x_range[0], x_range[1], num_samples).tolist()
        return points

@dataclass
class CurriculumQuaternionSampler:
    x_range_start: List[float] | Tuple[float, float]
    x_range_end: List[float] | Tuple[float, float]
    rate_sampler: CurriculumRateSampler
    
    def __post_init__(self):
        assert len(self.x_range_start) == 2
        assert len(self.x_range_end) == 2
    
    def sample(self, step:int, num_samples: int) -> list[tuple[float, float, float, float]]:
        rate = self.rate_sampler.sample(step)
        x_range = [
            self.x_range_start[0] + rate * (self.x_range_end[0] - self.x_range_start[0]),
            self.x_range_start[1] + rate * (self.x_range_end[1] - self.x_range_start[1])
        ]
        points = np.random.uniform(x_range[0], x_range[1], num_samples)
        quat = np.stack([np.cos(points/2), np.zeros_like(points), np.zeros_like(points), np.sin(points/2)], axis=-1).tolist()
        return quat

@dataclass
class CurriculumUniformPlaneSampler:
    x_range_start: List[float] | Tuple[float, float]
    x_range_end: List[float] | Tuple[float, float]
    y_range_start: List[float] | Tuple[float, float]
    y_range_end: List[float] | Tuple[float, float]
    rate_sampler: CurriculumRateSampler
    
    
    def __post_init__(self):
        assert len(self.x_range_start) == 2
        assert len(self.x_range_end) == 2
        assert len(self.y_range_start) == 2
        assert len(self.y_range_end) == 2
    
    def sample(self, step:int, num_samples: int) -> list[tuple[float, float]]:
        rate = self.rate_sampler.sample(step)
        x_range = [
            self.x_range_start[0] + rate * (self.x_range_end[0] - self.x_range_start[0]),
            self.x_range_start[1] + rate * (self.x_range_end[1] - self.x_range_start[1])
        ]
        y_range = [
            self.y_range_start[0] + rate * (self.y_range_end[0] - self.y_range_start[0]),
            self.y_range_start[1] + rate * (self.y_range_end[1] - self.y_range_start[1])
        ]
        x_vals = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_vals = np.random.uniform(y_range[0], y_range[1], num_samples)
        
        points = np.stack((x_vals, y_vals), axis=-1).tolist()
        return points

@dataclass
class CurriculumUniformCubicSampler:
    x_range_start: List[float] | Tuple[float, float]
    x_range_end: List[float] | Tuple[float, float]
    y_range_start: List[float] | Tuple[float, float]
    y_range_end: List[float] | Tuple[float, float]
    z_range_start: List[float] | Tuple[float, float]
    z_range_end: List[float] | Tuple[float, float]
    rate_sampler: CurriculumRateSampler
    
    def __post_init__(self):
        assert len(self.x_range_start) == 2
        assert len(self.x_range_end) == 2
        assert len(self.y_range_start) == 2
        assert len(self.y_range_end) == 2
        assert len(self.z_range_start) == 2
        assert len(self.z_range_end) == 2
    
    def sample(self, step:int, num_samples: int) -> list[tuple[float, float, float]]:
        rate = self.rate_sampler.sample(step)
        x_range = [
            self.x_range_start[0] + rate * (self.x_range_end[0] - self.x_range_start[0]),
            self.x_range_start[1] + rate * (self.x_range_end[1] - self.x_range_start[1])
        ]
        y_range = [
            self.y_range_start[0] + rate * (self.y_range_end[0] - self.y_range_start[0]),
            self.y_range_start[1] + rate * (self.y_range_end[1] - self.y_range_start[1])
        ]
        z_range = [
            self.z_range_start[0] + rate * (self.z_range_end[0] - self.z_range_start[0]),
            self.z_range_start[1] + rate * (self.z_range_end[1] - self.z_range_start[1])
        ]
        x_vals = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_vals = np.random.uniform(y_range[0], y_range[1], num_samples)
        z_vals = np.random.uniform(z_range[0], z_range[1], num_samples)
        
        points = np.stack((x_vals, y_vals, z_vals), axis=-1).tolist()
        return points


## performance dependent curriculum ##
## Changes difficulty level of task based on the current performance of the agent

@dataclass
class PerformanceCurriculumLineSampler:
    x_start: float
    x_end: float
    num_curriculums: int
    update_frequency: int
    maximum_episode_length: int
    ratio: float = 0.9
    iteration_threshold: int = 1000
    def __post_init__(self):
        self.curriculum_idx = 0
        self.counter = 0
    def sample(self, step:int, mean_episode_length: float, num_samples: int) -> list[float]:
        if step > (self.curriculum_idx+1) * self.iteration_threshold:
            if mean_episode_length >= self.maximum_episode_length * self.ratio:
                if self.counter >= self.update_frequency:
                    self.curriculum_idx += 1
                    self.counter = 0
                else:
                    self.counter += 1
            else:
                self.counter = 0
        self.curriculum_idx = min(self.curriculum_idx, self.num_curriculums - 1)
        value = self.x_start + (self.x_end - self.x_start + 1) * (self.curriculum_idx / self.num_curriculums)
        return (value * np.ones(num_samples)).tolist()