# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
    


##############################################
## config class for rsl rl algorithm branch ##
##############################################

"""
Different from the main branch, rsl rl algorithm branch has configuration for agents (algorithm) and learning parameters (runner).
Currently, we only have PPO and SAC agents, but more agents will be added in the future.
"""

@configclass
class RslRlPPOAgentCfg:
    """
    Configuration for the PPO agent.
    
    device (str): The device for the rl-agent.
    batch_count (int): number of minibatch per update (this minibatch is used to update the network). 
                       this number can be computed as num_env*num_steps_per_env//mini_batch_size
    gamma (float): The discount factor.
    
    actor_activations (list[str]): The activation functions for the actor network.
    actor_hidden_dims (list[int]): The hidden dimensions for the actor network.
    actor_input_normalization (bool): Whether to use empirical normalization for the actor network.
    
    critic_activations (list[str]): The activation functions for the critic network.
    critic_hidden_dims (list[int]): The hidden dimensions for the critic network.
    critic_input_normalization (bool): Whether to use empirical normalization for the critic network.
    
    actor_noise_std (float): The noise standard deviation for the actor network.
    clip_ratio (float): The clipping ratio for the policy.
    entropy_coeff (float): The coefficient for the entropy loss.
    gae_lambda (float): The lambda parameter for Generalized Advantage Estimation (GAE).
    gradient_clip (float): The maximum gradient norm.
    learning_rate (float): The learning rate for the policy.
    schedule (str): The learning rate schedule. Can be "fixed" or "adaptive". Defaults to "fixed".
    target_kl (float): The target KL-divergence for the adaptive learning rate schedule.
    value_coeff (float): The coefficient for the value function loss in the PPO objective.
    """
    
    device: str = "cuda:0"
    recurrent:bool = False
    batch_count: int = MISSING # environment_count * num_steps_per_env // 64
    learning_epochs: int = MISSING
    gamma:float = MISSING
    
    actor_activations: list[str] = MISSING
    actor_hidden_dims: list[int] = MISSING
    actor_input_normalization: bool = MISSING
    
    critic_activations:list[str] = MISSING
    critic_hidden_dims:list[int] = MISSING
    critic_input_normalization:bool = MISSING
    
    actor_noise_std: float = MISSING
    clip_ratio:float = MISSING
    entropy_coeff:float = MISSING
    gae_lambda:float = MISSING
    gradient_clip:float = MISSING
    learning_rate:float = MISSING
    schedule:str = MISSING
    target_kl:float = MISSING
    value_coeff:float = MISSING

@configclass
class RslRlSACAgentCfg:
    """
    Configuration for the SAC agent.
    
    device (str): The device for the rl-agent.
    batch_count (int): number of minibatch per update (this minibatch is used to update the network). 
                       this number can be computed as num_env*num_steps_per_env//mini_batch_size
    gamma (float): The discount factor.
    
    actor_activations (list[str]): The activation functions for the actor network.
    actor_hidden_dims (list[int]): The hidden dimensions for the actor network.
    actor_input_normalization (bool): Whether to use empirical normalization for the actor network.
    
    critic_activations (list[str]): The activation functions for the critic network.
    critic_hidden_dims (list[int]): The hidden dimensions for the critic network.
    critic_input_normalization (bool): Whether to use empirical normalization for the critic network.
    
    action_max (float): The maximum action value.
    action_min (float): The minimum action value.
    actor_lr (float): The learning rate for the actor network.
    actor_noise_std (float): The noise standard deviation for the actor network.
    alpha (float): Initial entropy regularization coefficient.
    alpha_lr (float): Learning rate for entropy regularization coefficient.
    chimera (bool): Whether to use separate heads for computing action mean and std (True) or treat the std as a tunable parameter (True).
    critic_lr (float): The learning rate for the critic network.
    gradient_clip (float): The maximum gradient norm.
    log_std_max (float): The maximum log standard deviation.
    log_std_min (float): The minimum log standard deviation.
    storage_initial_size (int): Initial size of the replay storage.
    storage_size (int): Maximum size of the replay storage.
    target_entropy (float): Target entropy for the actor policy. Defaults to action space dimensionality.
    """
    device: str = "cuda:0"
    batch_count: int = MISSING # environment_count * num_steps_per_env // 64
    learning_epochs: int = MISSING
    gamma:float = MISSING
    
    actor_activations: list[str] = MISSING
    actor_hidden_dims: list[int] = MISSING
    actor_input_normalization: bool = MISSING
    
    critic_activations:list[str] = MISSING
    critic_hidden_dims:list[int] = MISSING
    critic_input_normalization:bool = MISSING
    
    action_max: float = MISSING
    action_min: float = MISSING
    actor_lr: float = MISSING
    actor_noise_std: float = MISSING
    alpha: float = MISSING
    alpha_lr: float = MISSING
    chimera: bool = MISSING
    critic_lr: float = MISSING
    gradient_clip: float = MISSING
    log_std_max: float = MISSING
    log_std_min: float = MISSING
    storage_initial_size: int = MISSING
    storage_size: int = MISSING
    target_entropy: float = MISSING
    
@configclass
class RslRlPolicyRunnerCfg:
    """
    class_name (str): The class name of the runner. [PPO, SAC]
    seed (int): The seed for the experiment.
    device (str): The device for the rl-agent.
    num_steps_per_env (int): The number of steps per environment per update.
    max_iterations (int): The maximum number of iterations.
    agent (RslRlPPOAgentCfg, RslRlSACAgentCfg): The agent configuration.
    save_interval (int): The number of iterations between saves.
    experiment_name (str): The experiment name.
    run_name (str): The run name.
                    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
                    then it is appended to the run directory's name, i.e. the logging directory's name will become
                    ``{time-stamp}_{run_name}``.
    logger (str): The logger to use. [tensorboard, neptune, wandb]
    neptune_project (str): The neptune project name.
    wandb_project (str): The wandb project name.
    resume (bool): Whether to resume.
    load_run (str): The run directory to load.
                    If regex expression, the latest (alphabetical order) matching run will be loaded.
    load_checkpoint (str): The checkpoint file to load. Default is ``"model_.*.pt"`` (all).
                           If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
    
    class_name: str = MISSING # [PPO, SAC]
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    agent: RslRlPPOAgentCfg = MISSING

    ##
    # Checkpointing parameters
    ##
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    
    ##
    # Logging parameters
    ##
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"

    ##
    # Loading parameters
    ##
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
