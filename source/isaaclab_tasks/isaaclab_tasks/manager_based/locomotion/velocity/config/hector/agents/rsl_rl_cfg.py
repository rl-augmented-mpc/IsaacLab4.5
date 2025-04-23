# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlPPOAgentCfg, RslRlSACAgentCfg, RslRlPolicyRunnerCfg # algorithm branch

## rsl_rl algorithm branch ##

# @configclass
# class PPORunnerCfg(RslRlPolicyRunnerCfg):
#     class_name="PPO"
#     seed=0 
#     device="cuda:0"
#     agent=RslRlPPOAgentCfg(
#         actor_activations=["elu", "elu", "elu", "linear"],
#         actor_hidden_dims=[512, 256, 128],
#         actor_input_normalization=False,
#         actor_noise_std=0.1,
#         batch_count=4, # num mini batch
#         learning_epochs=5,
#         critic_activations=["elu", "elu", "elu", "linear"],
#         critic_hidden_dims=[512, 256, 128],
#         critic_input_normalization=False,
#         clip_ratio=0.2,
#         entropy_coeff=0.01,
#         gae_lambda=0.95,
#         gamma=0.99,
#         gradient_clip=0.5,
#         learning_rate=0.0003,
#         schedule="adaptive",
#         target_kl=0.01,
#         value_coeff=1.0,
#         recurrent=True,
#         )
#     num_steps_per_env=24
#     max_iterations=30000
#     save_interval=50
#     experiment_name="ppo"
#     logger="wandb"
#     wandb_project="rsl_rl_test"
#     run_name = "ppo"
#     # resume=True
#     # load_run = "2025-01-16_20-11-08_ppo"

# @configclass
# class SACRunnerCfg(RslRlPolicyRunnerCfg):
#     class_name="SAC"
#     seed=0 
#     device="cuda:0"
#     agent=RslRlSACAgentCfg(
#         actor_activations=["elu", "elu", "elu", "linear"],
#         actor_hidden_dims=[512, 256, 128],
#         actor_input_normalization=False,
#         actor_noise_std=0.1,
#         batch_count=4, # num mini batch
#         learning_epochs=5,
#         critic_activations=["elu", "elu", "elu", "linear"],
#         critic_hidden_dims=[512, 256, 128],
#         critic_input_normalization=False,
#         action_max=1.0,
#         action_min=-1.0,
#         actor_lr=1e-3,
#         alpha=0.2, 
#         alpha_lr=1e-3, 
#         chimera=True, 
#         critic_lr=1e-3,
#         gradient_clip=1.0, 
#         log_std_max=4.0, 
#         log_std_min=-20.0, 
#         storage_initial_size=0, 
#         storage_size=100000,
#         target_entropy=None,
#         gamma=0.99,
#         # recurrent=True,
#         )
#     num_steps_per_env=24
#     max_iterations=30000
#     save_interval=50
#     experiment_name="sac"
#     logger="wandb"
#     wandb_project="rsl_rl_test"
#     run_name = "sac"


## rsl_rl master branch ##

@configclass
class HectorPPOMLPRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 32 # horizon for rollout
    max_iterations = 20000
    save_interval = 500
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", 
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        clip_actions=True, 
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    logger = "wandb"
    wandb_project = "rl_mpc_manager"
    experiment_name = "ppo_rsl_rl_stepping_stone"
    
@configclass
class HectorPPOGRURunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 24 # horizon for rollout
    max_iterations = 20000
    save_interval = 500
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent", 
        rnn_type="GRU",
        init_noise_std=0.1,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )  # type: ignore
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    logger = "wandb"
    wandb_project = "rl_mpc_manager"
    experiment_name = "ppo_rsl_rl_gru_friction"
    # resume=True
    # load_run = "2025-01-17_22-16-15"

@configclass
class HectorPPOLSTMRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 24 # horizon for rollout
    max_iterations = 20000
    save_interval = 500
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent", 
        rnn_type="LSTM",
        init_noise_std=0.1,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )  # type: ignore
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # logger = "wandb"
    # wandb_project = "rl_mpc"
    experiment_name = "ppo_rsl_rl_lstm_friction"
    # resume=True
    # load_run = "2025-01-17_22-16-15"

@configclass
class HectorPPOGRUSoftRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 24 # horizon for rollout
    max_iterations = 50000
    save_interval = 50
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent", 
        rnn_type="GRU",
        init_noise_std=0.1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )  # type: ignore
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    logger = "wandb"
    wandb_project = "rl_mpc_gm"
    experiment_name = "ppo_rsl_rl_gru_soft"