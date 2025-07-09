# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hector RL MPC environment.
"""

import gymnasium as gym
from . import agents


BASE_CLASS = [
    "HECTOR-ManagerBased-RL",
    "HECTOR-ManagerBased-RL-PLAY",
    
    "HECTOR-ManagerBased-RL-SAC-Rough-Blind",
    "HECTOR-ManagerBased-RL-SAC-Rough-Blind-PLAY",
    
    "HECTOR-ManagerBased-RL-SAC-Rough-Perceptive",
    "HECTOR-ManagerBased-RL-SAC-Rough-Perceptive-PLAY",
    
    "HECTOR-ManagerBased-RL-SAC-SLIP",
    "HECTOR-ManagerBased-RL-SAC-SLIP-PLAY",
    
    "HECTOR-ManagerBased-RL-L2T-Rough",
    "HECTOR-ManagerBased-RL-L2T-Rough-PLAY",
]

ARGS = [
    # PPO env
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HECTORRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOMLPRunnerCfg", 
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_mlp_encoder.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_encoder.yaml",
    },
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HECTORRoughEnvCfgPLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOMLPRunnerCfg", 
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_mlp_encoder.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_encoder.yaml",
    },
    
    # SAC blind rough env
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_sac_cfg:HECTORRoughEnvBlindLocomotionSACCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_st.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
    },
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_sac_cfg:HECTORRoughEnvBlindLocomotionSACCfgPLAY",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_st.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
    },
    
    # SAC percptive rough env
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_sac_cfg:HECTORRoughEnvPerceptiveLocomotionSACCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_random_block.yaml",
    },
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_sac_cfg:HECTORRoughEnvPerceptiveLocomotionSACCfgPLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_random_block.yaml",
    },
    
    # SAC slip blind env
    {
        "env_cfg_entry_point": f"{__name__}.slip_env_sac_cfg:HECTORSlipEnvSACCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_slip.yaml",
    },
    {
        "env_cfg_entry_point": f"{__name__}.slip_env_sac_cfg:HECTORSlipEnvSACCfgPLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_slip.yaml",
    },
    
    # L2T rough env
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_l2t_cfg:HECTORRoughEnvL2TCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_l2t_cfg.yaml",
    },
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_l2t_cfg:HECTORRoughEnvL2TCfgPLAY",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_l2t_cfg.yaml",
    },
]

for i in range(len(BASE_CLASS)):
    gym.register(
        id=BASE_CLASS[i], 
        entry_point="isaaclab.envs:ManagerBasedRLEnv", 
        disable_env_checker=True, 
        kwargs=ARGS[i])