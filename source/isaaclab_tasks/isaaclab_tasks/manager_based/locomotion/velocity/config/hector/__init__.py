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
    "Hector-ManagerBased-RL-PLAY",
]

ARGS = [
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HECTORRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOMLPRunnerCfg", 
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml",
    },
    {
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HECTORRoughEnvCfgPLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOMLPRunnerCfg", 
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml",
    },
]

for i in range(len(BASE_CLASS)):
    gym.register(id=BASE_CLASS[i], entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs=ARGS[i])