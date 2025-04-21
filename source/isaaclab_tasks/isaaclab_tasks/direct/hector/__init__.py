# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hector RL MPC environment.
"""

import gymnasium as gym
from . import agents

# hierarchical arch env
from .task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg, HierarchicalArchPrimeCfg, HierarchicalArchPrimeFullCfg
from .task.hierarchical_arch import HierarchicalArch, HierarchicalArchPrime, HierarchicalArchPrimeFull

# stepping stone
from .task_cfg.stepping_stone_cfg import SteppingStoneCfg
from .task.stepping_stone import SteppingStone

# soft terrain env
# from .task_cfg.soft_terrain_cfg import SoftTerrainEnvCfg
# from .task.soft_terrain import SoftTerrainEnv


BASE_CLASS = [
    "Hector-Hierarchical-Prime-Rigid",
    "Hector-Hierarchical-Prime-Full-Rigid",
    "SteppingStone",
]

ENTRY_POINT = [
    f"{__name__}.task.hierarchical_arch:HierarchicalArchPrime",
    f"{__name__}.task.hierarchical_arch:HierarchicalArchPrimeFull",
    f"{__name__}.task.stepping_stone:SteppingStone",
]

ARGS = [
    {"env_cfg_entry_point": f"{__name__}.task_cfg.hierarchical_arch_cfg:HierarchicalArchPrimeCfg", 
     "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRURunnerCfg"},
    {"env_cfg_entry_point": f"{__name__}.task_cfg.hierarchical_arch_cfg:HierarchicalArchPrimeFullCfg", 
     "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOMLPRunnerCfg", 
     "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml"},
    {"env_cfg_entry_point": f"{__name__}.task_cfg.stepping_stone_cfg:SteppingStoneCfg", 
     "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOMLPRunnerCfg", 
     "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml",
    #  "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac.yaml",
     },
]

for i in range(len(BASE_CLASS)):
    gym.register(id=BASE_CLASS[i], entry_point=ENTRY_POINT[i], disable_env_checker=True, kwargs=ARGS[i])

###################
### Soft Ground ###
###################

# gym.register(
#     id="Hector-Soft-Terrain",
#     entry_point=f"{__name__}.task.soft_terrain:SoftTerrainEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.task_cfg.soft_terrain_cfg:SoftTerrainEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRUSoftRunnerCfg",
#         # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPORunnerCfg",
#         # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SACRunnerCfg",
#     },
# )