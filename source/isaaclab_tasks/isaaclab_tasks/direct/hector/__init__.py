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
from .task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg, HierarchicalArchPrimeCfg, HierarchicalArchAccelPFCfg, HierarchicalArchPrimeFullCfg
from .task.hierarchical_arch import HierarchicalArch, HierarchicalArchPrime, HierarchicalArchAccelPF, HierarchicalArchPrimeFull

# soft terrain env
from .task_cfg.soft_terrain_cfg import SoftTerrainEnvCfg
from .task.soft_terrain import SoftTerrainEnv


####################
### Rigid Ground ###
####################

gym.register(
    id="Hector-Hierarchical-Prime-Rigid",
    entry_point=f"{__name__}.task.hierarchical_arch:HierarchicalArchPrime",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg.hierarchical_arch_cfg:HierarchicalArchPrimeCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRURunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPORunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SACRunnerCfg",
    },
)

gym.register(
    id="Hector-Hierarchical-Prime-Full-Rigid",
    entry_point=f"{__name__}.task.hierarchical_arch:HierarchicalArchPrimeFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg.hierarchical_arch_cfg:HierarchicalArchPrimeFullCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRURunnerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOLSTMRunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPORunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SACRunnerCfg",
    },
)

gym.register(
    id="Hector-Hierarchical-AceelPF-Rigid",
    entry_point=f"{__name__}.task.hierarchical_arch:HierarchicalArchAccelPF",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg.hierarchical_arch_cfg:HierarchicalArchAccelPFCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRURunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPORunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SACRunnerCfg",
    },
)

###################
### Soft Ground ###
###################

gym.register(
    id="Hector-Soft-Terrain",
    entry_point=f"{__name__}.task.soft_terrain:SoftTerrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg.soft_terrain_cfg:SoftTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRUSoftRunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPORunnerCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SACRunnerCfg",
    },
)