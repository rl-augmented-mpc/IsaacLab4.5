# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hector RL MPC environment.
"""

import gymnasium as gym
from . import agents
    
# from .tasks_cfg.parallel_arch_cfg import ParallelArchCfg, ParallelArchRatioCfg
# from .tasks_cfg.hierarchical_arch_cfg import HierarchicalArchCfg, HierarchicalArchPrimeCfg, HierarchicalArchAccelPFCfg
# from .tasks_cfg.gravel_cfg import GravelEnvCfg
# from .tasks_cfg.soft_terrain_cfg import SoftTerrainEnvCfg
# from .tasks_cfg.discrete_soft_terrain_cfg import DiscreteSoftTerrainArchCfg

# from .tasks.parallel_arch import ParallelArch, ParallelArchRatio
# from .tasks.hierarchical_arch import HierarchicalArch, HierarchicalArchPrime, HierarchicalArchAccelPF
# from .tasks.gravel import GravelEnv
# from .tasks.soft_terrain import SoftTerrainEnv
# from .tasks.discrete_soft_terrain import DiscreteSoftTerrainArch

from .task_cfg.soft_terrain_cfg import SoftTerrainEnvCfg
from .task.soft_terrain import SoftTerrainEnv

# gym.register(
#     id="Hector-Parallel-Rigid",
#     entry_point="omni.isaac.lab_tasks.direct.hector:ParallelArch",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": ParallelArchCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

# gym.register(
#     id="Hector-Parallel-Ratio-Rigid",
#     entry_point="omni.isaac.lab_tasks.direct.hector:ParallelArchRatio",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": ParallelArchRatioCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

# gym.register(
#     id="Hector-Hierarchical-Rigid",
#     entry_point="omni.isaac.lab_tasks.direct.hector:HierarchicalArch",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": HierarchicalArchCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

# gym.register(
#     id="Hector-Hierarchical-Prime-Rigid",
#     entry_point="omni.isaac.lab_tasks.direct.hector:HierarchicalArchPrime",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": HierarchicalArchPrimeCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPORunnerCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPOGRURunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

# gym.register(
#     id="Hector-Hierarchical-AceelPF-Rigid",
#     entry_point="omni.isaac.lab_tasks.direct.hector:HierarchicalArchAccelPF",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": HierarchicalArchAccelPFCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPORunnerCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPOGRURunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

# gym.register(
#     id="Hector-Hierarchical-Prime-Gravel",
#     entry_point="omni.isaac.lab_tasks.direct.hector:GravelEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GravelEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorGMPPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

# gym.register(
#     id="Hector-Hierarchical-Prime-Soft",
#     entry_point="omni.isaac.lab_tasks.direct.hector:SoftTerrainEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": SoftTerrainEnvCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorGMPPORunnerCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HectorPPOGRUSoftRunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SACRunnerCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac.yaml",
#     },
# )

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