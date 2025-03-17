# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """
# Hector RL MPC environment.
# """

# import gymnasium as gym
# from . import agents

# # hierarchical arch env
# from .task_cfg.base_arch_cfg import BaseArchCfg
# from .task.base_arch import BaseArch

# gym.register(
#     id="Hector-Cuda",
#     entry_point=f"{__name__}.task.base_arch:BaseArch",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.task_cfg.base_arch_cfg:BaseArchCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HectorPPOGRURunnerCfg",
#     },
# )