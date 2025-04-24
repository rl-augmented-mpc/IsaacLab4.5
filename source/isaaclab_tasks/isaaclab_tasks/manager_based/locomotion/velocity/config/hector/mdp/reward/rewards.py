# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

""" 
helper functions 
"""

def get_ground_gradient(height_map_2d:torch.Tensor)->torch.Tensor:
    grad_x = torch.gradient(height_map_2d, dim=1)[0] # gradient returns tuple
    grad_y = torch.gradient(height_map_2d, dim=2)[0] # gradient returns tuple
    height_map_2d_grad = torch.abs(grad_x) + torch.abs(grad_y)
    return height_map_2d_grad

def get_ground_gradient_at_landing_point(
    num_envs: int,
    foot_placement: torch.Tensor, 
    gait_contact: torch.Tensor, 
    height_map_2d: torch.Tensor,
    ):
    
    foot_placement = foot_placement.reshape(-1, 2, 2)[gait_contact==0]
    height_map_2d_grad = get_ground_gradient(height_map_2d)
    height, width = height_map_2d_grad.shape[1:3]
    
    # get neighboring 3x3 grid with center being projected foot placement
    resolution = 0.1
    row_index = (-(foot_placement[:, 1]//resolution).long() + int((height-1)/2)).long()
    row_index = torch.clamp(row_index, 0, height-1).unsqueeze(1).unsqueeze(2).repeat(1, 3, 3)
    col_index = ((foot_placement[:, 0]//resolution).long() + int((width-1)/2)).long()
    col_index = torch.clamp(col_index, 0, width-1).unsqueeze(1).unsqueeze(2).repeat(1, 3, 3)
    
    row_index[:, :, 0] = row_index[:, :, 1] - 1
    row_index[:, :, 2] = row_index[:, :, 1] + 1
    col_index[:, 0, :] = col_index[:, 1, :] - 1
    col_index[:, 2, :] = col_index[:, 1, :] + 1
    
    indices = (width*col_index.view(num_envs, -1) + row_index.view(num_envs, -1)).long() # (num_envs, 9)
    indices = torch.clamp(indices, 0, height*width-1).view(-1)
    
    env_ids = torch.arange(num_envs).view(-1, 1).repeat(1,9).view(-1)
    ground_gradient = height_map_2d_grad.reshape(num_envs, -1).repeat(9, 1)[env_ids, indices] # type: ignore
    ground_gradient_at_foot = ground_gradient.view(num_envs, 9).mean(dim=1)
    
    return ground_gradient_at_foot


""" 
rewards
"""

def individual_action_l2(env: ManagerBasedRLEnv, action_idx:int) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.square(env.action_manager.action[:, action_idx])

def foot_placement_reward(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    action_name: str = "mpc_action",
    std: float = 0.5,
):
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_map = raycaster.data.ray_hits_w[..., 2]  - raycaster.data.pos_w[:, 2].unsqueeze(1)
    width, height = int(1.0/0.1 + 1), int(1.0/0.1 + 1)
    height_map_2d = height_map.view(-1, height, width)
    
    action_term = env.action_manager.get_term(action_name)
    gait_contact = action_term.gait_contact
    foot_placement_b = action_term.foot_placement_b
    
    ground_gradient_at_foot = get_ground_gradient_at_landing_point(
        env.num_envs,
        foot_placement_b,
        gait_contact,
        height_map_2d
    )
    
    reward = torch.exp(-torch.square(ground_gradient_at_foot)/std**2)
    return reward


def leg_body_angle_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
):
    action_term = env.action_manager.get_term(action_name)
    leg_body_angle = action_term.leg_angle # (num_envs, 4)
    
    return torch.sum(torch.square(leg_body_angle), dim=1)