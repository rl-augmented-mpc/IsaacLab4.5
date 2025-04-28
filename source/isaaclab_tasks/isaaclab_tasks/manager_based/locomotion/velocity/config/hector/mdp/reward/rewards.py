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

# def get_ground_gradient_at_landing_point(
#     num_envs: int,
#     foot_placement: torch.Tensor,
#     gait_contact: torch.Tensor,
#     height_map_2d: torch.Tensor,
#     ):
#     foot_placement = foot_placement.reshape(-1, 2, 2)[gait_contact==0]
#     # height_map_2d_grad = get_ground_gradient(height_map_2d)
#     # height, width = height_map_2d_grad.shape[1:3]
#     height, width = height_map_2d.shape[1:3]
    
#     # get neighboring 3x3 grid with center being projected foot placement
#     resolution = 0.1
#     row_index = (-(foot_placement[:, 1]//resolution).long() + int((height-1)/2)).long().clamp(0, height-1)
#     col_index = ((foot_placement[:, 0]//resolution).long() + int((width-1)/2)).long().clamp(0, width-1)
#     row_indexes = torch.zeros(num_envs, 3).long()
#     row_indexes[:, 0] = row_index - 1
#     row_indexes[:, 1] = row_index
#     row_indexes[:, 2] = row_index + 1
#     col_indexes = torch.zeros(num_envs, 3).long()
#     col_indexes[:, 0] = col_index - 1
#     col_indexes[:, 1] = col_index
#     col_indexes[:, 2] = col_index + 1
    
#     # create indices for 3x3 grid
#     indices = torch.zeros(num_envs, 9).long()
#     indices[:, 0] = row_indexes[:, 0] * width + col_indexes[:, 0]
#     indices[:, 1] = row_indexes[:, 0] * width + col_indexes[:, 1]
#     indices[:, 2] = row_indexes[:, 0] * width + col_indexes[:, 2]
#     indices[:, 3] = row_indexes[:, 1] * width + col_indexes[:, 0]
#     indices[:, 4] = row_indexes[:, 1] * width + col_indexes[:, 1]
#     indices[:, 5] = row_indexes[:, 1] * width + col_indexes[:, 2]
#     indices[:, 6] = row_indexes[:, 2] * width + col_indexes[:, 0]
#     indices[:, 7] = row_indexes[:, 2] * width + col_indexes[:, 1]
#     indices[:, 8] = row_indexes[:, 2] * width + col_indexes[:, 2]
#     indices = torch.clamp(indices, 0, height*width-1).view(-1)
    
#     env_ids = torch.arange(num_envs).view(-1, 1).repeat(1,9).view(-1)
#     # ground_gradient = height_map_2d_grad.reshape(num_envs, -1).repeat(9, 1)[env_ids, indices] # type: ignore
#     ground_gradient = height_map_2d.reshape(num_envs, -1).repeat(9, 1)[env_ids, indices] # type: ignore
#     ground_gradient_at_foot = ground_gradient.view(num_envs, 9).mean(dim=1)
#     return ground_gradient_at_foot


def get_ground_gradient_at_landing_point(
    num_envs: int,
    foot_placement: torch.Tensor,
    gait_contact: torch.Tensor,
    height_map_2d: torch.Tensor,
    ):
    """
    This function returns flatness value which is the difference between the max and min height in a 3x3 grid. 
    So, if robot is stepping on non-flat surface, flatness value results in terrain height value.
    """
    
    foot_placement = foot_placement.reshape(-1, 2, 2)[gait_contact==0]
    height, width = height_map_2d.shape[1:3]
    
    # get neighboring 3x3 grid with center being projected foot placement
    resolution = 0.1
    row_index = (int(width/2) - (foot_placement[:, 1]//resolution).long()).long().clamp(0, width-1)
    col_index = (int(height/2) + (foot_placement[:, 0]//resolution).long()).long().clamp(0, height-1)
    row_indexes = torch.zeros(num_envs, 3).long()
    col_indexes = torch.zeros(num_envs, 3).long()
    row_indexes[:, 0] = row_index - 1
    row_indexes[:, 1] = row_index
    row_indexes[:, 2] = row_index + 1
    col_indexes[:, 0] = col_index - 1
    col_indexes[:, 1] = col_index
    col_indexes[:, 2] = col_index + 1
    
    # create indices for 3x3 grid
    indices = torch.zeros(num_envs, 9).long()
    indices[:, 0] = row_indexes[:, 0] * width + col_indexes[:, 0]
    indices[:, 1] = row_indexes[:, 0] * width + col_indexes[:, 1]
    indices[:, 2] = row_indexes[:, 0] * width + col_indexes[:, 2]
    indices[:, 3] = row_indexes[:, 1] * width + col_indexes[:, 0]
    indices[:, 4] = row_indexes[:, 1] * width + col_indexes[:, 1]
    indices[:, 5] = row_indexes[:, 1] * width + col_indexes[:, 2]
    indices[:, 6] = row_indexes[:, 2] * width + col_indexes[:, 0]
    indices[:, 7] = row_indexes[:, 2] * width + col_indexes[:, 1]
    indices[:, 8] = row_indexes[:, 2] * width + col_indexes[:, 2]
    indices = torch.clamp(indices, 0, height*width-1).view(-1)
    
    env_ids = torch.arange(num_envs).view(-1, 1).repeat(1,9).view(-1)
    patch_height_map = height_map_2d.reshape(num_envs, -1).repeat(9, 1)[env_ids, indices].view(num_envs, 9) # type: ignore
    flatness = torch.max(patch_height_map, dim=1).values - torch.min(patch_height_map, dim=1).values
    return flatness


""" 
rewards
"""

def individual_action_l2(env: ManagerBasedRLEnv, action_idx:int, action_name: str = "mpc_action",) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    action_term = env.action_manager.get_term(action_name)
    processed_actions = action_term.processed_actions
    return torch.square(processed_actions[:, action_idx]).view(-1)

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
    
    ground_flatness_at_foot = get_ground_gradient_at_landing_point(
        env.num_envs,
        foot_placement_b,
        gait_contact,
        height_map_2d
    )
    
    truncated_height_map = height_map_2d[:, width//2-3:width//2+4, height//2-3:height//2+4].reshape(env.num_envs, -1)
    flatness_in_height_map = torch.max(truncated_height_map, dim=1).values - torch.min(truncated_height_map, dim=1).values
    reward = torch.exp(-torch.square(ground_flatness_at_foot)/std**2) * (flatness_in_height_map > 0.01).float() # do not give reward if height map is flat
    return reward


def leg_body_angle_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
):
    action_term = env.action_manager.get_term(action_name)
    leg_body_angle = action_term.leg_angle # (num_envs, 4)
    
    return torch.sum(torch.square(leg_body_angle), dim=1)

def mpc_cost_l1(
    env: ManagerBasedRLEnv,
    action_name: str = "mpc_action",
):
    action_term = env.action_manager.get_term(action_name)
    mpc_cost = action_term.mpc_cost.clamp(max=1e3) # (num_envs, 1)
    return torch.abs(mpc_cost)