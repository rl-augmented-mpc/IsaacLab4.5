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


def get_ground_gradient_at_landing_point(
    num_envs: int,
    foot_placement: torch.Tensor,
    gait_contact: torch.Tensor,
    height_map_2d: torch.Tensor,
    resolution: float = 0.1,
    ):
    """
    This function returns flatness value which is the difference between the max and min height in a 3x3 grid. 
    So, if robot is stepping on non-flat surface, flatness value results in terrain height value.
    """
    
    swing_foot_placement = foot_placement.clone().reshape(-1, 2, 2)
    swing_foot_placement = (swing_foot_placement * (gait_contact==0).unsqueeze(2)).sum(dim=1)
    height, width = height_map_2d.shape[1:3]
    
    # get neighboring 3x3 grid with center being projected foot placement
    row_index = (int(width/2) - (swing_foot_placement[:, 1]//resolution).long()).long().clamp(0, width-1)
    col_index = (int(height/2) + (swing_foot_placement[:, 0]//resolution).long()).long().clamp(0, height-1)
    
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


def bilinear_interpolation(
    foot_position_2d: torch.Tensor, 
    height_map_2d: torch.Tensor,
    resolution: float = 0.1,
):
    """
    foot position_2d: (num_envs, num_sample, 2, 2)
    height_map_2d: (num_envs, height, width)
    
    P1 ----- P2
    |    x   |
    P3 ----- P4
    """
    num_envs = foot_position_2d.shape[0]
    num_foot_edge = foot_position_2d.shape[2]
    num_foot = foot_position_2d.shape[1]
    
    env_ids = torch.arange(num_envs)
    height, width = height_map_2d.shape[1:3]
    
    # in image space
    # foot position_2d_flat: (num_envs, num_foot_edge*num_foot, 2)
    foot_position_2d_flat = foot_position_2d.clone().reshape(num_envs, -1, 2)
    foot_position_discrete = torch.zeros_like(foot_position_2d_flat)
    foot_position_discrete[:, :, 0] = height//2 - (foot_position_2d_flat[:, :, 1]/resolution) # row 
    foot_position_discrete[:, :, 1] = width//2 + (foot_position_2d_flat[:, :, 0]/resolution) # col
    
    # (num_envs, num_foot_edge*num_foot, 2)
    P1 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P2 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P3 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P4 = torch.zeros_like(foot_position_discrete).to(torch.long)
    
    P1[:, :, 0] = foot_position_discrete[:, :, 0].floor().clamp(0, height-1)
    P1[:, :, 1] = foot_position_discrete[:, :, 1].floor().clamp(0, width-1)
    P2[:, :, 0] = foot_position_discrete[:, :, 0].floor().clamp(0, height-1)
    P2[:, :, 1] = foot_position_discrete[:, :, 1].ceil().clamp(0, width-1)
    P3[:, :, 0] = foot_position_discrete[:, :, 0].ceil().clamp(0, height-1)
    P3[:, :, 1] = foot_position_discrete[:, :, 1].floor().clamp(0, width-1)
    P4[:, :, 0] = foot_position_discrete[:, :, 0].ceil().clamp(0, height-1)
    P4[:, :, 1] = foot_position_discrete[:, :, 1].ceil().clamp(0, width-1)
    
    P1_height = height_map_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P1[:, :, 0] * width + P1[:, :, 1]].view(num_envs, -1)
    P2_height = height_map_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P2[:, :, 0] * width + P2[:, :, 1]].view(num_envs, -1)
    P3_height = height_map_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P3[:, :, 0] * width + P3[:, :, 1]].view(num_envs, -1)
    P4_height = height_map_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P4[:, :, 0] * width + P4[:, :, 1]].view(num_envs, -1)
    
    height_at_foot = (P1_height + P2_height + P3_height + P4_height) / 4
    height_at_foot = height_at_foot.reshape(-1, num_foot, num_foot_edge).transpose(1, 2) # (num_envs, num_foot_edge, num_foot)
    
    return height_at_foot
    

def get_ground_roughness_at_landing_point(
    num_envs: int,
    foot_position: torch.Tensor,
    foot_contact: torch.Tensor,
    height_map_2d: torch.Tensor,
    resolution: float = 0.1,
    ):
    """
    This function returns flatness value which is the difference between the max and min height in a 3x3 grid. 
    So, if robot is stepping on non-flat surface, flatness value results in terrain height value.
    
    
    Foot edge configurations 
    y|
    x-> 
    1 ------ 2
    |    x   |
    3 ------ 4
    """
    # foot size to consider stepping over 
    foot_size_x = 0.145
    foot_size_y = 0.073
    
    # in cartesian space 
    foot_position_2d = foot_position.clone().reshape(-1, 2, 3)[:, :, :2] # foot position wrt body frame
    
    foot_edge_positions = torch.zeros(num_envs, 2, 4, 2)
    foot_edge_positions[:, 0, :, :] = foot_position_2d[:, :1, :].repeat(1, 4, 1)
    foot_edge_positions[:, 1, :, :] = foot_position_2d[:, 1:, :].repeat(1, 4, 1)
    foot_edge_positions[:, :, 0, 0] -= foot_size_x/2
    foot_edge_positions[:, :, 0, 1] += foot_size_y/2
    foot_edge_positions[:, :, 1, 0] += foot_size_x/2
    foot_edge_positions[:, :, 1, 1] += foot_size_y/2
    foot_edge_positions[:, :, 2, 0] -= foot_size_x/2
    foot_edge_positions[:, :, 2, 1] -= foot_size_y/2
    foot_edge_positions[:, :, 3, 0] += foot_size_x/2
    foot_edge_positions[:, :, 3, 1] -= foot_size_y/2
    
    height_at_foot = bilinear_interpolation(
        foot_position_2d=foot_edge_positions, 
        height_map_2d=height_map_2d, 
        resolution=resolution
    ) # (num_envs, 4, 2)
    
    roughness_at_foot = torch.abs(height_at_foot.max(dim=1).values - height_at_foot.min(dim=1).values) # (num_envs, 2)
    roughness_at_foot = (roughness_at_foot * foot_contact).sum(dim=1) # (num_envs, 1)
    return roughness_at_foot

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
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    
    width, height = int(grid_x/resolution + 1), int(grid_y/resolution + 1)
    height_map_2d = height_map.view(-1, height, width)
    
    action_term = env.action_manager.get_term(action_name)
    gait_contact = action_term.gait_contact
    foot_placement_b = action_term.foot_placement_b
    
    ground_flatness_at_foot = get_ground_gradient_at_landing_point(
        env.num_envs,
        foot_placement_b,
        gait_contact,
        height_map_2d, 
        resolution,
    )
    
    truncated_height_map = height_map_2d[:, width//2-2:width//2+3, height//2-2:height//2+3].reshape(env.num_envs, -1)
    flatness_in_height_map = torch.max(truncated_height_map, dim=1).values - torch.min(truncated_height_map, dim=1).values
    reward = torch.exp(-torch.square(ground_flatness_at_foot)/std**2) * (flatness_in_height_map > 0.01).float() # do not give reward if height map is flat
    return reward

def stance_foot_position_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    action_name: str = "mpc_action",
    std: float = 0.5,
):
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    height_map = raycaster.data.ray_hits_w[..., 2]  - raycaster.data.pos_w[:, 2].unsqueeze(1)
    
    resolution = raycaster.cfg.pattern_cfg.resolution
    grid_x, grid_y = raycaster.cfg.pattern_cfg.size
    width, height = int(grid_x/resolution + 1), int(grid_y/resolution + 1)
    height_map_2d = height_map.view(-1, height, width)
    
    contacts = (contact_sensor.data.net_forces_w.norm(dim=2) > 1.0).float()
    
    action_term = env.action_manager.get_term(action_name)
    foot_position_b = action_term.foot_pos_b.reshape(-1, 2, 3)
    
    # retrieves ground flatness where stance foot position is projected onto height map.
    ground_flatness_at_foot = get_ground_roughness_at_landing_point(
        env.num_envs,
        foot_position_b,
        contacts,
        height_map_2d, 
        resolution,
    )
    
    window_size = 5 # 5*0.05 = 0.25m
    truncated_height_map = height_map_2d[:, width//2-window_size:width//2+window_size+1, height//2-window_size:height//2+window_size+1].reshape(env.num_envs, -1)
    stepping_stone_mask = torch.max(truncated_height_map, dim=1).values - torch.min(truncated_height_map, dim=1).values
    reward = torch.exp(-torch.square(ground_flatness_at_foot)/std**2) * (stepping_stone_mask > 0.01).float() # do not give reward if height map do not see stepping stone
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