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


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # type: ignore # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    elif env.scene.cfg.terrain.terrain_type == "custom_curriculum":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        grid_width *= terrain_gen_cfg.num_sub_patches
        grid_length *= terrain_gen_cfg.num_sub_patches
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
    
    
def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    min_foot_height = (
        (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]).min(dim=1).values
    )
    
    return asset.data.root_pos_w[:, 2] - min_foot_height < minimum_height

def root_height_above_maximum_adaptive(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    min_foot_height = (
        (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]).min(dim=1).values
    )
    
    return asset.data.root_pos_w[:, 2] - min_foot_height > maximum_height


def bad_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    offset: float = 0.55,
) -> torch.Tensor:
    """Terminate when the foot contacts with the ground."""
    raycaster: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    height_map = raycaster.data.pos_w[:, 2].unsqueeze(1) - raycaster.data.ray_hits_w[..., 2] - offset
    contact = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=2) > 1.0 # contact
    # contact = (contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=3) > 1.0).sum(dim=1).float() == 1.0 # first contact
    
    elevation = height_map.max(dim=1).values - height_map.min(dim=1).values  # (num_envs, num_rays)
    roughness = height_map - height_map[:, 0].unsqueeze(1) # (num_envs, num_rays)
    roughness = torch.abs(roughness).mean(dim=1)
    
    ratio = 0.3
    threshold_up = elevation * ratio
    threshold_down = elevation * (1 - ratio)
    bad_contact = (elevation > 1e-3) * contact.squeeze(-1)
    
    going_up = (height_map[:, -1] - height_map[:, 0] <  -1e-3) * contact.squeeze(-1)
    going_down = (height_map[:, -1] - height_map[:, 0] > 1e-3) * contact.squeeze(-1)
    bad_contact[going_up] = roughness[going_up] < threshold_up[going_up]
    bad_contact[going_down] = roughness[going_down] > threshold_down[going_down]
    
    return bad_contact