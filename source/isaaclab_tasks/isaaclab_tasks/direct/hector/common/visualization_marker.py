# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

class ContactVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "grf":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robot/hector/props/arrow_z.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
        self.max_contact_force = 400.0/4
    
    def visualize(self, link_positions:torch.Tensor, link_rotations:torch.Tensor, local_contact_force:torch.Tensor):
        """_summary_

        Args:
            link_positions (torch.Tensor): (num_envs, num_instances, 3)
            link_rotations (torch.Tensor): (num_envs, num_instances, 3, 3)
            local_contact_force (torch.Tensor): (num_envs, num_instances, 3)
        """
        batch_size = link_positions.shape[0]
        instance_size = link_positions.shape[1]
        device = link_positions.device
        
        global_contact_force = torch.bmm(link_rotations.reshape(-1, 3, 3), local_contact_force.reshape(-1, 3, 1)).squeeze(-1).reshape(batch_size, instance_size, 3)
        axis_angle = torch.nn.functional.normalize(global_contact_force, p=2, dim=2)
        marker_orientations = torch.zeros(batch_size, instance_size, 4, device=device)
        marker_orientations[:, :, 1] = axis_angle[:, :, 0]
        marker_orientations[:, :, 2] = axis_angle[:, :, 1]
        marker_orientations[:, :, 3] = axis_angle[:, :, 2]
        marker_scales = torch.ones_like(global_contact_force)
        marker_scales[:, :, 2] = torch.norm(global_contact_force, dim=-1)/self.max_contact_force
        self.marker.visualize(link_positions.reshape(-1, 3), marker_orientations.reshape(-1, 4), marker_scales.reshape(-1, 3))

class PenetrationVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "penetration":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robot/hector/props/arrow_z_small.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
        self.marker_scale = 10
    
    def visualize(self, link_positions:torch.Tensor, link_rotations:torch.Tensor):
        """_summary_

        Args:
            link_positions (torch.Tensor): (num_envs, num_instances, 3)
            link_rotations (torch.Tensor): (num_envs, num_instances, 3, 3)
        """
        batch_size = link_positions.shape[0]
        instance_size = link_positions.shape[1]
        device = link_positions.device
        penetration = -1 * (link_positions[:, :, 2] * (link_positions[:, :, 2] < 0.0))
        axis_angle = torch.zeros_like(link_positions)
        axis_angle[:, :, 2] = 1.0
        
        marker_orientations = torch.zeros(batch_size, instance_size, 4, device=device)
        marker_orientations[:, :, 1] = axis_angle[:, :, 0]
        marker_orientations[:, :, 2] = axis_angle[:, :, 1]
        marker_orientations[:, :, 3] = axis_angle[:, :, 2]
        marker_scales = torch.ones_like(link_positions)
        marker_scales[:, :, 2] = penetration * self.marker_scale
        
        link_positions = link_positions.reshape(-1, 3)
        marker_orientations = marker_orientations.reshape(-1, 4)
        marker_scales = marker_scales.reshape(-1, 3)
        
        self.marker.visualize(link_positions, marker_orientations, marker_scales)