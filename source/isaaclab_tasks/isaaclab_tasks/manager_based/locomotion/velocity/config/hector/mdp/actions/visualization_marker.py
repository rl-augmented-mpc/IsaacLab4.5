# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR # type: ignore

class RFTForceVisualizer:
    def __init__(self, prim_path:str, num_contact_points:int=1):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "grf":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/arrow_z.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
        self.max_contact_force = 300.0/num_contact_points
    
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
        marker_orientations = torch.zeros(batch_size, instance_size, 4, device=device)
        axis_angle = torch.nn.functional.normalize(global_contact_force, p=2, dim=2)
        marker_orientations[:, :, 1] = axis_angle[:, :, 0]
        marker_orientations[:, :, 2] = axis_angle[:, :, 1]
        marker_orientations[:, :, 3] = axis_angle[:, :, 2]
        marker_scales = torch.ones_like(global_contact_force)
        marker_scales[:, :, 2] = torch.norm(global_contact_force, dim=-1)/self.max_contact_force
        self.marker.visualize(link_positions.reshape(-1, 3), marker_orientations.reshape(-1, 4), marker_scales.reshape(-1, 3))

class HSRForceVisualizer:
    def __init__(self, prim_path:str, num_contact_points:int):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "grf":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/arrow_x_grf.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
        self.max_contact_force = 30.0/num_contact_points
    
    def visualize(self, link_positions:torch.Tensor, link_quat:torch.Tensor, local_contact_force:torch.Tensor):
        """_summary_

        Args:
            link_positions (torch.Tensor): (num_envs, num_instances, 3)
            link_rotations (torch.Tensor): (num_envs, num_instances, 3, 3)
            local_contact_force (torch.Tensor): (num_envs, num_instances, 3)
        """
        marker_orientations = link_quat
        marker_scales = torch.ones_like(local_contact_force)*3
        marker_scales[:, :, 0] = -torch.norm(local_contact_force, dim=-1)/self.max_contact_force
        link_positions[:, :, 2] += 0.05
        self.marker.visualize(link_positions.reshape(-1, 3), marker_orientations.reshape(-1, 4), marker_scales.reshape(-1, 3))

class PenetrationVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "penetration":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/arrow_z_small.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
        self.marker_scale = 5
    
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

class FootPlacementVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.foot_size_x = 0.145
        self.foot_size_y = 0.073
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "left_fps":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/left_foot_print.usd",
                scale=(1.0, 1.0, 1.0),
                ), 
                "right_fps":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/right_foot_print.usd",
                scale=(1.0, 1.0, 1.0),
                ), 
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
    
    def visualize(self, fps:torch.Tensor, orientation:torch.Tensor):
        """_summary_

        Args:
            reibert_fps (torch.Tensor): (num_envs, 2, 3)
            augmented_fps (torch.Tensor): (num_envs, 2, 3)
        """
        positions = fps.clone()
        num_envs = positions.shape[0]
        indices = torch.arange(self.marker.num_prototypes, device=positions.device).reshape(1, -1).repeat(num_envs, 1) # (num_envs, 4)
        positions = positions.reshape(-1, 3) # (num_envs*2, 3)
        indices = indices.reshape(-1)
        self.marker.visualize(translations=positions, orientations=orientation, marker_indices=indices)

class SlackedFootPlacementVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.foot_size_x = 0.145
        self.foot_size_y = 0.073
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "left_fps":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/left_foot_print_margin.usd",
                scale=(1.0, 1.0, 1.0),
                ), 
                "right_fps":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/right_foot_print_margin.usd",
                scale=(1.0, 1.0, 1.0),
                ), 
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
    
    def visualize(self, fps:torch.Tensor, orientation:torch.Tensor):
        """_summary_

        Args:
            reibert_fps (torch.Tensor): (num_envs, 2, 3)
            augmented_fps (torch.Tensor): (num_envs, 2, 3)
        """
        positions = fps.clone()
        num_envs = positions.shape[0]
        indices = torch.arange(self.marker.num_prototypes, device=positions.device).reshape(1, -1).repeat(num_envs, 1) # (num_envs, 4)
        positions = positions.reshape(-1, 3) # (num_envs*2, 3)
        indices = indices.reshape(-1)
        self.marker.visualize(translations=positions, orientations=orientation, marker_indices=indices)


class SwingFootVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "left": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "right": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
    
    def visualize(self, swing_foot_ref:torch.Tensor):
        """_summary_

        Args:
            swing_foot_ref (torch.Tensor): (num_envs, 2, 3)
        """
        num_envs = swing_foot_ref.shape[0]
        indices = torch.arange(self.marker.num_prototypes, device=swing_foot_ref.device).reshape(1, -1).repeat(num_envs, 1) # (num_envs, 2)
        positions = swing_foot_ref.reshape(-1, 3) # (num_envs*2, 3)
        indices = indices.reshape(-1)
        self.marker.visualize(translations=positions, marker_indices=indices)


class VelocityVisualizer:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.markers_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "penetration":
                sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/arrow_x.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            }
        )
        self.marker = VisualizationMarkers(self.markers_cfg)
        self.marker_scale = 1.0
    
    def visualize(self, link_positions:torch.Tensor, link_quat:torch.Tensor, velocity_b:torch.Tensor):
        """_summary_

        Args:
            link_positions (torch.Tensor): (num_envs, 3)
            link_rotations (torch.Tensor): (num_envs, 3, 3)
            local_velocity (torch.Tensor): (num_envs, 3)
        """
        marker_scales = torch.ones_like(velocity_b)
        marker_scales[:, 0] = torch.sign(velocity_b[:, 0]) * torch.norm(velocity_b, dim=-1)/self.marker_scale
        
        marker_positions = link_positions.clone()
        marker_positions[:, 2] += 0.2
        self.marker.visualize(marker_positions.reshape(-1, 3), link_quat.reshape(-1, 4), marker_scales.reshape(-1, 3))