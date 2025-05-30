# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from pxr import UsdPhysics

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedEnv
from . import actions_cfg

from .robot_helper import RobotCore
from .mpc_controller import MPC_Conf, MPCController
from .visualization_marker import FootPlacementVisualizer, SlackedFootPlacementVisualizer


class MPCAction(ActionTerm):

    cfg: actions_cfg.MPCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    
    def __init__(self, cfg: actions_cfg.MPCActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # create robot helper object
        body_names = self._asset.data.body_names
        foot_idx = [i for i in range(len(body_names)) if body_names[i] in ["L_sole", "R_sole"]]
        self.robot_api = RobotCore(self._asset, self.num_envs, torch.tensor(foot_idx, device=self.device, dtype=torch.long))

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)
        
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_actions = np.zeros((self.num_envs, self._num_joints), dtype=np.float32)
        self._action_lb = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._action_ub = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._action_lb[:] = torch.tensor(self.cfg.action_range[0], device=self.device)
        self._action_ub[:] = torch.tensor(self.cfg.action_range[1], device=self.device)
        
        
        # create mpc object array
        # ============================
        # each stance/swing phase includes 10 mpc time steps
        # this means iteration_between_mpc = phase_steps/(10*dt)
        # mpc is updated at same rate as rl policy
        mpc_conf = MPC_Conf(
            control_dt=env.physics_dt, control_iteration_between_mpc=self.cfg.control_iteration_between_mpc, 
            horizon_length=self.cfg.horizon_length, mpc_decimation=int(env.step_dt//env.physics_dt))
        self.mpc_controller = [MPCController(mpc_conf) for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.mpc_controller[i].set_planner(self.cfg.foot_placement_planner)
            self.mpc_controller[i].set_terrain_friction(self.cfg.friction_cone_coef)
        
        # create tensors to store mpc state
        self.state = torch.zeros(self.num_envs, 33, device=self.device)
        self.root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.root_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.root_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.root_rot_mat = torch.zeros(self.num_envs, 3, 3, device=self.device, dtype=torch.float32)
        self.root_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.root_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.joint_pos = torch.zeros(self.num_envs, self._num_joints, device=self.device, dtype=torch.float32)
        self.joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device, dtype=torch.float32)
        
        self.grw = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float32) # ground reaction wrench wrt body frame
        self.grw_accel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # ground reaction acceleration
        self.gait_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self.gait_contact[:, 0] = 1.0 # left foot contact at the beginning
        self.swing_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self.foot_placement_w = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.foot_placement_b = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32) # in body frame
        self.foot_pos_w = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # foot position in world frame
        self.foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # foot position in body frame
        self.ref_foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # reference foot position in body frame
        self.leg_angle = torch.zeros(self.num_envs, 4, device=self.device)
        self.mpc_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.mpc_cost = torch.zeros(self.num_envs, device=self.device)
        
        # reference
        self.command = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.reference_height = self.cfg.nominal_height * np.ones(self.num_envs, dtype=np.float32)
        self.foot_placement_height = np.zeros(self.num_envs, dtype=np.float32)
        
        # markers
        self.foot_placement_visualizer = FootPlacementVisualizer("/Visuals/foot_placement")
        self.slacked_foot_placement_visualizer = SlackedFootPlacementVisualizer("/Visuals/slacked_foot_placement")
    
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """
        mpc control parameters:
        - gait stepping frequency 
        - swing foot height 
        - swing trajectory control points
        """
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # print(f"raw actions: {self._raw_actions}")
        # self._raw_actions[:, 0] = 2*torch.rand(self.num_envs, device=self.device) - 1 # randomize gait frequency
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        # split processed actions into individual control parameters
        sampling_time = self.cfg.nominal_mpc_dt * (1 + self._processed_actions[:, -3].cpu().numpy())
        swing_foot_height = self._processed_actions[:, 1].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, 2].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update reference
        self._get_mpc_state()
        self._get_reference_velocity()
        self._get_reference_height()
        self._get_footplacement_height()
        
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i])
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.command[i],
                height=self.reference_height[i],
            )
        self.visualize_marker()
        
    def _get_reference_velocity(self):
        ramp_up_duration = 1.2 # seconds
        ramp_up_coef = torch.clip(self.mpc_counter/int(ramp_up_duration/self._env.physics_dt), 0.0, 1.0).unsqueeze(1)
        self.command[:, :] = (ramp_up_coef * self._env.command_manager.get_command(self.cfg.command_name)).cpu().numpy()
    
    def _get_reference_height(self):
        sensor= self._env.scene.sensors["height_scanner"]
        height_map = sensor.data.ray_hits_w[..., 2]
        
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        target_pos = (self.foot_pos_b.reshape(self.num_envs, 2, 3) * (self.gait_contact==1).unsqueeze(2)).sum(dim=1)
        
        # # rough discretization
        # body_center_in_image = (width//2 - scan_offset[0], height//2 - scan_offset[1])
        # col_index = ((target_pos[:, 0]/scan_resolution) + body_center_in_image[0]).ceil().clamp(0, width-1)
        # row_index = ((target_pos[:, 1]/scan_resolution) + body_center_in_image[1]).ceil().clamp(0, height-1)
        # indices = (width*row_index + col_index).long() # flatten index
        # ground_height = height_map[torch.arange(self.num_envs), indices]
        
        # bilinear interpolation
        x_img = target_pos[:, 0] / scan_resolution + (width // 2 - scan_offset[0])
        y_img = target_pos[:, 1] / scan_resolution + (height // 2 - scan_offset[1])

        # Clamp to valid range before ceil/floor to avoid out-of-bounds
        x0 = x_img.floor().clamp(0, width - 2)     # [N]
        x1 = (x0 + 1).clamp(0, width - 1)          # [N]
        y0 = y_img.floor().clamp(0, height - 2)    # [N]
        y1 = (y0 + 1).clamp(0, height - 1)         # [N]

        # Calculate weights for interpolation
        wx = (x_img - x0).unsqueeze(1)             # [N, 1]
        wy = (y_img - y0).unsqueeze(1)             # [N, 1]

        # Convert to long for indexing
        x0 = x0.long()
        x1 = x1.long()
        y0 = y0.long()
        y1 = y1.long()

        # Flattened index computation
        idx00 = y0 * width + x0
        idx10 = y0 * width + x1
        idx01 = y1 * width + x0
        idx11 = y1 * width + x1

        # Gather the four corner heights
        z00 = height_map[torch.arange(self.num_envs), idx00]
        z10 = height_map[torch.arange(self.num_envs), idx10]
        z01 = height_map[torch.arange(self.num_envs), idx01]
        z11 = height_map[torch.arange(self.num_envs), idx11]

        # Bilinear interpolation
        z0 = (1 - wx) * z00.unsqueeze(1) + wx * z10.unsqueeze(1)  # along x
        z1 = (1 - wx) * z01.unsqueeze(1) + wx * z11.unsqueeze(1)  # along x
        ground_height = ((1 - wy) * z0 + wy * z1).squeeze(1)      # along y
        
        ground_level_odometry_frame = self.robot_api._init_pos[:, 2] - self.robot_api.default_root_state[:, 2]
        self.reference_height = self.cfg.nominal_height + (ground_height - ground_level_odometry_frame).cpu().numpy()
        
        # squat motion
        # randomize_duration = int(2.0/self._env.physics_dt) # 2sec
        # time_step = self.mpc_counter % randomize_duration
        # coef = torch.zeros(self.num_envs, device=self.device)
        # coef[time_step<=randomize_duration//2] = time_step/(randomize_duration//2) # [0, 1]
        # coef[time_step>randomize_duration//2] = (randomize_duration-time_step)/(randomize_duration//2) # [1, 0]
        # offset = -0.2*coef
        # self.reference_height = self.cfg.nominal_height + (ground_height - ground_level_odometry_frame).cpu().numpy() + offset.cpu().numpy()
        
    def _get_footplacement_height(self):
        sensor= self._env.scene.sensors["height_scanner"]
        height_map = sensor.data.ray_hits_w[..., 2]
        
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        target_pos = (self.foot_placement_b.reshape(self.num_envs, 2, 2) * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        
        # # rough discretization
        # body_center_in_image = (width//2 - scan_offset[0], height//2 - scan_offset[1])
        # col_index = ((target_pos[:, 0]/scan_resolution) + body_center_in_image[0]).ceil().clamp(0, width-1)
        # row_index = ((target_pos[:, 1]/scan_resolution) + body_center_in_image[1]).ceil().clamp(0, height-1)
        # indices = (width*row_index + col_index).long() # flatten index
        # ground_height = height_map[torch.arange(self.num_envs), indices]
        
        # bilinear interpolation
        x_img = target_pos[:, 0] / scan_resolution + (width // 2 - scan_offset[0])
        y_img = target_pos[:, 1] / scan_resolution + (height // 2 - scan_offset[1])

        # Clamp to valid range before ceil/floor to avoid out-of-bounds
        x0 = x_img.floor().clamp(0, width - 2)     # [N]
        x1 = (x0 + 1).clamp(0, width - 1)          # [N]
        y0 = y_img.floor().clamp(0, height - 2)    # [N]
        y1 = (y0 + 1).clamp(0, height - 1)         # [N]

        # Calculate weights for interpolation
        wx = (x_img - x0).unsqueeze(1)             # [N, 1]
        wy = (y_img - y0).unsqueeze(1)             # [N, 1]

        # Convert to long for indexing
        x0 = x0.long()
        x1 = x1.long()
        y0 = y0.long()
        y1 = y1.long()

        # Flattened index computation
        idx00 = y0 * width + x0
        idx10 = y0 * width + x1
        idx01 = y1 * width + x0
        idx11 = y1 * width + x1

        # Gather the four corner heights
        z00 = height_map[torch.arange(self.num_envs), idx00]
        z10 = height_map[torch.arange(self.num_envs), idx10]
        z01 = height_map[torch.arange(self.num_envs), idx01]
        z11 = height_map[torch.arange(self.num_envs), idx11]

        # Bilinear interpolation
        z0 = (1 - wx) * z00.unsqueeze(1) + wx * z10.unsqueeze(1)  # along x
        z1 = (1 - wx) * z01.unsqueeze(1) + wx * z11.unsqueeze(1)  # along x
        ground_height = ((1 - wy) * z0 + wy * z1).squeeze(1)      # along y
        
        ground_level_odometry_frame = self.robot_api._init_pos[:, 2] - self.robot_api.default_root_state[:, 2]
        self.foot_placement_height = np.clip((ground_height - ground_level_odometry_frame).cpu().numpy(), 0.0, None)
    
    def visualize_marker(self):
        fp = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
        default_position = self.robot_api._init_pos[:, :3].clone()
        default_position[:, 2] -= self.robot_api.default_root_state[:, 2]
        
        sensor= self._env.scene.sensors["height_scanner"]
        height_map = sensor.data.ray_hits_w[..., 2]
        
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        
        foot_height = []
        for foot in range(2):
            target_pos = self.foot_placement_b.reshape(self.num_envs, 2, 2)[:, foot, :]
            
            x_img = target_pos[:, 0] / scan_resolution + (width // 2 - scan_offset[0])
            y_img = target_pos[:, 1] / scan_resolution + (height // 2 - scan_offset[1])

            # Clamp to valid range before ceil/floor to avoid out-of-bounds
            x0 = x_img.floor().clamp(0, width - 2)     # [N]
            x1 = (x0 + 1).clamp(0, width - 1)          # [N]
            y0 = y_img.floor().clamp(0, height - 2)    # [N]
            y1 = (y0 + 1).clamp(0, height - 1)         # [N]

            # Calculate weights for interpolation
            wx = (x_img - x0).unsqueeze(1)             # [N, 1]
            wy = (y_img - y0).unsqueeze(1)             # [N, 1]

            # Convert to long for indexing
            x0 = x0.long()
            x1 = x1.long()
            y0 = y0.long()
            y1 = y1.long()

            # Flattened index computation
            idx00 = y0 * width + x0
            idx10 = y0 * width + x1
            idx01 = y1 * width + x0
            idx11 = y1 * width + x1

            # Gather the four corner heights
            z00 = height_map[torch.arange(self.num_envs), idx00]
            z10 = height_map[torch.arange(self.num_envs), idx10]
            z01 = height_map[torch.arange(self.num_envs), idx01]
            z11 = height_map[torch.arange(self.num_envs), idx11]

            # Bilinear interpolation
            z0 = (1 - wx) * z00.unsqueeze(1) + wx * z10.unsqueeze(1)  # along x
            z1 = (1 - wx) * z01.unsqueeze(1) + wx * z11.unsqueeze(1)  # along x
            ground_height = ((1 - wy) * z0 + wy * z1).squeeze(1)      # along y
            foot_height.append(ground_height)
        
        fp[:, 0, :2] = self.foot_placement_w[:, :2]
        fp[:, 1, :2] = self.foot_placement_w[:, 2:]
        fp[:, 0, 2] = foot_height[0]
        fp[:, 1, 2] = foot_height[1]
        
        # convert from robot odometry frame to simulation global frame
        world_to_odom_rot = self.robot_api._init_rot
        fp[:, 0, :] = torch.bmm(world_to_odom_rot, fp[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position
        fp[:, 1, :] = torch.bmm(world_to_odom_rot, fp[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position
        
        orientation = self.robot_api.root_quat_w.repeat(2, 1)
        self.foot_placement_visualizer.visualize(fp, orientation)
        self.slacked_foot_placement_visualizer.visualize(fp, orientation)

    def apply_actions(self):
        # obtain state
        self._get_state()
        
        # compute mpc
        for i in range(self.num_envs):
            self.mpc_controller[i].update_state(self.state[i].cpu().numpy())
            self.mpc_controller[i].run()
            self._joint_actions[i] = self.mpc_controller[i].get_action()
        joint_actions = torch.from_numpy(self._joint_actions).to(self.device)
        self.robot_api.set_joint_effort_target(joint_actions, self._joint_ids)
        self.mpc_counter += 1
    
    def _get_state(self) -> None:
        """
        Get robot's center of mass state and joint state. 
        NOTE that the center of mass pose is relative to the initial spawn position (i.e. odometry adding nominal height as offset). 
        """
        
        self.root_rot_mat = self.robot_api.root_rot_mat_local
        self.root_quat = self.robot_api.root_quat_local
        self.root_yaw = self.robot_api.root_yaw_local
        self.root_pos = self.robot_api.root_pos_local
        
        self.root_lin_vel_b = self.robot_api.root_lin_vel_b
        self.root_ang_vel_b = self.robot_api.root_ang_vel_b
        
        self.joint_pos = self.robot_api.joint_pos[:, self._joint_ids]
        self.joint_pos = self._add_joint_offset(self.joint_pos) # map hardware joint zeros and simulation joint zeros
        self.joint_vel = self.robot_api.joint_vel[:, self._joint_ids]
        
        self.state = torch.cat(
            (
                self.root_pos,
                self.root_quat,
                self.root_lin_vel_b,
                self.root_ang_vel_b,
                self.joint_pos,
                self.joint_vel,
            ),
            dim=-1,
        )
        
    def _get_mpc_state(self)->None:
        grw_accel = []
        grw = []
        gait_contact = []
        swing_phase = []
        foot_placement_w = []
        foot_pos_b = []
        foot_ref_pos_b = []
        mpc_cost = []
        
        for i in range(len(self.mpc_controller)):
            grw_accel.append(self.mpc_controller[i].accel_gyro(self.root_rot_mat[i].cpu().numpy()))
            grw.append(self.mpc_controller[i].grfm)
            gait_contact.append(self.mpc_controller[i].contact_state)
            swing_phase.append(self.mpc_controller[i].swing_phase)
            foot_placement_w.append(self.mpc_controller[i].foot_placement)
            foot_pos_b.append(self.mpc_controller[i].foot_pos_b)
            foot_ref_pos_b.append(self.mpc_controller[i].ref_foot_pos_b)
            mpc_cost.append(self.mpc_controller[i].mpc_cost)
        
        self.grw_accel = torch.from_numpy(np.array(grw_accel)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self.grw = torch.from_numpy(np.array(grw)).to(self.device).view(self.num_envs, 12).to(torch.float32)
        self.gait_contact = torch.from_numpy(np.array(gait_contact)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self.swing_phase = torch.from_numpy(np.array(swing_phase)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self.foot_placement_w = torch.from_numpy(np.array(foot_placement_w)).to(self.device).view(self.num_envs, 4).to(torch.float32)
        self.foot_pos_b = torch.from_numpy(np.array(foot_pos_b)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self.ref_foot_pos_b = torch.from_numpy(np.array(foot_ref_pos_b)).to(self.device).view(self.num_envs, 6).to(torch.float32)

        # transform foot placement to body frame
        self.foot_placement_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.foot_placement_b[:, 0] = self.foot_placement_w[:, 0]*torch.cos(self.root_yaw) + self.foot_placement_w[:, 1]*torch.sin(self.root_yaw) - self.root_pos[:, 0]
        self.foot_placement_b[:, 1] = -self.foot_placement_w[:, 0]*torch.sin(self.root_yaw) + self.foot_placement_w[:, 1]*torch.cos(self.root_yaw) - self.root_pos[:, 1]
        self.foot_placement_b[:, 2] = self.foot_placement_w[:, 2]*torch.cos(self.root_yaw) + self.foot_placement_w[:, 3]*torch.sin(self.root_yaw) - self.root_pos[:, 0]
        self.foot_placement_b[:, 3] = -self.foot_placement_w[:, 2]*torch.sin(self.root_yaw) + self.foot_placement_w[:, 3]*torch.cos(self.root_yaw) - self.root_pos[:, 1]
        
        # body-leg angle
        stance_leg_r_left = torch.abs(self.foot_pos_b[:, :3]).clamp(min=1e-6)  # avoid division by zero
        stance_leg_r_right = torch.abs(self.foot_pos_b[:, 3:]).clamp(min=1e-6)  # avoid division by zero
        self.leg_angle[:, 0] = torch.abs(torch.atan2(stance_leg_r_left[:, 0], stance_leg_r_left[:, 2])) # left sagittal
        self.leg_angle[:, 1] = torch.abs(torch.atan2(stance_leg_r_left[:, 1], stance_leg_r_left[:, 2])) # left lateral
        self.leg_angle[:, 2] = torch.abs(torch.atan2(stance_leg_r_right[:, 0], stance_leg_r_right[:, 2])) # right sagittal
        self.leg_angle[:, 3] = torch.abs(torch.atan2(stance_leg_r_right[:, 1], stance_leg_r_right[:, 2])) # right lateral
        
        # compute mpc cost
        self.mpc_cost = torch.from_numpy(np.array(mpc_cost)).to(self.device).view(self.num_envs).to(torch.float32)
        
        swing_foot_pos_b = (self.foot_pos_b.reshape(self.num_envs, 2, 3) * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        swing_foot_ref_pos_b = (self.ref_foot_pos_b.reshape(self.num_envs, 2, 3) * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        # swing_foot_pos_gt = (self.robot_api.foot_pos_b * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        swing_foot_pos_gt = (self.robot_api.foot_pos_local * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        
        # swing_foot_pos_b = self.foot_pos_b.reshape(self.num_envs, 2, 3)
        # swing_foot_ref_pos_b = self.ref_foot_pos_b.reshape(self.num_envs, 2, 3)
        # swing_foot_pos_gt = self.robot_api.foot_pos_b
        
        # print("swing foot ref pos", swing_foot_ref_pos_b)
        # print("swing foot pos", swing_foot_pos_b)
        # print("swing foot pos gt", swing_foot_pos_gt)
        # print("swing leg ", 1-self.gait_contact)
    
    def _add_joint_offset(self, joint_pos:torch.Tensor) -> torch.Tensor:
        joint_pos[:, 2] += torch.pi/4
        joint_pos[:, 3] -= torch.pi/2
        joint_pos[:, 4] += torch.pi/4
        
        joint_pos[:, 7] += torch.pi/4
        joint_pos[:, 8] -= torch.pi/2
        joint_pos[:, 9] += torch.pi/4
        
        return joint_pos

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # reset default position of robot
        # self.robot_api.reset_default_pose(self.robot_api.root_state_w[env_ids, :7], env_ids) # type: ignore
        default_pose = torch.cat([self.robot_api.root_pos_w, self.robot_api.root_quat_w], dim=-1)
        self.robot_api.reset_default_pose(default_pose[env_ids, :], env_ids) # type: ignore
        # reset action
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        # reset mpc controller
        for i in env_ids.cpu().numpy(): # type: ignore
            self.mpc_controller[i].reset()
            self.mpc_controller[i].update_gait_parameter(
                np.array([self.cfg.double_support_duration, self.cfg.double_support_duration]), 
                np.array([self.cfg.single_support_duration, self.cfg.single_support_duration]),)
        self.mpc_counter[env_ids] = 0

class MPCAction2(MPCAction):
    """
    This is a subclass of MPCAction that uses the new action space.
    """
    
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """
        mpc control parameters:
        - centroidal acceleration (R^6)
        - gait stepping frequency 
        - swing foot height 
        - swing trajectory control points
        - foot placement in body frame (R^2)
        """
        return 9
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        # split processed actions into individual control parameters
        A_residual = np.zeros((self.num_envs, 13, 13), dtype=np.float32)
        B_residual = np.zeros((self.num_envs, 13, 12), dtype=np.float32)
        
        # split processed actions into individual control parameters
        centroidal_lin_acc = self._processed_actions[:, :3]
        centroidal_ang_acc = self._processed_actions[:, 3:6]
        centroidal_lin_acc = torch.bmm(self.root_rot_mat, centroidal_lin_acc.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_acc = torch.bmm(self.root_rot_mat, centroidal_ang_acc.unsqueeze(-1)).squeeze(-1)
        A_residual[:, 6:9, -1] = centroidal_lin_acc.cpu().numpy()
        A_residual[:, 9:12, -1] = centroidal_ang_acc.cpu().numpy()
        
        sampling_time = self.cfg.nominal_mpc_dt * (1 + self._processed_actions[:, -3].cpu().numpy())
        swing_foot_height = self._processed_actions[:, -2].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, -1].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update reference
        self._get_mpc_state()
        self._get_reference_velocity()
        self._get_reference_height()
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_srbd_residual(A_residual=A_residual[i], B_residual=B_residual[i])
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=self.cfg.nominal_stepping_frequency, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i])
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.command[i],
                height=self.reference_height[i],
            )
        self.visualize_marker()
        
class MPCAction3(MPCAction):
    """
    This is a subclass of MPCAction that uses the new action space.
    """
    
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """
        mpc control parameters:
        - gait stepping frequency 
        - swing foot height 
        - swing trajectory control points
        """
        return 3
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # self._raw_actions[:, 0] = 2*torch.rand(self.num_envs, device=self.device) - 1 # randomize sampling time
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        stepping_frequency = self.cfg.nominal_stepping_frequency + self._processed_actions[:, 0].cpu().numpy()
        swing_foot_height = self._processed_actions[:, 1].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, 2].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update reference
        self._get_mpc_state()
        self._get_reference_velocity()
        self._get_reference_height()
        for i in range(self.num_envs):
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=stepping_frequency[i], 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i])
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.command[i],
                height=self.reference_height[i],
            )
        self.visualize_marker()