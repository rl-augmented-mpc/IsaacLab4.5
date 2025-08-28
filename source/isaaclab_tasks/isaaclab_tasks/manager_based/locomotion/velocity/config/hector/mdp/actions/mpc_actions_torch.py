# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs import ManagerBasedEnv

from biped_pympc import MPCController, MPCConf, ControllerConf
from . import mpc_actions_cfg
from .robot_helper import RobotCore
from isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp.marker import (
    FootPlacementVisualizer, 
    SwingFootVisualizer, 
    PositionTrajectoryVisualizer
    )


class BlindLocomotionTorchMPCAction(ActionTerm):

    cfg: mpc_actions_cfg.BlindLocomotionTorchMPCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    
    def __init__(self, cfg: mpc_actions_cfg.BlindLocomotionTorchMPCActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # create robot helper object
        body_names = self._asset.data.body_names
        foot_idx = [i for i in range(len(body_names)) if body_names[i] in ["L_sole", "R_sole"]]
        self.robot_api = RobotCore(self._asset, torch.tensor(foot_idx, device=self.device, dtype=torch.long), self.num_envs, self.device)

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
        
        
        # mpc controller
        robot_conf = ControllerConf(
            ssp_durations = self.cfg.single_support_duration,
            dsp_durations = self.cfg.double_support_duration,
            swing_height = self.cfg.nominal_swing_height,
        )
        mpc_conf = MPCConf(
            dt=env.physics_dt,
            dt_mpc=cfg.nominal_mpc_dt,
            decimation=int(env.step_dt//env.physics_dt),
            solver=cfg.solver_name,
            print_solve_time=cfg.print_solve_time,
            Q = torch.tensor(
                cfg.Q, device=self.device, dtype=torch.float32), 
            R = torch.tensor(
                cfg.R, device=self.device, dtype=torch.float32
            ),
            
        )
        self.mpc_controller = MPCController(robot_conf, mpc_conf, self.num_envs, self.device, cfg.gait_id)
        
        # create tensors to store mpc state
        # floating base state
        self.state = torch.zeros(self.num_envs, 33, device=self.device)
        self.root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.root_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.root_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.root_rot_mat = torch.zeros(self.num_envs, 3, 3, device=self.device, dtype=torch.float32)
        self.root_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.root_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        # joint state
        self.joint_pos = torch.zeros(self.num_envs, self._num_joints, device=self.device, dtype=torch.float32)
        self.joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device, dtype=torch.float32)
        
        self.grw = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float32) # ground reaction wrench wrt body frame
        self.grw_accel = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # ground reaction acceleration
        self.gait_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self.gait_contact[:, 0] = 1.0 # left foot contact at the beginning
        self.swing_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self.foot_placement_w = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self.foot_placement = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # in body frame
        self.foot_pos_w = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # foot position in world frame
        self.foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # foot position in body frame
        self.ref_foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # reference foot position in body frame
        self.leg_angle = torch.zeros(self.num_envs, 4, device=self.device)


        self.mpc_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.mpc_cost = torch.zeros(self.num_envs, device=self.device)
        self.foot_position_trajectory = torch.zeros(self.num_envs, 10, 3, device=self.device, dtype=torch.float32) # trajectory of the foot placement in world frame
        
        # reference
        self.command = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.reference_height = self.cfg.nominal_height * torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.foot_placement_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # markers
        self.foot_placement_visualizer = FootPlacementVisualizer("/Visuals/foot_placement")
        self.foot_position_visualizer = SwingFootVisualizer("/Visuals/foot_position")
        self.foot_trajectory_visualizer = PositionTrajectoryVisualizer("/Visuals/foot_trajectory", color=(0.0, 0.0, 1.0))
    
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
        # self._raw_actions[:, 0] = 2*torch.rand(self.num_envs, device=self.device) - 1 # randomize gait frequency
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        # split processed actions into individual control parameters
        sampling_time = self.cfg.nominal_mpc_dt * (1 + self._processed_actions[:, -3])
        swing_foot_height = self._processed_actions[:, 1]
        trajectory_control_points = self._processed_actions[:, 2]
        
        # form actual control parameters (nominal value + residual)
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update reference
        self._get_mpc_state()
        self._get_reference_velocity()
        # self._get_reference_height()
        # self._get_footplacement_height()
        
        self.mpc_controller.update_mpc_sampling_time(sampling_time)
        self.mpc_controller.set_swing_parameters(swing_foot_height, cp1, cp2)
        self.mpc_controller.set_command(
            self.command, 
            self.reference_height,
        )
        
        self._get_state()
        self.mpc_controller.update_state(self.state)
        self.mpc_controller.run_mpc()
        
        self.visualize_marker()
        
    def _get_reference_velocity(self):
        # # ramp up
        ramp_up_duration = 0.4 # seconds
        ramp_up_coef = torch.clip(self.mpc_counter/int(ramp_up_duration/self._env.physics_dt), 0.0, 1.0).unsqueeze(1)

        # no ramp up
        # ramp_up_coef = 1.0
        self.command[:, :] = (ramp_up_coef * self._env.command_manager.get_command(self.cfg.command_name))
    
    def _get_reference_height(self):
        # sensor= self._env.scene.sensors["height_scanner"]
        # height_map = sensor.data.ray_hits_w[..., 2]
        
        # scan_width, scan_height = sensor.cfg.pattern_cfg.size
        # scan_resolution = sensor.cfg.pattern_cfg.resolution
        # width = int(scan_width/scan_resolution + 1)
        # height = int(scan_height/scan_resolution + 1)
        # scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        # target_pos = (self.foot_pos_b.reshape(self.num_envs, 2, 3) * (self.gait_contact==1).unsqueeze(2)).sum(dim=1)
        
        # # # rough discretization
        # # body_center_in_image = (width//2 - scan_offset[0], height//2 - scan_offset[1])
        # # col_index = ((target_pos[:, 0]/scan_resolution).long() + body_center_in_image[0]).clamp(0, width-1)
        # # row_index = ((target_pos[:, 1]/scan_resolution).long() + body_center_in_image[1]).clamp(0, height-1)
        # # indices = (width*row_index + col_index).long() # flatten index
        # # ground_height = height_map[torch.arange(self.num_envs), indices]
        
        # # bilinear interpolation
        # x_img = target_pos[:, 0] / scan_resolution + (width // 2 - scan_offset[0])
        # y_img = target_pos[:, 1] / scan_resolution + (height // 2 - scan_offset[1])

        # # Clamp to valid range before ceil/floor to avoid out-of-bounds
        # x0 = x_img.floor().clamp(0, width - 2)     # [N]
        # x1 = (x0 + 1).clamp(0, width - 1)          # [N]
        # y0 = y_img.floor().clamp(0, height - 2)    # [N]
        # y1 = (y0 + 1).clamp(0, height - 1)         # [N]

        # # Calculate weights for interpolation
        # wx = (x_img - x0).unsqueeze(1)             # [N, 1]
        # wy = (y_img - y0).unsqueeze(1)             # [N, 1]

        # # Convert to long for indexing
        # x0 = x0.long()
        # x1 = x1.long()
        # y0 = y0.long()
        # y1 = y1.long()

        # # Flattened index computation
        # idx00 = y0 * width + x0
        # idx10 = y0 * width + x1
        # idx01 = y1 * width + x0
        # idx11 = y1 * width + x1

        # # Gather the four corner heights
        # z00 = height_map[torch.arange(self.num_envs), idx00]
        # z10 = height_map[torch.arange(self.num_envs), idx10]
        # z01 = height_map[torch.arange(self.num_envs), idx01]
        # z11 = height_map[torch.arange(self.num_envs), idx11]

        # # Bilinear interpolation
        # z0 = (1 - wx) * z00.unsqueeze(1) + wx * z10.unsqueeze(1)  # along x
        # z1 = (1 - wx) * z01.unsqueeze(1) + wx * z11.unsqueeze(1)  # along x
        # ground_height = ((1 - wy) * z0 + wy * z1).squeeze(1)      # along y
        
        # ground_level_odometry_frame = self.robot_api._init_pos[:, 2] - self.robot_api.default_root_state[:, 2]
        # self.reference_height = self.cfg.nominal_height + (ground_height - ground_level_odometry_frame)
        
        # squat motion
        randomize_duration = int(2.0/self._env.physics_dt) # 2sec
        time_step = self.mpc_counter % randomize_duration
        coef = torch.zeros(self.num_envs, device=self.device)
        coef[time_step<=randomize_duration//2] = time_step/(randomize_duration//2) # [0, 1]
        coef[time_step>randomize_duration//2] = (randomize_duration-time_step)/(randomize_duration//2) # [1, 0]
        offset = -0.15*coef
        self.reference_height = self.cfg.nominal_height + offset
        
    def _get_footplacement_height(self):
        sensor= self._env.scene.sensors["height_scanner"]
        height_map = sensor.data.ray_hits_w[..., 2]
        
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        target_pos = (self.foot_placement_b.reshape(self.num_envs, 2, 3)[:, :, :2] * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        
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
        # ground_height = torch.min(torch.stack([z00, z10, z01, z11], dim=1), dim=1).values
        
        ground_level_odometry_frame = self.robot_api._init_pos[:, 2] - self.robot_api.default_root_state[:, 2]
        self.foot_placement_height = torch.clamp(ground_height - ground_level_odometry_frame, 0.0, None)
    
    def visualize_marker(self):
        # prepare transformation matrices
        world_to_base_trans = self.robot_api.root_pos_w[:, :3].clone()
        world_to_base_rot = self.robot_api.root_rot_mat_w.clone()

        # visualize foot placement
        fp = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
        fp[:, 0, :3] = self.foot_placement[:, :3]
        fp[:, 1, :3] = self.foot_placement[:, 3:]
        # from base frame to world frame
        fp[:, 0, :] = (world_to_base_rot @ fp[:, 0, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans
        fp[:, 1, :] = (world_to_base_rot @ fp[:, 1, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans
        fp[:, :, 2] += 0.01 # better visibility
        fp[:, :, 2] = fp[:, :, 2] * (1-self.gait_contact) - 100 * self.gait_contact # hide foot placement of stance foot
        self.foot_placement_visualizer.visualize(fp)

        # visualize foot sole positions
        foot_pos = self.robot_api.foot_pos
        self.foot_position_visualizer.visualize(foot_pos)

        # visullize reference trajectory
        foot_traj = self.foot_position_trajectory.clone()
        for i in range(10):
            # from base frame to world frame
            foot_traj[:, i, :] = (world_to_base_rot @ foot_traj[:, i, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans
        self.foot_trajectory_visualizer.visualize(foot_traj)


    def apply_actions(self):
        # obtain state
        self._get_state()
        self.mpc_controller.update_state(self.state)
        self.mpc_controller.run_lowlevel()
        
        joint_actions = self.mpc_controller.get_action()
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

        # # define height of floating base as torso - stance foot distance 
        # fz = torch.abs(self.foot_pos_b.reshape(-1, 2, 3)[:, :, 2] * self.gait_contact) # (num_envs, 2)
        # self.root_pos[:, 2] = fz.max(dim=1).values
        
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
        
        self.grw_accel = self.mpc_controller.centroidal_accel
        self.grw = self.mpc_controller.ground_reaction_wrench.reshape(self.num_envs, 12)
        self.gait_contact = self.mpc_controller.contact_state
        self.swing_phase = self.mpc_controller.swing_phase
        self.foot_placement_w = self.mpc_controller.foot_placement.reshape(self.num_envs, -1)
        self.foot_placement = self.mpc_controller.foot_placement_b.reshape(self.num_envs, -1)
        self.foot_pos_b = self.mpc_controller.foot_pos_b.reshape(self.num_envs, 6)
        self.ref_foot_pos_b = self.mpc_controller.ref_foot_pos_b.reshape(self.num_envs, 6)
        self.mpc_cost = self.mpc_controller.mpc_cost.squeeze(-1)
        
        # base-leg angle
        stance_leg_r_left = torch.abs(self.foot_pos_b[:, :3]).clamp(min=1e-6)  # avoid division by zero
        stance_leg_r_right = torch.abs(self.foot_pos_b[:, 3:]).clamp(min=1e-6)  # avoid division by zero
        self.leg_angle[:, 0] = torch.abs(torch.atan2(stance_leg_r_left[:, 0], stance_leg_r_left[:, 2])) # left sagittal
        self.leg_angle[:, 1] = torch.abs(torch.atan2(stance_leg_r_left[:, 1], stance_leg_r_left[:, 2])) # left lateral
        self.leg_angle[:, 2] = torch.abs(torch.atan2(stance_leg_r_right[:, 0], stance_leg_r_right[:, 2])) # right sagittal
        self.leg_angle[:, 3] = torch.abs(torch.atan2(stance_leg_r_right[:, 1], stance_leg_r_right[:, 2])) # right lateral

        # swing leg traj
        self.foot_position_trajectory = self.mpc_controller.swing_foot_trajectory
    
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
        
        # reset controller
        self.mpc_controller.reset(env_ids) # type: ignore
        self.mpc_counter[env_ids] = 0