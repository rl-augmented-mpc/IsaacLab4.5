# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs import ManagerBasedEnv


from . import mpc_actions_cfg
from .robot_helper import RobotCore
from .mpc_controller import MPC_Conf, MPCController
from isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp.marker import (
    FootPlacementVisualizer, 
    PositionTrajectoryVisualizer, 
    SwingFootVisualizer, 
    GridPointVisualizer
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp.util import log_score_filter



"""
Base Blind Locomotion class.
"""
class BlindLocomotionMPCAction(ActionTerm):

    cfg: mpc_actions_cfg.BlindLocomotionMPCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    
    def __init__(self, cfg: mpc_actions_cfg.BlindLocomotionMPCActionCfg, env: ManagerBasedEnv):
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
        
        # create mpc object array
        mpc_conf = MPC_Conf(
            control_dt=env.physics_dt, control_iteration_between_mpc=self.cfg.control_iteration_between_mpc, 
            horizon_length=self.cfg.horizon_length, mpc_decimation=int(env.step_dt//env.physics_dt))
        self.mpc_controller = [MPCController(mpc_conf) for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.mpc_controller[i].set_planner(self.cfg.foot_placement_planner)
            self.mpc_controller[i].set_swing_foot_reference(self.cfg.swing_foot_reference_frame)
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
        
        self.foot_placement_w = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self.foot_placement = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # in body frame]
        self.foot_pos_w = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # foot position in world frame
        self.foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # foot position in body frame
        self.foot_pos_b[:, [2, 5]] = -self.cfg.nominal_height # set reasonable initial value
        self.ref_foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # reference foot position in body frame
        self.ref_foot_pos_b[:, 5] = -self.cfg.nominal_height # set reasonable initial value
        self.ref_foot_pos_w = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32) # reference foot position in world frame
        self.leg_angle = torch.zeros(self.num_envs, 4, device=self.device)
        
        self.mpc_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.mpc_cost = torch.zeros(self.num_envs, device=self.device)
        self.position_trajectory = torch.zeros(self.num_envs, 10, 3, device=self.device, dtype=torch.float32) # trajectory of the foot placement in world frame
        self.orientation_trajectory = torch.zeros(self.num_envs, 10, 3, device=self.device, dtype=torch.float32) # trajectory of the foot placement in world frame
        self.foot_position_trajectory = torch.zeros(self.num_envs, 10, 3, device=self.device, dtype=torch.float32) # trajectory of the foot placement in body frame
        
        # reference
        self.twist = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.reference_height = self.cfg.nominal_height * np.ones(self.num_envs, dtype=np.float32)
        self.foot_placement_height = np.zeros(self.num_envs, dtype=np.float32)
        
        # markers
        self.foot_placement_visualizer = FootPlacementVisualizer("/Visuals/foot_placement")
        self.foot_position_visualizer = SwingFootVisualizer("/Visuals/foot_position")
        self.foot_trajectory_visualizer = PositionTrajectoryVisualizer("/Visuals/foot_trajectory", color=(0.0, 0.0, 1.0))
        
        # command
        self.command = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.original_command = self._env.command_manager.get_command(self.cfg.command_name)
    
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """
        mpc control parameters:
        - mpc sampling time
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
    
    # ** MDP loop **
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
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
        self.process_mpc_reference()
        
        # send updated parameters to MPC
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
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()
    
    def process_mpc_reference(self, sensor_name: str= "height_scanner"):
        self._get_reference_velocity()
        
    def _get_reference_velocity(self):
        # # ramp up
        # ramp_up_duration = 1.2 # seconds
        # ramp_up_coef = torch.clip(self.mpc_counter/int(ramp_up_duration/self._env.physics_dt), 0.0, 1.0).unsqueeze(1)
        # command = ramp_up_coef * self.original_command

        # no ramp up
        command = self.original_command
        
        # update command
        self.twist[:, :] = command.cpu().numpy()
        # update command manager
        self._env.command_manager._terms[self.cfg.command_name].vel_command_b = command
        
    def _get_reference_height(self, sensor_name: str = "height_scanner_fine"):
        raise NotImplementedError
        
    def _get_footplacement_height(self, sensor_name: str = "height_scanner_fine"):
        raise NotImplementedError
    
    def _process_heightmap(self):
        raise NotImplementedError
    
    def update_visual_marker(self):
        """
        Handle visualization of 
        - planned footholds
        - reference foot trajectory
        - current foot position
        """
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
        orientation = self.robot_api.root_quat_w[:, None, :].repeat(1, 2, 1).view(-1, 4)
        self.foot_placement_visualizer.visualize(fp, orientation)

        # visualize foot sole positions
        foot_pos = self.robot_api.foot_pos
        self.foot_position_visualizer.visualize(foot_pos)

        # visullize reference trajectory
        position_traj = self.position_trajectory.clone()
        foot_traj = self.foot_position_trajectory.clone()
        for i in range(10):
            # from base frame to world frame
            position_traj[:, i, :] = (world_to_base_rot @ position_traj[:, i, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans
            foot_traj[:, i, :] = (world_to_base_rot @ foot_traj[:, i, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans

        self.foot_trajectory_visualizer.visualize(foot_traj)
        
    # ** physics loop **
    def apply_actions(self):
        self._get_state()
        # compute mpc
        for i in range(self.num_envs):
            self.mpc_controller[i].update_state(self.state[i].cpu().numpy())
            self.mpc_controller[i].run()
            self._joint_actions[i] = self.mpc_controller[i].get_action()
        joint_actions = torch.from_numpy(self._joint_actions).to(self.device)
        self.robot_api.set_joint_effort_target(joint_actions, self._joint_ids)
        self.mpc_counter += 1
        
        self._get_mpc_state()
    
    def _get_state(self) -> None:
        """
        Get robot's center of mass state and joint state. 
        NOTE that the center of mass pose is relative to the initial spawn position (i.e. odometry adding nominal height as offset). 
        """
        
        self.root_rot_mat = self.robot_api.root_rot_mat_local
        self.root_quat = self.robot_api.root_quat_local
        self.root_yaw = self.robot_api.root_yaw_local
        self.root_pos = self.robot_api.root_pos_local

        # # define heigh tof floating base as mean of each foot position
        # fz = torch.abs(self.foot_pos_b.reshape(-1, 2, 3)[:, :, 2]) # (num_envs, 2)
        # self.root_pos[:, 2] = fz.mean(dim=1)

        # # define heigh tof floating base as max of each foot position as done in hardware
        # fz = torch.abs(self.foot_pos_b.reshape(-1, 2, 3)[:, :, 2]) # (num_envs, 2)
        # self.root_pos[:, 2] = fz.max(dim=1).values

        # define height of floating base as torso - stance foot distance 
        fz = torch.abs(self.foot_pos_b.reshape(-1, 2, 3)[:, :, 2] * self.gait_contact) # (num_envs, 2)
        self.root_pos[:, 2] = fz.max(dim=1).values

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
        foot_placement = []
        foot_pos_b = []
        foot_ref_pos_b = []
        mpc_cost = []
        
        position_traj = []
        orientation_traj = []
        foot_position_traj = []
        
        for i in range(len(self.mpc_controller)):
            grw_accel.append(self.mpc_controller[i].accel_gyro(self.root_rot_mat[i].cpu().numpy()))
            grw.append(self.mpc_controller[i].grfm)
            gait_contact.append(self.mpc_controller[i].contact_state)
            swing_phase.append(self.mpc_controller[i].swing_phase)
            foot_placement.append(self.mpc_controller[i].foot_placement_base)
            foot_pos_b.append(self.mpc_controller[i].foot_pos_b)
            foot_ref_pos_b.append(self.mpc_controller[i].ref_foot_pos_b)
            mpc_cost.append(self.mpc_controller[i].mpc_cost)
            pos_traj, ori_traj, foot_traj = self.mpc_controller[i].reference_trajectory
            position_traj.append(pos_traj)
            orientation_traj.append(ori_traj)
            foot_position_traj.append(foot_traj)
        
        self.grw_accel = torch.from_numpy(np.array(grw_accel)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self.grw = torch.from_numpy(np.array(grw)).to(self.device).view(self.num_envs, 12).to(torch.float32)
        self.gait_contact = torch.from_numpy(np.array(gait_contact)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self.swing_phase = torch.from_numpy(np.array(swing_phase)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self.foot_placement = torch.from_numpy(np.array(foot_placement)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self.foot_pos_b = torch.from_numpy(np.array(foot_pos_b)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self.ref_foot_pos_b = torch.from_numpy(np.array(foot_ref_pos_b)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self.position_trajectory = torch.from_numpy(np.array(position_traj)).to(self.device).view(self.num_envs, 10, 3).to(torch.float32)
        self.orientation_trajectory = torch.from_numpy(np.array(orientation_traj)).to(self.device).view(self.num_envs, 10, 3).to(torch.float32)
        self.foot_position_trajectory = torch.from_numpy(np.array(foot_position_traj)).to(self.device).view(self.num_envs, 10, 3).to(torch.float32)
        
        # transform foot position in body frame to odometry frame
        self.ref_foot_pos_w[:, :3] = (self.root_rot_mat @ self.ref_foot_pos_b[:, :3].unsqueeze(-1)).squeeze(-1) + self.root_pos[:, :3]
        self.ref_foot_pos_w[:, 3:] = (self.root_rot_mat @ self.ref_foot_pos_b[:, 3:].unsqueeze(-1)).squeeze(-1) + self.root_pos[:, :3]
        
        # body-leg angle
        stance_leg_r_left = torch.abs(self.foot_pos_b[:, :3]).clamp(min=1e-6)  # avoid division by zero
        stance_leg_r_right = torch.abs(self.foot_pos_b[:, 3:]).clamp(min=1e-6)  # avoid division by zero
        self.leg_angle[:, 0] = torch.abs(torch.atan2(stance_leg_r_left[:, 0], stance_leg_r_left[:, 2])) # left sagittal
        self.leg_angle[:, 1] = torch.abs(torch.atan2(stance_leg_r_left[:, 1], stance_leg_r_left[:, 2])) # left lateral
        self.leg_angle[:, 2] = torch.abs(torch.atan2(stance_leg_r_right[:, 0], stance_leg_r_right[:, 2])) # right sagittal
        self.leg_angle[:, 3] = torch.abs(torch.atan2(stance_leg_r_right[:, 1], stance_leg_r_right[:, 2])) # right lateral
        
        # compute mpc cost
        self.mpc_cost = torch.from_numpy(np.array(mpc_cost)).to(self.device).view(self.num_envs).to(torch.float32)

        # get state for visualization 
        self.foot_pos_w[:, :3] = self.robot_api.foot_pos_local[:, 0, :]
        self.foot_pos_w[:, 3:] = self.robot_api.foot_pos_local[:, 1, :]
    
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
        self.robot_api.reset_default_pose(self.robot_api.root_state_w[env_ids, :7], env_ids) # type: ignore

        # reset action
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        
        # reset command
        self._env.command_manager._terms[self.cfg.command_name]._resample_command(env_ids) # type: ignore
        self.original_command[env_ids, :] = self._env.command_manager.get_command(self.cfg.command_name)[env_ids, :]
        
        # reset mpc controller
        self.mpc_counter[env_ids] = 0
        self._get_state()
        for i in env_ids.cpu().numpy(): # type: ignore
            self.mpc_controller[i].reset()
            self.mpc_controller[i].update_gait_parameter(
                np.array([self.cfg.double_support_duration, self.cfg.double_support_duration]), 
                np.array([self.cfg.single_support_duration, self.cfg.single_support_duration]),)
            self.mpc_controller[i].switch_fsm("passive")
            self.mpc_controller[i].update_state(self.state[i].cpu().numpy())
            self.mpc_controller[i].run()
        self._get_mpc_state()
        
        # switch to walking mode again
        for i in range(self.num_envs):
            self.mpc_controller[i].switch_fsm("walking")




"""
Perceptive Locomotion class in a sense that the agent has access to terrain elevation map.
"""
class PerceptiveLocomotionMPCAction(BlindLocomotionMPCAction):

    cfg: mpc_actions_cfg.PerceptiveLocomotionMPCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    
    def __init__(self, cfg: mpc_actions_cfg.PerceptiveLocomotionMPCActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        
        # # heightmap related
        # self.grid_point_boundary = torch.empty(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        # self.grid_point_boundary_in_body = torch.empty(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)

        # num_samples = 10
        # self.grid_point_world = torch.empty(self.num_envs, 2*num_samples, 3, device=self.device, dtype=torch.float32)
        # self.grid_point_height = torch.empty(self.num_envs, 2*num_samples, device=self.device, dtype=torch.float32)
        # self.grid_point_visualizer = GridPointVisualizer("/Visuals/safe_region", color=(0.0, 0.3, 0.0))
    
    """
    Operations.
    """
    
    # ** MDP loop **
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
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
        self.process_mpc_reference()
        
        # send updated parameters to MPC
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
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()

    def process_mpc_reference(self, sensor_name: str= "height_scanner"):
        self._get_reference_velocity()
        self._get_reference_height(sensor_name=sensor_name)
        self._get_footplacement_height(sensor_name=sensor_name)
        # self._process_heightmap(sensor_name="height_scanner_fine")
        
    def _get_reference_velocity(self):
        # # ramp up
        # ramp_up_duration = 1.2/self._env.physics_dt
        # ramp_up_coef = torch.clip(self.mpc_counter/ramp_up_duration, 0.0, 1.0).unsqueeze(1)
        # command = ramp_up_coef * self.original_command

        # no ramp up
        command = self.original_command
        
        # update command
        self.twist[:, :] = command.cpu().numpy()
        # update command manager
        self._env.command_manager._terms[self.cfg.command_name].vel_command_b = command
        
    
    def _get_reference_height(self, sensor_name: str = "height_scanner_fine"):
        """
        project current stance foot position to heightmap and sample the height.
        Then, add this value to nominal height H which gives reference height 
        z_ref = H + z_ground.
        """

        def bilinear_interpolation(heightmap: torch.Tensor, p: torch.Tensor, map_resolution:float, width:int, height:int, scan_offset:tuple[int, int]=(0, 0)) -> torch.Tensor:
            x_img = p[:, 0] / map_resolution + (width // 2 - scan_offset[0])
            y_img = p[:, 1] / map_resolution + (height // 2 - scan_offset[1])

            # Clamp to valid range before ceil/floor to avoid out-of-bounds
            x0 = x_img.floor().clamp(0, width - 2)
            x1 = (x0 + 1).clamp(0, width - 1)
            y0 = y_img.floor().clamp(0, height - 2)
            y1 = (y0 + 1).clamp(0, height - 1)

            # Calculate weights for interpolation
            wx = (x_img - x0).unsqueeze(1)  # [N]
            wy = (y_img - y0).unsqueeze(1)  # [N]

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
            z00 = heightmap[torch.arange(self.num_envs), idx00]
            z10 = heightmap[torch.arange(self.num_envs), idx10]
            z01 = heightmap[torch.arange(self.num_envs), idx01]
            z11 = heightmap[torch.arange(self.num_envs), idx11]

            wx = (x_img - x0)
            wy = (y_img - y0)

            z0 = (1 - wx) * z00 + wx * z10  # along x
            z1 = (1 - wx) * z01 + wx * z11  # along x
            return (1 - wy) * z0 + wy * z1

        sensor= self._env.scene.sensors[sensor_name]
        height_map = sensor.data.ray_hits_w[..., 2].clone()
        
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))

        ground_level_odometry_frame = self.robot_api._init_pos[:, 2] # type: ignore

        target_pos = (self.foot_pos_b.reshape(self.num_envs, 2, 3) * (self.gait_contact==1).unsqueeze(2)).sum(dim=1)
        target_pos[:, 0] += 0.07
        # ground_height = bilinear_interpolation(height_map, target_pos, scan_resolution, width, height, scan_offset) - ground_level_odometry_frame

        # plane fitting
        target_pos_1 = self.foot_placement.reshape(self.num_envs, 2, 3)[:, 0, :].clone()
        target_pos_2 = self.foot_placement.reshape(self.num_envs, 2, 3)[:, 1, :].clone()
        ground_height_1 = bilinear_interpolation(height_map, target_pos_1, scan_resolution, width, height, scan_offset) - ground_level_odometry_frame
        ground_height_2 = bilinear_interpolation(height_map, target_pos_2, scan_resolution, width, height, scan_offset) - ground_level_odometry_frame
        dx = target_pos_2[:, 0] - target_pos_1[:, 0]
        dy = target_pos_2[:, 1] - target_pos_1[:, 1]
        dz = ground_height_2 - ground_height_1
        
        a = dz * dx / (dx**2 + dy**2 + 1e-6)
        b = dz * dy / (dx**2 + dy**2 + 1e-6)
        c = ground_height_1 - a * target_pos_1[:, 0] - b * target_pos_1[:, 1]

        # query height
        xq, yq = target_pos[:, 0], target_pos[:, 1]
        ground_height = a * xq + b * yq + c

        self.reference_height = self.cfg.nominal_height + ground_height.cpu().numpy()
        
    def _get_footplacement_height(self, sensor_name: str = "height_scanner_fine"):
        sensor= self._env.scene.sensors[sensor_name]
        height_map = sensor.data.ray_hits_w[..., 2].clone()
        
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        target_pos = (self.foot_placement.reshape(self.num_envs, 2, 3) * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        
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
        
        ground_level_odometry_frame = self.robot_api._init_pos[:, 2]
        self.foot_placement_height = (ground_height - ground_level_odometry_frame).cpu().numpy()
    
    # def _process_heightmap(self):
    #     sensor= self._env.scene.sensors["height_scanner_fine"]
    #     scan_width, scan_height = sensor.cfg.pattern_cfg.size
    #     scan_resolution = sensor.cfg.pattern_cfg.resolution
    #     width = int(scan_width/scan_resolution + 1)
    #     height = int(scan_height/scan_resolution + 1)
        
    #     world_to_odom_trans = self.robot_api._init_pos[:, :3].clone()
    #     world_to_odom_rot = self.robot_api._init_rot.clone()
        
    #     grid_point = sensor.data.ray_hits_w[..., :3] # in sim world frame
    #     grid_point_local = (world_to_odom_rot[:, None, :, :].transpose(2, 3) @ (grid_point - world_to_odom_trans[:, None, :]).unsqueeze(-1)).squeeze(-1) # sim world to odometry
    #     grid_point_local = grid_point_local - self.root_pos[:, None, :3] # offset to odometry frame
    #     grid_point_local[:, :, 2] += self.root_pos[:, None, 2] # add nominal height offset
    #     grid_point_local = (self.root_rot_mat[:, None, :, :].transpose(2, 3) @ grid_point_local.unsqueeze(-1)).squeeze(-1) # odometry to body frame
        
    #     elevation_map = sensor.data.ray_hits_w[..., 2].view(-1, 1, height, width)
    #     log_score = log_score_filter(elevation_map, alpha=50.0).view(-1, height*width)
    #     unsafe_region = log_score < 0.6
        
    #     grid_point_boundary = grid_point.clone()
    #     grid_point_boundary_in_body = grid_point_local.clone()
        
    #     # flatten 
    #     N = grid_point_boundary.shape[1]
    #     grid_point_boundary = grid_point_boundary.view(-1, 3)
    #     grid_point_boundary_in_body = grid_point_boundary_in_body.view(-1, 3)
    #     unsafe_region = unsafe_region.view(-1)
        
    #     # push safe region to edge of elev map
    #     grid_point_boundary[~unsafe_region, 2] = -1.0
    #     grid_point_boundary_in_body[~unsafe_region, :2] = -0.5
        
    #     # reshape
    #     grid_point_boundary = grid_point_boundary.view(self.num_envs, N, 3)
    #     grid_point_boundary_in_body = grid_point_boundary_in_body.view(self.num_envs, N, 3)
        
    #     self.grid_point_boundary = grid_point_boundary.clone()
    #     self.grid_point_boundary_in_body = grid_point_boundary_in_body.clone()

    def _process_heightmap(self, sensor_name: str = "height_scanner_fine"):
        """
        Get height samples along lines connecting current foot position and planned footholds.
        """
        sensor = self._env.scene.sensors[sensor_name]
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width / scan_resolution + 1)
        height = int(scan_height / scan_resolution + 1)
        height_scan = sensor.data.ray_hits_w[..., 2].clone().reshape(self.num_envs, -1)  # shape: (num_envs, width * height)

        # Start and end points for sampling
        start_point = self.foot_pos_b.reshape(self.num_envs, 2, 3).clone()
        end_point = self.foot_placement.reshape(self.num_envs, 2, 3).clone()
        vec = end_point - start_point  # (num_envs, 2, 3)
        num_samples = 10

        # Generate sample points between foot and foothold
        sample_steps = torch.linspace(0, 1, num_samples, device=self.device).reshape(1, 1, num_samples, 1)
        sample_points = start_point[:, :, None, :] + vec[:, :, None, :] * sample_steps
        sample_points = sample_points.reshape(self.num_envs, 2 * num_samples, 3)  # (num_envs, num_samples*2, 3)

        # Convert xy to grid indices
        sample_points_xy = sample_points[:, :, :2].clone()
        sample_points_xy[:, :, 0] = sample_points_xy[:, :, 0] / scan_resolution + (width // 2 - int(sensor.cfg.offset.pos[0] / scan_resolution))
        sample_points_xy[:, :, 1] = sample_points_xy[:, :, 1] / scan_resolution + (height // 2 - int(sensor.cfg.offset.pos[1] / scan_resolution))

        # Clamp and compute surrounding pixel indices
        x0 = sample_points_xy[:, :, 0].floor().clamp(0, width - 2)
        x1 = (x0 + 1).clamp(0, width - 1)
        y0 = sample_points_xy[:, :, 1].floor().clamp(0, height - 2)
        y1 = (y0 + 1).clamp(0, height - 1)

        wx = (sample_points_xy[:, :, 0] - x0)
        wy = (sample_points_xy[:, :, 1] - y0)

        x0 = x0.long()
        x1 = x1.long()
        y0 = y0.long()
        y1 = y1.long()

        # Compute flattened indices
        idx00 = y0 * width + x0
        idx10 = y0 * width + x1
        idx01 = y1 * width + x0
        idx11 = y1 * width + x1

        # Use torch.gather for batched index access
        z00 = torch.gather(height_scan, 1, idx00)
        z10 = torch.gather(height_scan, 1, idx10)
        z01 = torch.gather(height_scan, 1, idx01)
        z11 = torch.gather(height_scan, 1, idx11)

        # Bilinear interpolation
        z0 = (1 - wx) * z00 + wx * z10
        z1 = (1 - wx) * z01 + wx * z11
        ground_height = (1 - wy) * z0 + wy * z1  # shape: (num_envs, num_samples * 2)

        offset = 0.56
        self.grid_point_height = sensor.data.pos_w[:, 2].unsqueeze(1) - ground_height - offset

        # Form samples for visualization
        height_samples = torch.cat([sample_points[:, :, :2], torch.zeros_like(sample_points[:, :, 2:])], dim=2)  # (num_envs, num_samples*2, 3)

        # Transform to world frame
        world_to_base_trans = self.robot_api.root_pos_w[:, :3].clone()
        world_to_base_rot = self.robot_api.root_rot_mat_w.clone()  # shape: (num_envs, 3, 3)
        height_samples = (world_to_base_rot[:, None, :, :] @ height_samples.unsqueeze(-1)).squeeze(-1) + world_to_base_trans[:, None, :]
        height_samples[:, :, 2] = ground_height
        self.grid_point_world = height_samples.clone()


    def update_visual_marker(self):
        # prepare transformation matrices
        world_to_odom_trans = self.robot_api._init_pos[:, :3].clone()
        world_to_odom_rot = self.robot_api._init_rot.clone()
        world_to_base_trans = self.robot_api.root_pos_w[:, :3].clone()
        world_to_base_rot = self.robot_api.root_rot_mat_w.clone()
        
        sensor= self._env.scene.sensors["height_scanner"]
        height_map = sensor.data.ray_hits_w[..., 2]
        scan_width, scan_height = sensor.cfg.pattern_cfg.size
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        width = int(scan_width/scan_resolution + 1)
        height = int(scan_height/scan_resolution + 1)
        scan_offset = (int(sensor.cfg.offset.pos[0]/scan_resolution), int(sensor.cfg.offset.pos[1]/scan_resolution))
        
        foot_height = []
        for foot in range(2):
            target_pos = self.foot_placement[:, 3*foot:3*(foot+1)]    
            x_img = target_pos[:, 0] / scan_resolution + (width // 2 - scan_offset[0]) # col
            y_img = target_pos[:, 1] / scan_resolution + (height // 2 - scan_offset[1]) # row

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
            ground_height = ((1 - wy) * z0 + wy * z1).squeeze(1).clip(-1.0, 1.0)      # along y
            foot_height.append(ground_height) # this is rel to simulation world frame
        
        # visualize foot placement
        fp = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
        fp[:, 0, :3] = self.foot_placement[:, :3]
        fp[:, 1, :3] = self.foot_placement[:, 3:]
        fp[:, 0, :] = (world_to_base_rot @ fp[:, 0, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans
        fp[:, 1, :] = (world_to_base_rot @ fp[:, 1, :].unsqueeze(-1)).squeeze(-1) + world_to_base_trans
        fp[:, :, 2] = fp[:, :, 2] * (1-self.gait_contact) - 100 * self.gait_contact # hide foot placement of stance foot
        orientation = self.robot_api.root_quat_w[:, None, :].repeat(1, 2, 1).view(-1, 4)
        self.foot_placement_visualizer.visualize(fp, orientation)

        # visualize foot position
        foot_pos = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
        foot_pos[:, 0, :] = (world_to_odom_rot @ self.foot_pos_w[:, :3].unsqueeze(-1)).squeeze(-1) + world_to_odom_trans
        foot_pos[:, 1, :] = (world_to_odom_rot @ self.foot_pos_w[:, 3:6].unsqueeze(-1)).squeeze(-1) + world_to_odom_trans
        self.foot_position_visualizer.visualize(foot_pos)

        # visualize foot trajectory
        position_traj_world = self.position_trajectory.clone()
        foot_traj_world = self.foot_position_trajectory.clone()
        for i in range(10):
            position_traj_world[:, i, :] = (world_to_odom_rot @ position_traj_world[:, i, :].unsqueeze(-1)).squeeze(-1) + world_to_odom_trans
            foot_traj_world[:, i, :] = (world_to_odom_rot @ foot_traj_world[:, i, :].unsqueeze(-1)).squeeze(-1) + world_to_odom_trans
        self.foot_trajectory_visualizer.visualize(foot_traj_world)
        
        # # visualize grid points
        # self.grid_point_visualizer.visualize(self.grid_point_world)

    # ** physics loop **
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



"""
Sub-class that inherits from BlindLocomotionMPCAction/PerceptiveLocomotionMPCAction. 
Only difference is the number of action space and the way to process actions.
"""

class BlindLocomotionMPCAction2(BlindLocomotionMPCAction):
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
        - mpc sampling time (R^1)
        - swing foot height (R^1)
        - swing trajectory control points (R^1)
        """
        return 9
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
        # transform to specific range
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
        self.process_mpc_reference()
        
        # send updated parameters to MPC
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_srbd_residual(A_residual=A_residual[i], B_residual=B_residual[i])
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i])
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()


class BlindLocomotionMPCAction3(BlindLocomotionMPCAction):
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
        - mpc sampling time (R^1)
        - swing foot height (R^1)
        - swing trajectory control points (R^1)
        - body velocity ratio (R^1)
        """
        return 4
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
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
        self.process_mpc_reference()
        
        # send updated parameters to MPC
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i], 
                )
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()
        
    def _get_reference_velocity(self):
        """
        Compute reference velocity as
        \tilde{v} = v_0 * (1 + \delta{v})
        -1.5 <= \delta{v} <= 0.5
        """
        # get reference body velocity from policy
        rx = self._processed_actions[:, 3]
        self.command[:, 0] = self.original_command[:, 0] * (1 + rx)
        self.command[:, 1] = self.original_command[:, 1]
        self.command[:, 2] = self.original_command[:, 2]
        
        # update command
        self.twist[:, :] = self.command.cpu().numpy()
        # update command manager
        self._env.command_manager._terms[self.cfg.command_name].vel_command_b = self.command
        
class BlindLocomotionMPCAction4(BlindLocomotionMPCAction):
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
        - mpc sampling time (R^1)
        - swing foot height (R^1)
        - swing trajectory control points (R^1)
        - body velocity ratio (R^2)
        """
        return 11
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        # split processed actions into individual control parameters
        A_residual = np.zeros((self.num_envs, 13, 13), dtype=np.float32)
        B_residual = np.zeros((self.num_envs, 13, 12), dtype=np.float32)
        
        # transform accel to global frame
        centroidal_lin_acc = self._processed_actions[:, :3]
        centroidal_ang_acc = self._processed_actions[:, 3:6]
        centroidal_lin_acc = torch.bmm(self.root_rot_mat, centroidal_lin_acc.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_acc = torch.bmm(self.root_rot_mat, centroidal_ang_acc.unsqueeze(-1)).squeeze(-1)
        A_residual[:, 6:9, -1] = centroidal_lin_acc.cpu().numpy()
        A_residual[:, 9:12, -1] = centroidal_ang_acc.cpu().numpy()
        
        sampling_time = self.cfg.nominal_mpc_dt * (1 + self._processed_actions[:, 6].cpu().numpy())
        swing_foot_height = self._processed_actions[:, 7].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, 8].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update reference
        self.process_mpc_reference()
        
        # send updated parameters to MPC
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_srbd_residual(
                A_residual=A_residual[i], 
                B_residual=B_residual[i], 
                )
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i], 
                )
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()
        
    def _get_reference_velocity(self):
        """
        Compute reference velocity as
        \tilde{v} = v_0 * (1 + \delta{v})
        -1.5 <= \delta{v} <= 0.5
        """
        # get reference body velocity from policy
        self.command[:, 0] = self.original_command[:, 0] * (1 + self._processed_actions[:, 9])
        self.command[:, 2] = self.original_command[:, 2] * (1 + self._processed_actions[:, 10])
        
        # update command
        self.twist[:, :] = self.command.cpu().numpy()
        # update command manager
        self._env.command_manager._terms[self.cfg.command_name].vel_command_b = self.command



class PerceptiveLocomotionMPCAction2(PerceptiveLocomotionMPCAction):
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
        - mpc sampling time (R^1)
        - swing foot height (R^1)
        - swing trajectory control points (R^1)
        """
        return 9
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
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
        self.process_mpc_reference()
        
        # send updated parameters to MPC
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_srbd_residual(A_residual=A_residual[i], B_residual=B_residual[i])
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i])
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()


class PerceptiveLocomotionMPCAction3(PerceptiveLocomotionMPCAction):
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
        - mpc sampling time (R^1)
        - swing foot height (R^1)
        - swing trajectory control points (R^1)
        - body velocity ratio (R^2)
        """
        return 5
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
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
        self.process_mpc_reference()
        
        # send updated parameters to MPC
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i], 
                )
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()
        
    def _get_reference_velocity(self):
        """
        Compute reference velocity as
        \tilde{v} = v_0 * (1 + \delta{v})
        -1.5 <= \delta{v} <= 0.5
        """
        # get reference body velocity from policy
        rx = self._processed_actions[:, 3]
        rz = self._processed_actions[:, 4]
        self.command[:, 0] = self.original_command[:, 0] * (1 + rx)
        self.command[:, 1] = self.original_command[:, 1]
        self.command[:, 2] = self.original_command[:, 2] * (1 + rz)
        
        # update command
        self.twist[:, :] = self.command.cpu().numpy()
        # update command manager
        self._env.command_manager._terms[self.cfg.command_name].vel_command_b = self.command
        
class PerceptiveLocomotionMPCAction4(PerceptiveLocomotionMPCAction):
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
        - mpc sampling time (R^1)
        - swing foot height (R^1)
        - swing trajectory control points (R^1)
        - body velocity ratio (R^2)
        """
        return 11
    
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clone()
        # clip negative action value
        negative_action_clip_idx = self.cfg.negative_action_clip_idx
        if negative_action_clip_idx is not None:
            self._raw_actions[:, negative_action_clip_idx] = self._raw_actions[:, negative_action_clip_idx].clamp(0.0, 1.0) # clip negative value
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        # split processed actions into individual control parameters
        A_residual = np.zeros((self.num_envs, 13, 13), dtype=np.float32)
        B_residual = np.zeros((self.num_envs, 13, 12), dtype=np.float32)
        
        # transform accel to global frame
        centroidal_lin_acc = self._processed_actions[:, :3]
        centroidal_ang_acc = self._processed_actions[:, 3:6]
        centroidal_lin_acc = torch.bmm(self.root_rot_mat, centroidal_lin_acc.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_acc = torch.bmm(self.root_rot_mat, centroidal_ang_acc.unsqueeze(-1)).squeeze(-1)
        A_residual[:, 6:9, -1] = centroidal_lin_acc.cpu().numpy()
        A_residual[:, 9:12, -1] = centroidal_ang_acc.cpu().numpy()
        
        sampling_time = self.cfg.nominal_mpc_dt * (1 + self._processed_actions[:, 6].cpu().numpy())
        swing_foot_height = self._processed_actions[:, 7].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, 8].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update reference
        self.process_mpc_reference()
        
        # send updated parameters to MPC
        for i in range(self.num_envs):
            self.mpc_controller[i].update_sampling_time(sampling_time[i])
            self.mpc_controller[i].set_srbd_residual(
                A_residual=A_residual[i], 
                B_residual=B_residual[i], 
                )
            self.mpc_controller[i].set_swing_parameters(
                stepping_frequency=1.0, 
                foot_height=swing_foot_height[i], 
                cp1=cp1[i], 
                cp2=cp2[i], 
                pf_z=self.foot_placement_height[i], 
                )
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.twist[i],
                height=self.reference_height[i],
            )
            
        self.update_visual_marker()
        
    def _get_reference_velocity(self):
        """
        Compute reference velocity as
        \tilde{v} = v_0 * (1 + \delta{v})
        -1.5 <= \delta{v} <= 0.5
        """
        # get reference body velocity from policy
        self.command[:, 0] = self.original_command[:, 0] * (1 + self._processed_actions[:, 9])
        self.command[:, 2] = self.original_command[:, 2] * (1 + self._processed_actions[:, 10])
        
        # update command
        self.twist[:, :] = self.command.cpu().numpy()
        # update command manager
        self._env.command_manager._terms[self.cfg.command_name].vel_command_b = self.command