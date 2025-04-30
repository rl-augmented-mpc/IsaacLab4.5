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
        self.robot_api = RobotCore(self._asset, self.num_envs, torch.tensor([19, 20], device=self.device, dtype=torch.long))

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
            control_dt=env.physics_dt, control_iteration_between_mpc=int(0.2/(10*env.physics_dt)), 
            horizon_length=10, mpc_decimation=int(env.step_dt//env.physics_dt))
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
        self.reference_height = self.cfg.reference_height * np.ones(self.num_envs, dtype=np.float32)
    
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
        self._processed_actions[:] = self._action_lb + (self._raw_actions + 1) * (self._action_ub - self._action_lb) / 2
        
        # split processed actions into individual control parameters
        stepping_frequency = self._processed_actions[:, 0].cpu().numpy()
        swing_foot_height = self._processed_actions[:, 1].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, 2].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        stepping_frequency = self.cfg.nominal_stepping_frequency + stepping_frequency
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        # update mpc state
        ramp_up_duration = 0.5 # seconds
        ramp_up_coef = torch.clip(self.mpc_counter/int(ramp_up_duration/self._env.physics_dt), 0.0, 1.0).unsqueeze(1)
        self.command[:, :] = (ramp_up_coef * self._env.command_manager.get_command(self.cfg.command_name)).cpu().numpy()
        
        for i in range(self.num_envs):
            self.mpc_controller[i].set_swing_parameters(stepping_frequency=stepping_frequency[i], foot_height=swing_foot_height[i], cp1=cp1[i], cp2=cp2[i])
            self.mpc_controller[i].set_command(
                gait_num=2, #1:standing, 2:walking
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=self.command[i],
                height=self.reference_height[i],
            )
        
        self._get_mpc_state()
        self._get_height_at_swing_foot()
    
    def _get_height_at_swing_foot(self):
        scan_width = 1.0
        scan_resolution = 0.1
        
        sensor= self._env.scene.sensors["height_scanner"]
        height_map = sensor.data.ray_hits_w[..., 2]
        
        swing_foot_pos = (self.foot_pos_b.reshape(self.num_envs, 2, 3) * (self.gait_contact==0).unsqueeze(2)).sum(dim=1)
        swing_foot_pos[:, 0] += 0.05 # track toe pos
        
        px = (swing_foot_pos[:, 0]//scan_resolution).long() + int(scan_width//scan_resolution+1)/2
        py = -(swing_foot_pos[:, 1]//scan_resolution).long() + int(scan_width//scan_resolution+1)/2
        indices = (int(scan_width/scan_resolution + 1)*px + py).long()
        self.reference_height = self.cfg.reference_height + height_map[torch.arange(self.num_envs), indices].cpu().numpy() # type: ignore

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
        
        # define height as torso-to-base distance
        self.root_pos = self.robot_api.root_pos_local
        # https://github.gatech.edu/GeorgiaTechLIDARGroup/HECTOR_HW_new/blob/Unified_Framework/Interface/HW_interface/src/stateestimator/PositionVelocityEstimator.cpp
        # Is this right for non-flat terrain? probably not...
        # toe_index, _ = self._robot.find_bodies(["L_toe", "R_toe"], preserve_order=True)
        # foot_pos = (self._robot_api.body_pos_w[:, toe_index, 2]-0.04) - self._robot_api.root_pos_w [:, 2].view(-1, 1) # com to foot in world frame
        # phZ, _ = torch.max(-foot_pos, dim=1)
        # self._root_pos[:, 2] = phZ
        
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
        self.leg_angle[:, 0] = torch.abs(torch.atan2(stance_leg_r_left[:, 0], stance_leg_r_left[:, 2]))
        self.leg_angle[:, 1] = torch.abs(torch.atan2(stance_leg_r_left[:, 1], stance_leg_r_left[:, 2]))
        self.leg_angle[:, 2] = torch.abs(torch.atan2(stance_leg_r_right[:, 0], stance_leg_r_right[:, 2]))
        self.leg_angle[:, 3] = torch.abs(torch.atan2(stance_leg_r_right[:, 1], stance_leg_r_right[:, 2]))
        
        # compute mpc cost
        self.mpc_cost = torch.from_numpy(np.array(mpc_cost)).to(self.device).view(self.num_envs).to(torch.float32)
    
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
        # reset mpc controller
        for i in env_ids.cpu().numpy(): # type: ignore
            self.mpc_controller[i].reset()
            self.mpc_controller[i].update_gait_parameter(
                np.array([int(self.cfg.double_support_duration/self._env.physics_dt), int(self.cfg.double_support_duration/self._env.physics_dt)]), 
                np.array([int(self.cfg.single_support_duration/self._env.physics_dt), int(self.cfg.single_support_duration/self._env.physics_dt)]),)
        self.mpc_counter[env_ids] = 0