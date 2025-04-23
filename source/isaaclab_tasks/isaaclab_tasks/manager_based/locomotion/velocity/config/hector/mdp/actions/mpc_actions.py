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

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

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
        self._action_ub = torch.zeros_like(self._action_lb)
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
        
        # create tensors for mpc control counter
        self.mpc_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # create mpc state
        self.state = torch.zeros(self.num_envs, 33, device=self.device)
    
    
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
        
        positive_action_mask = (actions > 0).to(torch.float32)
        self._processed_actions[:] = positive_action_mask * self._action_ub * self._raw_actions + (1 - positive_action_mask) * self._action_lb * (-self._raw_actions)
        
        # split residual actions
        stepping_frequency = self._processed_actions[:, 0].cpu().numpy()
        swing_foot_height = self._processed_actions[:, 1].cpu().numpy()
        trajectory_control_points = self._processed_actions[:, 2].cpu().numpy()
        
        # form actual control parameters (nominal value + residual)
        stepping_frequency = self.cfg.nominal_stepping_frequency + stepping_frequency
        swing_foot_height = self.cfg.nominal_swing_height + swing_foot_height
        cp1 = self.cfg.nominal_cp1_coef + trajectory_control_points
        cp2 = self.cfg.nominal_cp2_coef + trajectory_control_points
        
        command = self._env.command_manager.get_command(self.cfg.command_name).cpu().numpy()
        for i in range(self.num_envs):
            self.mpc_controller[i].set_swing_parameters(stepping_frequency=stepping_frequency[i], foot_height=swing_foot_height[i], cp1=cp1[i], cp2=cp2[i])
            self.mpc_controller[i].set_command(
                gait_num=2, 
                roll_pitch=np.zeros(2, dtype=np.float32),
                twist=command[i],
                height=self.cfg.reference_height,
            )

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
        self.joint_effort = self.robot_api.joint_effort[:, self._joint_ids]
        
        # # foot body angle
        # self.leg_angle = torch.zeros(self.num_envs, 4, device=self.device)
        # stance_leg_r_left = torch.abs(self._foot_pos_b[:, :3])
        # stance_leg_r_right = torch.abs(self._foot_pos_b[:, 3:])
        # self.leg_angle[:, 0] = torch.abs(torch.atan2(stance_leg_r_left[:, 0], self._root_pos[:, 2]))
        # self.leg_angle[:, 1] = torch.abs(torch.atan2(stance_leg_r_left[:, 1], self._root_pos[:, 2]))
        # self.leg_angle[:, 2] = torch.abs(torch.atan2(stance_leg_r_right[:, 0], self._root_pos[:, 2]))
        # self.leg_angle[:, 3] = torch.abs(torch.atan2(stance_leg_r_right[:, 1], self._root_pos[:, 2]))
        
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
    
    def _add_joint_offset(self, joint_pos:torch.Tensor) -> torch.Tensor:
        joint_pos[:, 2] += torch.pi/4
        joint_pos[:, 3] -= torch.pi/2
        joint_pos[:, 4] += torch.pi/4
        
        joint_pos[:, 7] += torch.pi/4
        joint_pos[:, 8] -= torch.pi/2
        joint_pos[:, 9] += torch.pi/4
        
        return joint_pos

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        # reset mpc controller
        for i in range(self.num_envs):
            self.mpc_controller[i].reset()