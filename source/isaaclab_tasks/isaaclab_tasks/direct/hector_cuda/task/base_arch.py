# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence
import carb

# Isaaclab core
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

# Task core
from isaaclab_tasks.direct.hector.common.robot_core import RobotCore
from isaaclab_tasks.direct.hector_cuda.hector_pytorch.simulation.mpc_wrapper import MPCWrapper

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.base_arch_cfg import BaseArchCfg



class BaseArch(DirectRLEnv):
    cfg: BaseArchCfg
    
    def __init__(self, cfg: BaseArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # get joint_ids that maps sim joint order to controller joint order
        self._joint_ids, self._joint_names = self._robot.find_joints(self.cfg.joint_names, preserve_order=True)
        
        # actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions_op = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._joint_actions = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device)
        self.action_lb = torch.tensor(self.cfg.action_lb, device=self.device, dtype=torch.float32)
        self.action_ub = torch.tensor(self.cfg.action_ub, device=self.device, dtype=torch.float32)
        
        # state
        self._root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._ref_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._root_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._init_rot_mat = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._root_rot_mat = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._root_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._ref_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._root_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._root_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._desired_root_lin_vel_b = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._desired_root_ang_vel_b = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._desired_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._joint_pos = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_vel = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_effort = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_accel = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._state = torch.zeros(self.num_envs, self.cfg.num_states, device=self.device, dtype=torch.float32)
        
        # RL observation
        self._obs = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_observation_space["policy"]), device=self.device)
        
        # MPC output
        self._grfm_mpc = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float32)
        self._accel_gyro_mpc = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._gait_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._swing_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._reibert_fps = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._augmented_fps = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32) # reibert + residual
        
        # Buffer for residual learning
        self._joint_action_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        self._grfm_residuals = np.zeros((self.num_envs, 12), dtype=np.float32)
        self._accel_peturbation = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._ang_accel_peturbation = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._A_residual = np.zeros((self.num_envs, 13, 13), dtype=np.float32)
        self._B_residual = np.zeros((self.num_envs, 13, 12), dtype=np.float32)
        self._foot_placement_residuals = np.zeros((self.num_envs,4), dtype=np.float32)
        self._gait_stepping_frequency = self.cfg.nominal_gait_stepping_frequency * np.ones(self.num_envs, dtype=np.float32)
        self._foot_height = self.cfg.nominal_foot_height * np.ones(self.num_envs, dtype=np.float32)
        
        # contact state
        self._gt_grf = torch.zeros(self.num_envs, 6, device=self.device)
        self._gt_contact = torch.zeros(self.num_envs, 2, device=self.device)
        
        # desired commands for MPC
        self._desired_twist = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._desired_gait = np.ones(self.num_envs, dtype=np.int32)
        self._desired_roll_pitch = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._desired_twist_np = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._dsp_duration = np.zeros(self.num_envs, dtype=np.float32)
        self._ssp_duration = np.zeros(self.num_envs, dtype=np.float32)
        
        # setup MPC wrapper
        # mpc_conf = MPC_Conf(
        #     control_dt=self.cfg.dt, control_iteration_between_mpc=self.cfg.iteration_between_mpc, 
        #     horizon_length=self.cfg.horizon_length, mpc_decimation=self.cfg.mpc_decimation)
        # self.mpc = [MPCWrapper(mpc_conf) for _ in range(self.num_envs)] # class array
        self.mpc = MPCWrapper(self.num_envs, self.device)
        
        self.mpc_ctrl_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # for logging
        self.episode_sums = {
            "height_reward": 0.0,
            "lin_vel_reward": 0.0,
            "ang_vel_reward": 0.0,
            "alive_reward": 0.0,
            "contact_reward": 0.0,
            "position_reward": 0.0,
            "yaw_reward": 0.0,
            
            "roll_penalty": 0.0,
            "pitch_penalty": 0.0,
            "action_penalty": 0.0,
            "energy_penalty": 0.0,
            "foot_energy_penalty": 0.0,
            "vx_penalty": 0.0,
            "vy_penalty": 0.0,
            "wz_penalty": 0.0,
            "feet_slide_penalty": 0.0,
            "left_hip_roll_penalty": 0.0,
            "right_hip_roll_penalty": 0.0,
            "hip_pitch_deviation_penalty": 0.0,
            "action_saturation_penalty": 0.0,
        }
        
        # rendering
        self.is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
    
    def _setup_scene(self)->None:
        """
        Environment specific setup.
        Setup the robot, terrain, sensors, and lights in the scene.
        """
        # robot
        self._robot = Articulation(self.cfg.robot)
        self._robot_api = RobotCore(self._robot, self.cfg.foot_patch_num)
        self.scene.articulations["robot"] = self._robot
        
        # sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # base terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.8433914, 0.0, 0.5372996, 0.0))
    
    def _update_mpc_input(self)->None:
        """
        Set the input for MPC controller.
        """
        self._set_mpc_reference()
        twist = torch.cat((self._desired_root_lin_vel_b, self._desired_root_ang_vel_b), dim=1)
        self.mpc.set_command(twist, self._desired_height)
        self.mpc_ctrl_counter += 1
    
    def _set_mpc_reference(self):
        """
        The gait, reference height, roll, pitch, and command velocity are set here.
        Instead of sending the command right away, we warm up the robot with 0 command velocity.
        Then, gradually increase the command velocity for the first 0.5s (ramp up phase).
        After this, the command should be set to gait=2 and desired velocity.
        """
        first_time = self.episode_length_buf == 0
        ramp_up_step = int(0.5/self.physics_dt) # 0.5s
        warm_up_mask = self.mpc_ctrl_counter > self.cfg.gait_change_cutoff
        ramp_up_coef = torch.clip((self.mpc_ctrl_counter - self.cfg.gait_change_cutoff)/ramp_up_step, 0.0, 1.0)
        self._desired_root_lin_vel_b[:, 0] = warm_up_mask * ramp_up_coef * torch.from_numpy(self._desired_twist_np[:, 0]).to(self.device)
        self._desired_root_lin_vel_b[:, 1] = warm_up_mask * ramp_up_coef * torch.from_numpy(self._desired_twist_np[:, 1]).to(self.device)
        self._desired_root_ang_vel_b[:, 0] = warm_up_mask * ramp_up_coef * torch.from_numpy(self._desired_twist_np[:, 2]).to(self.device)
        self._desired_height = self.cfg.reference_height * torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self._desired_gait = torch.where(first_time, 1, 2).cpu().numpy()
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        RL control step running at decimation*dt (200Hz)
        Process RL output.
        """
        self._actions = actions.clone()
        # scale action to -1 to 1
        if self.cfg.clip_action:
            self._actions = torch.tanh(self._actions)
        # rescale action
        if self.cfg.scale_action:
            self._actions_op = self._actions * self.action_ub
        else:
            self._actions_op = self._actions.clone()
        
        self._update_mpc_input()
    
    def _split_action(self, policy_action:torch.Tensor)->torch.Tensor:
        """
        Split policy action into GRW residuals.
        """
        grfm = policy_action[:, :12]
        return grfm
    
    def _apply_action(self)->None:
        """
        low level control loop + motor actuation running at dt (1000Hz)
        **********************
        This is kind of like motor actuation loop.
        It is actually not applying self._action to articulation, 
        but rather setting joint effort target.
        And this effort target is passed to actuator model to get dof torques.
        Finally, env.step method calls write_data_to_sim() to write torque to articulation.
        """
        
        # process rl actions here
        # ....
        self._get_state()
        # run mpc controller
        self.mpc.update_state(self._state)
        self.mpc.run()
        self._joint_actions = self.mpc.get_action()
        # print(self._joint_actions)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
    
    def _get_state(self) -> None:
        """
        Get robot's center of mass state and joint state. 
        NOTE that the center of mass pose is relative to the initial spawn position (i.e. odometry adding nominal height as offset). 
        """
        
        default_state = self._robot_api.default_root_state
        
        rot_mat = matrix_from_quat(self._robot_api.root_quat_w)
        self._init_rot_mat = matrix_from_quat(default_state[:, 3:7])
        self._root_rot_mat = torch.bmm(torch.transpose(self._init_rot_mat, 1, 2), rot_mat) # find relative orientation from initial rotation
        self._root_quat = quat_from_matrix(self._root_rot_mat)
        
        self._root_yaw = torch.atan2(2*(self._root_quat[:, 0]*self._root_quat[:, 3] + self._root_quat[:, 1]*self._root_quat[:, 2]), 1 - 2*(self._root_quat[:, 2]**2 + self._root_quat[:, 3]**2)).view(-1, 1)
        self._root_yaw = torch.atan2(torch.sin(self._root_yaw), torch.cos(self._root_yaw)) # standardize to -pi to pi
        
        # root_pos is absolute position in fixed frame, we need robot's odometry (relative position from initial position)
        self._root_pos = self._robot_api.root_pos_w - self.scene.env_origins
        self._root_pos[:, :2] -= default_state[:, :2]
        self._root_pos = torch.bmm(torch.transpose(self._init_rot_mat, 1, 2), self._root_pos.view(-1, 3, 1)).view(-1, 3) # find relative position from initial position
        # process height same as hardware code
        # https://github.gatech.edu/GeorgiaTechLIDARGroup/HECTOR_HW_new/blob/Unified_Framework/Interface/HW_interface/src/stateestimator/PositionVelocityEstimator.cpp
        toe_index, _ = self._robot.find_bodies(["L_toe", "R_toe"], preserve_order=True)
        foot_pos = (self._robot_api.body_pos_w[:, toe_index, 2]-0.04) - self._robot_api.root_pos_w [:, 2].view(-1, 1) # com to foot in world frame
        phZ, _ = torch.max(-foot_pos, dim=1)
        self._root_pos[:, 2] = phZ
        
        self._root_lin_vel_b = self._robot_api.root_lin_vel_b
        self._root_ang_vel_b = self._robot_api.root_ang_vel_b
        
        self._joint_pos = self._robot_api.joint_pos[:, self._joint_ids]
        self._add_joint_offset()
        self._joint_vel = self._robot_api.joint_vel[:, self._joint_ids]
        self._joint_effort = self._robot_api.joint_effort[:, self._joint_ids]
        
        # reference calculation 
        self._ref_yaw = self._ref_yaw + self._desired_root_ang_vel_b[:, 0:1] * self.cfg.dt
        self._ref_pos[:, 0] = self._ref_pos[:, 0] + self._desired_root_lin_vel_b[:, 0] * torch.cos(self._ref_yaw.squeeze()) * self.cfg.dt - \
            self._desired_root_lin_vel_b[:, 1] * torch.sin(self._ref_yaw.squeeze()) * self.cfg.dt
        self._ref_pos[:, 1] = self._ref_pos[:, 1] + self._desired_root_lin_vel_b[:, 0] * torch.sin(self._ref_yaw.squeeze()) * self.cfg.dt + \
            self._desired_root_lin_vel_b[:, 1] * torch.cos(self._ref_yaw.squeeze()) * self.cfg.dt
        
        self._state = torch.cat(
            (
                self._root_pos,
                self._root_quat,
                self._root_lin_vel_b,
                self._root_ang_vel_b,
                self._joint_pos,
                self._joint_vel,
                self._joint_effort,
            ),
            dim=-1,
        )
    
    def _add_joint_offset(self, env_ids: Sequence[int]|None = None):
        """
        Add joint offset to simulation joint data. 
        This is because of difference in zero-joint position in simulation and hardware. 
        These offsets are synced with hector URDF.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._joint_pos[env_ids, 2] += 0.25*torch.pi 
        self._joint_pos[env_ids, 3] -= 0.5*torch.pi 
        self._joint_pos[env_ids, 4] += 0.25*torch.pi 
        self._joint_pos[env_ids, 7] += 0.25*torch.pi 
        self._joint_pos[env_ids, 8] -= 0.5*torch.pi
        self._joint_pos[env_ids, 9] += 0.25*torch.pi
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
        self._obs = torch.cat(
            (
                self._root_pos[:, 2:], #0:1 z
                self._root_quat, #1:5
                self._root_lin_vel_b, #5:8
                self._root_ang_vel_b, #8:11
                self._desired_root_lin_vel_b, #11:13
                self._desired_root_ang_vel_b, #13:14
                self._joint_pos, #14:24
                self._joint_vel, #24:34
                self._joint_effort, #34:44
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    def _get_contact_observation(self)->None:
        """
        Get ground-truth contact observation and contact metrics for RL observation.
        """
        contact_force = self._contact_sensor.data.net_forces_w
        if contact_force is None:
            self._gt_grf = torch.zeros(self.num_envs, 6, device=self.device)
            self._gt_contact = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            self._gt_grf = contact_force.view(-1, 6).to(self.device)
            self._gt_contact = (torch.norm(self._contact_sensor.data.net_forces_w, dim=-1) > 1.0).to(torch.float32).view(-1, 2)
    
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        # implement environment reset logic here
        super()._reset_idx(env_ids)
    
    def _reset_robot(self, env_ids: Sequence[int])->None:
        # implement robot reset logic here
        raise NotImplementedError
    
    def _get_rewards(self)->torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self)->tuple[torch.Tensor, torch.Tensor]:
        # timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        roll = torch.atan2(2*(self._root_quat[:, 0]*self._root_quat[:, 1] + self._root_quat[:, 2]*self._root_quat[:, 3]), 1 - 2*(self._root_quat[:, 1]**2 + self._root_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        
        pitch = torch.asin(2*(self._root_quat[:, 0]*self._root_quat[:, 2] - self._root_quat[:, 3]*self._root_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        
        # base angle and base height violation
        roll_reset = torch.abs(roll) > torch.pi/6
        pitch_reset = torch.abs(pitch) > torch.pi/6
        height_reset = (self._root_pos[:, 2] < 0.55-0.2) | (self._root_pos[:, 2] > 0.55+0.2)
        
        reset = roll_reset | pitch_reset
        reset = reset | height_reset
        
        return reset, time_out
    
    def log_episode_reward(self)->None:
        raise NotImplementedError
    
    ### specific to the architecture ###
    def log_state(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            # add state to log
            pass
        self.extras["log"].update(log)
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            # add action to log
            pass
        self.extras["log"].update(log)