# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import gymnasium as gym
import numpy as np
import torch
# torch.set_printoptions(threshold=10, edgeitems=2, precision=3, linewidth=80, profile="full")
from collections.abc import Sequence
from dataclasses import MISSING
import carb

# IsaacSim core
import omni.kit.app
import omni.log

# Isaaclab core
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.utils.spaces import sample_space, spec_to_gym_space
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

# Task core
from isaaclab_tasks.direct.hector.common.robot_core import RobotCore
from isaaclab_tasks.direct.hector.common.mpc_wrapper import MPC_Conf, MPCWrapper

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"

# Task cfg
# from isaaclab_tasks.direct.hector.task_cfg.base_arch_cfg import BaseArchCfg
from isaaclab_tasks.direct.hector.task_cfg.base_arch_e2e_cfg import BaseArchE2ECfg



class BaseArchE2E(DirectRLEnv):
    cfg: BaseArchE2ECfg
    
    def __init__(self, cfg: BaseArchE2ECfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # buffer
        self.max_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # get joint_ids that maps sim joint order to controller joint order
        self._joint_ids, self._joint_names = self._robot.find_joints(self.cfg.joint_names, preserve_order=True)
        
        # actions
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._actions_op = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._joint_actions = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device)
        
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
        self._joint_pos = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_vel = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_effort = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_effort_target = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_accel = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._state = torch.zeros(self.num_envs, self.cfg.num_states, device=self.device, dtype=torch.float32)
        self._foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        
        # RL observation
        self._obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        
        # height scan 
        self.height_map = None
        self.height_map_2d_grad = None
        
        # contact state
        self._gt_grf = torch.zeros(self.num_envs, 6, device=self.device)
        self._gt_contact = torch.zeros(self.num_envs, 2, device=self.device)
        self.roughness_at_fps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # desired commands
        self._desired_root_lin_vel_b = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._desired_root_ang_vel_b = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._desired_height = self.cfg.reference_height * torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        
        # set up robot core 
        self.foot_body_id = torch.tensor(self.cfg.foot_body_id, device=self.device, dtype=torch.long)
        self._robot_api = RobotCore(self._robot, self.num_envs, self.foot_body_id)
        
        # for logging
        self.episode_reward_sums = {
            "height_reward": torch.zeros(self.num_envs, device=self.device),
            "lin_vel_reward": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_reward": torch.zeros(self.num_envs, device=self.device),
            "alive_reward": torch.zeros(self.num_envs, device=self.device),
            "contact_reward": torch.zeros(self.num_envs, device=self.device),
            "position_reward": torch.zeros(self.num_envs, device=self.device),
            "yaw_reward": torch.zeros(self.num_envs, device=self.device),
            "swing_foot_tracking_reward": torch.zeros(self.num_envs, device=self.device),
        }
        
        self.episode_penalty_sums = {
            "roll_penalty": torch.zeros(self.num_envs, device=self.device), # roll penalty for the root orientation
            "pitch_penalty": torch.zeros(self.num_envs, device=self.device), # pitch penalty for the root orientation
            "velocity_penalty": torch.zeros(self.num_envs, device=self.device), # this is a velocity penalty for the root linear velocity
            "ang_velocity_penalty": torch.zeros(self.num_envs, device=self.device), # angular velocity penalty for the root angular velocity
            "feet_slide_penalty": torch.zeros(self.num_envs, device=self.device),
            "foot_distance_penalty": torch.zeros(self.num_envs, device=self.device),
            "toe_left_joint_penalty": torch.zeros(self.num_envs, device=self.device),
            "toe_right_joint_penalty": torch.zeros(self.num_envs, device=self.device),
            "contact_location_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_saturation_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, device=self.device),
            "energy_penalty": torch.zeros(self.num_envs, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, device=self.device),
        }
    
    def _setup_scene(self)->None:
        """
        Environment specific setup.
        Setup the robot, terrain, sensors, and lights in the scene.
        """
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        # sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # ray caster
        self._raycaster = RayCaster(self.cfg.ray_caster)
        self.scene.sensors["raycaster"] = self._raycaster
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.819152, 0.0, 0.5735764, 0.0))
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        RL control step
        Process RL output.
        """
        self._actions = actions.clone()
        
        # clip action to (-1, 1) (NOTE: rsl_rl does no thave activation at the last layer)
        if self.cfg.clip_action:
            self._actions = torch.tanh(self._actions)
        # rescale action to custom bounds
        if self.cfg.scale_action:
            joint_lb = self._robot_api.joint_pos_limit[:, :, 0]
            # joint_lb = self._add_joint_offset(joint_lb)
            joint_ub = self._robot_api.joint_pos_limit[:, :, 1]
            # joint_ub = self._add_joint_offset(joint_ub)
            positive_mask = (self._actions > 0).to(torch.float32)
            self._actions_op = positive_mask * joint_ub * self._actions + (1-positive_mask) * joint_lb * (-self._actions)
        else:
            self._actions_op = self._actions.clone()
    
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
        self._get_state()
        # process rl actions here
        self._joint_actions = self._actions_op.clone()
        # self._joint_actions = self._subtract_joint_offset(self._actions_op) # from hardware joint space to sim jont space
        self._robot_api.set_joint_position_target(self._joint_actions, self._joint_ids)
    
    def _get_state(self) -> None:
        """
        Get robot's center of mass state and joint state. 
        NOTE that the center of mass pose is relative to the initial spawn position (i.e. odometry adding nominal height as offset). 
        """
        
        self._init_rot_mat = self._robot_api._init_rot
        self._root_rot_mat = self._robot_api.root_rot_mat_local
        self._root_quat = self._robot_api.root_quat_local
        
        # TODO: consider including it in robot_api
        self._root_yaw = torch.atan2(2*(self._root_quat[:, 0]*self._root_quat[:, 3] + self._root_quat[:, 1]*self._root_quat[:, 2]), 1 - 2*(self._root_quat[:, 2]**2 + self._root_quat[:, 3]**2)).view(-1, 1)
        self._root_yaw = torch.atan2(torch.sin(self._root_yaw), torch.cos(self._root_yaw)) # standardize to -pi to pi
        
        # process z position same as hardware code
        self._root_pos = self._robot_api.root_pos_local
        # https://github.gatech.edu/GeorgiaTechLIDARGroup/HECTOR_HW_new/blob/Unified_Framework/Interface/HW_interface/src/stateestimator/PositionVelocityEstimator.cpp
        # Is this right for non-flat terrain? probably not...
        # toe_index, _ = self._robot.find_bodies(["L_toe", "R_toe"], preserve_order=True)
        # foot_pos = (self._robot_api.body_pos_w[:, toe_index, 2]-0.04) - self._robot_api.root_pos_w [:, 2].view(-1, 1) # com to foot in world frame
        # phZ, _ = torch.max(-foot_pos, dim=1)
        # self._root_pos[:, 2] = phZ
        
        self._root_lin_vel_b = self._robot_api.root_lin_vel_b
        self._root_ang_vel_b = self._robot_api.root_ang_vel_b
        
        # joint state
        self._joint_pos = self._robot_api.joint_pos[:, self._joint_ids]
        # self._joint_pos = self._add_joint_offset(self._joint_pos)
        self._joint_vel = self._robot_api.joint_vel[:, self._joint_ids]
        self._joint_effort = self._robot_api.joint_effort[:, self._joint_ids]
        
        # foot state
        self._foot_pos_b = self._robot_api.foot_pos_b.view(self.num_envs, -1)
        
        # reference calculation 
        self._ref_yaw = self._ref_yaw + self._desired_root_ang_vel_b[:, 0:1] * self.cfg.dt
        self._ref_yaw = torch.atan2(torch.sin(self._ref_yaw), torch.cos(self._ref_yaw)) # standardize to -pi to pi
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
            ),
            dim=-1,
        )
        
    def _add_joint_offset(self, joint_pos:torch.Tensor):
        """
        Add joint offset to simulation joint data. 
        This is because of difference in zero-joint position in simulation and hardware. 
        We align the definition to hardware. 
        """
        _joint_pos = joint_pos.clone()
        _joint_pos[:, 2] += 0.25*torch.pi 
        _joint_pos[:, 3] -= 0.5*torch.pi 
        _joint_pos[:, 4] += 0.25*torch.pi 
        _joint_pos[:, 7] += 0.25*torch.pi 
        _joint_pos[:, 8] -= 0.5*torch.pi
        _joint_pos[:, 9] += 0.25*torch.pi
        return _joint_pos
    
    def _subtract_joint_offset(self, joint_pos:torch.Tensor):
        """
        Add joint offset to simulation joint data. 
        This is because of difference in zero-joint position in simulation and hardware. 
        We align the definition to hardware. 
        """
        _joint_pos = joint_pos.clone()
        _joint_pos[:, 2] -= 0.25*torch.pi 
        _joint_pos[:, 3] += 0.5*torch.pi 
        _joint_pos[:, 4] -= 0.25*torch.pi 
        _joint_pos[:, 7] -= 0.25*torch.pi 
        _joint_pos[:, 8] += 0.5*torch.pi
        _joint_pos[:, 9] -= 0.25*torch.pi
        
        return _joint_pos
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        raise NotImplementedError
    
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
    
    def _get_exteroceptive_observation(self)->None:
        self.height_map = (self._raycaster.data.ray_hits_w[..., 2] - self._raycaster.data.pos_w[:, 2].unsqueeze(1)).clip(-2.0, 2.0) # local height wrt to base
        height_map_2d = self.height_map.view(self.num_envs, int(1.0/0.05+1), int(1.0/0.05+1))
        grad_x = torch.gradient(height_map_2d, dim=1)[0] # gradient returns tuple
        grad_y = torch.gradient(height_map_2d, dim=2)[0] # gradient returns tuple
        self.height_map_2d_grad = torch.abs(grad_x) + torch.abs(grad_y)
    
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        # implement environment reset logic here
        super()._reset_idx(env_ids)
    
    def _reset_robot(self, env_ids: Sequence[int])->None:
        # implement robot reset logic here
        raise NotImplementedError
    
    def _get_rewards(self)->torch.Tensor:
        raise NotImplementedError
    
    def _get_dones(self)->tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def log_episode_return(self)->None:
        raise NotImplementedError
    
    def log_state(self)->None:
        raise NotImplementedError
    
    def log_action(self)->None:
        raise NotImplementedError
    
    def log_reward(self)->None:
        raise NotImplementedError
    
    def _configure_gym_env_spaces(self):
        """
        Configure the action and observation spaces for the Gym environment.
        Modified from the parent class to handle history and exteroceptive observations.
        """
        # show deprecation message and overwrite configuration
        if self.cfg.num_actions is not None:
            omni.log.warn("DirectRLEnvCfg.num_actions is deprecated. Use DirectRLEnvCfg.action_space instead.")
            if isinstance(self.cfg.action_space, type(MISSING)):
                self.cfg.action_space = self.cfg.num_actions
        if self.cfg.num_observations is not None:
            omni.log.warn(
                "DirectRLEnvCfg.num_observations is deprecated. Use DirectRLEnvCfg.observation_space instead."
            )
            if isinstance(self.cfg.observation_space, type(MISSING)):
                self.cfg.observation_space = self.cfg.num_observations
        if self.cfg.num_states is not None:
            omni.log.warn("DirectRLEnvCfg.num_states is deprecated. Use DirectRLEnvCfg.state_space instead.")
            if isinstance(self.cfg.state_space, type(MISSING)):
                self.cfg.state_space = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space*self.cfg.num_history + self.cfg.num_extero_observations)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        self.state_space = None
        if self.cfg.state_space:
            self.single_observation_space["critic"] = spec_to_gym_space(self.cfg.state_space*self.cfg.num_history + self.cfg.num_extero_observations)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)