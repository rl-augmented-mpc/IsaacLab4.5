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
        self._joint_pos = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_vel = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_effort = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._joint_accel = torch.zeros(self.num_envs, self.cfg.num_joint_actions, device=self.device, dtype=torch.float32)
        self._state = torch.zeros(self.num_envs, self.cfg.num_states, device=self.device, dtype=torch.float32)
        
        # RL observation
        self._obs = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_observation_space["policy"]), device=self.device)
        
        # height scan 
        self.height_map = None
        
        # contact state
        self._gt_grf = torch.zeros(self.num_envs, 6, device=self.device)
        self._gt_contact = torch.zeros(self.num_envs, 2, device=self.device)
        
        # MPC output
        self._grfm_mpc = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float32)
        self._accel_gyro_mpc = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._gait_contact = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._swing_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._reibert_fps = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._reibert_fps_b = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32) # in body frame
        self._augmented_fps = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._augmented_fps_b = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32) # in body frame
        self._foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._ref_foot_pos_b = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        
        # Buffer for residual learning
        self._joint_action_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        self._joint_position_residuals = np.zeros((self.num_envs, 10), dtype=np.float32)
        self._grfm_residuals = np.zeros((self.num_envs, 12), dtype=np.float32)
        self._accel_peturbation = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._ang_accel_peturbation = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._A_residual = np.zeros((self.num_envs, 13, 13), dtype=np.float32)
        self._B_residual = np.zeros((self.num_envs, 13, 12), dtype=np.float32)
        self._foot_placement_residuals = np.zeros((self.num_envs,4), dtype=np.float32)
        self.residual_forward_vel = np.zeros(self.num_envs, dtype=np.float32)
        self._gait_stepping_frequency = self.cfg.nominal_gait_stepping_frequency * np.ones(self.num_envs, dtype=np.float32)
        self.nominal_foot_height = self.cfg.nominal_foot_height * np.ones(self.num_envs, dtype=np.float32)
        self._foot_height = self.cfg.nominal_foot_height * np.ones(self.num_envs, dtype=np.float32)
        
        # desired commands
        self._desired_root_lin_vel_b = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._desired_root_ang_vel_b = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._desired_twist = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._desired_height = self.cfg.reference_height * np.ones(self.num_envs, dtype=np.float32)
        self._desired_gait = 2 * np.ones(self.num_envs, dtype=np.int32)
        self._desired_roll_pitch = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._desired_twist_np = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._dsp_duration = np.zeros(self.num_envs, dtype=np.float32)
        self._ssp_duration = np.zeros(self.num_envs, dtype=np.float32)
        
        # setup MPC wrapper
        mpc_conf = MPC_Conf(
            control_dt=self.cfg.dt, control_iteration_between_mpc=self.cfg.iteration_between_mpc, 
            horizon_length=self.cfg.horizon_length, mpc_decimation=self.cfg.mpc_decimation)
        self.mpc = [MPCWrapper(mpc_conf) for _ in range(self.num_envs)] # class array
        for i in range(self.num_envs):
            self.mpc[i].set_planner("Raibert")
            # self.mpc[i].set_planner("LIP")
        self.mpc_ctrl_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
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
            "velocity_penalty": torch.zeros(self.num_envs, device=self.device), # this is a velocity penalty for the root linear velocity
            "ang_velocity_penalty": torch.zeros(self.num_envs, device=self.device), # angular velocity penalty for the root angular velocity
            "feet_slide_penalty": torch.zeros(self.num_envs, device=self.device),
            "hip_pitch_deviation_penalty": torch.zeros(self.num_envs, device=self.device),
            "foot_distance_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_saturation_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, device=self.device),
            "energy_penalty": torch.zeros(self.num_envs, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, device=self.device),
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
        
        # ray caster
        self._raycaster = RayCaster(self.cfg.ray_caster)
        self.scene.sensors["raycaster"] = self._raycaster
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.819152, 0.0, 0.5735764, 0.0))
    
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
         
        # run mpc controller
        self._run_mpc()
        
        # add residual here
        for i in range(len(self.mpc)):
            self._joint_action_augmented[i] = self.mpc[i].get_action()
            
        self._joint_actions = torch.from_numpy(self._joint_action_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
    
    def _run_mpc(self)->None:
        """
        Run MPC and update GRFM and contact state.
        MPC runs at every dt*mpc_decimation (200Hz)
        """
        accel_gyro = []
        grfm = []
        gait_contact = []
        swing_phase = []
        reibert_fps = []
        augmented_fps = []
        foot_pos_b = []
        foot_ref_pos_b = []
        
        self._get_state()
        for i in range(len(self.mpc)):
            self.mpc[i].set_swing_parameters(stepping_frequency=self._gait_stepping_frequency[i], foot_height=self._foot_height[i])
            self.mpc[i].set_terrain_slope(self.cfg.terrain_slope)
            self.mpc[i].add_foot_placement_residual(self._foot_placement_residuals[i])
            self.mpc[i].set_srbd_residual(A_residual=self._A_residual[i], B_residual=self._B_residual[i])
            self.mpc[i].update_state(self._state[i].cpu().numpy())
            self.mpc[i].run()
            
            accel_gyro.append(self.mpc[i].accel_gyro(self._root_rot_mat[i].cpu().numpy()))
            grfm.append(self.mpc[i].grfm)
            gait_contact.append(self.mpc[i].contact_state)
            swing_phase.append(self.mpc[i].swing_phase)
            reibert_fps.append(self.mpc[i].reibert_foot_placement)
            augmented_fps.append(self.mpc[i].foot_placement)
            foot_pos_b.append(self.mpc[i].foot_pos_b)
            foot_ref_pos_b.append(self.mpc[i].ref_foot_pos_b)
        
        self._accel_gyro_mpc = torch.from_numpy(np.array(accel_gyro)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self._grfm_mpc = torch.from_numpy(np.array(grfm)).to(self.device).view(self.num_envs, 12).to(torch.float32)
        self._gait_contact = torch.from_numpy(np.array(gait_contact)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._swing_phase = torch.from_numpy(np.array(swing_phase)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._reibert_fps = torch.from_numpy(np.array(reibert_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)

        # transform rb fps to body frame
        self._reibert_fps_b[:, 0] = self._reibert_fps[:, 0]*torch.cos(self._root_yaw.squeeze()) + self._reibert_fps[:, 1]*torch.sin(self._root_yaw.squeeze()) - self._root_pos[:, 0]
        self._reibert_fps_b[:, 1] = -self._reibert_fps[:, 0]*torch.sin(self._root_yaw.squeeze()) + self._reibert_fps[:, 1]*torch.cos(self._root_yaw.squeeze()) - self._root_pos[:, 1]
        self._reibert_fps_b[:, 2] = self._reibert_fps[:, 2]*torch.cos(self._root_yaw.squeeze()) + self._reibert_fps[:, 3]*torch.sin(self._root_yaw.squeeze()) - self._root_pos[:, 0]
        self._reibert_fps_b[:, 3] = -self._reibert_fps[:, 2]*torch.sin(self._root_yaw.squeeze()) + self._reibert_fps[:, 3]*torch.cos(self._root_yaw.squeeze()) - self._root_pos[:, 1]

        self._augmented_fps = torch.from_numpy(np.array(augmented_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)

        # transform augmented fps to body frame
        self._augmented_fps_b[:, 0] = self._augmented_fps[:, 0]*torch.cos(self._root_yaw.squeeze()) + self._augmented_fps[:, 1]*torch.sin(self._root_yaw.squeeze()) - self._root_pos[:, 0]
        self._augmented_fps_b[:, 1] = -self._augmented_fps[:, 0]*torch.sin(self._root_yaw.squeeze()) + self._augmented_fps[:, 1]*torch.cos(self._root_yaw.squeeze()) - self._root_pos[:, 1]
        self._augmented_fps_b[:, 2] = self._augmented_fps[:, 2]*torch.cos(self._root_yaw.squeeze()) + self._augmented_fps[:, 3]*torch.sin(self._root_yaw.squeeze()) - self._root_pos[:, 0]
        self._augmented_fps_b[:, 3] = -self._augmented_fps[:, 2]*torch.sin(self._root_yaw.squeeze()) + self._augmented_fps[:, 3]*torch.cos(self._root_yaw.squeeze()) - self._root_pos[:, 1]

        self._foot_pos_b = torch.from_numpy(np.array(foot_pos_b)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self._ref_foot_pos_b = torch.from_numpy(np.array(foot_ref_pos_b)).to(self.device).view(self.num_envs, 6).to(torch.float32)
    
    def _update_mpc_input(self)->None:
        """
        Set the input for MPC controller.
        """
        self._set_mpc_reference()
        for i in range(len(self.mpc)):
            lin_velocity = self._desired_root_lin_vel_b[i].cpu().numpy()
            ang_velocity = self._desired_root_ang_vel_b[i].cpu().numpy()
            desired_twist = np.array([lin_velocity[0], lin_velocity[1], ang_velocity[0]], dtype=np.float32)
            self.mpc[i].set_command(gait_num=self._desired_gait[i], 
                                roll_pitch=self._desired_roll_pitch[i], 
                                twist=desired_twist, 
                                height=self._desired_height[i])
        self.mpc_ctrl_counter += 1

    def _set_mpc_reference(self):
        """
        The gait, reference height, roll, pitch, and command velocity are set here.
        Instead of sending the command right away, we warm up the robot with 0 command velocity.
        Then, gradually increase the command velocity for the first 0.5s (ramp up phase).
        After this, the command should be set to gait=2 and desired velocity.
        """
        ramp_up_step = int(0.1/self.physics_dt)
        ramp_up_coef = torch.clip(self.mpc_ctrl_counter/ramp_up_step, 0.0, 1.0)
        # ramp_up_coef = 1.0
        self._desired_root_lin_vel_b[:, 0] = ramp_up_coef * torch.from_numpy(self._desired_twist_np[:, 0]).to(self.device)
        self._desired_root_lin_vel_b[:, 1] = ramp_up_coef * torch.from_numpy(self._desired_twist_np[:, 1]).to(self.device)
        self._desired_root_ang_vel_b[:, 0] = ramp_up_coef * torch.from_numpy(self._desired_twist_np[:, 2]).to(self.device)
        self._desired_gait[:] = 2
    
    def _get_state(self) -> None:
        """
        Get robot's center of mass state and joint state. 
        NOTE that the center of mass pose is relative to the initial spawn position (i.e. odometry adding nominal height as offset). 
        """
        
        self._init_rot_mat = self._robot_api._init_rot # type: ignore
        self._root_rot_mat = self._robot_api.root_rot_mat_local
        self._root_quat = self._robot_api.root_quat_local
        
        # TODO: consider including it in robot_api
        self._root_yaw = torch.atan2(2*(self._root_quat[:, 0]*self._root_quat[:, 3] + self._root_quat[:, 1]*self._root_quat[:, 2]), 1 - 2*(self._root_quat[:, 2]**2 + self._root_quat[:, 3]**2)).view(-1, 1)
        self._root_yaw = torch.atan2(torch.sin(self._root_yaw), torch.cos(self._root_yaw)) # standardize to -pi to pi
        
        self._root_pos = self._robot_api.root_pos_local
        # process z position same as hardware code
        # https://github.gatech.edu/GeorgiaTechLIDARGroup/HECTOR_HW_new/blob/Unified_Framework/Interface/HW_interface/src/stateestimator/PositionVelocityEstimator.cpp
        # toe_index, _ = self._robot.find_bodies(["L_toe", "R_toe"], preserve_order=True)
        # foot_pos = (self._robot_api.body_pos_w[:, toe_index, 2]-0.04) - self._robot_api.root_pos_w [:, 2].view(-1, 1) # com to foot in world frame
        # phZ, _ = torch.max(-foot_pos, dim=1)
        # self._root_pos[:, 2] = phZ
        
        self._root_lin_vel_b = self._robot_api.root_lin_vel_b
        self._root_ang_vel_b = self._robot_api.root_ang_vel_b
        
        self._joint_pos = self._robot_api.joint_pos[:, self._joint_ids]
        self._add_joint_offset()
        self._joint_vel = self._robot_api.joint_vel[:, self._joint_ids]
        self._joint_effort = self._robot_api.joint_effort[:, self._joint_ids]
        
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
    
    def _add_joint_offset(self, env_ids: torch.Tensor|Sequence[int]|None=None):
        """
        Add joint offset to simulation joint data. 
        This is because of difference in zero-joint position in simulation and hardware. 
        We align the definition to hardware. 
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
        # self.height_map = torch.clamp(self._raycaster.data.ray_hits_w[..., 2], min=-1, max=1) # global height
        self.height_map = (self._raycaster.data.ray_hits_w[..., 2] - self._raycaster.data.pos_w[:, 2].unsqueeze(1)).clip(-2.0, 2.0) # local height wrt to base
    
    
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