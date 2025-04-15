# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections.abc import Sequence

# IsaacLab core
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"

##
# Pre-defined configs
##

# Task core
from  isaaclab_tasks.direct.hector.common.utils.data_util import HistoryBuffer

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.stepping_stone_cfg import SteppingStoneCfg

# Base class
from isaaclab_tasks.direct.hector.task.hierarchical_arch import HierarchicalArch


class SteppingStone(HierarchicalArch):
    cfg: SteppingStoneCfg
    def __init__(self, cfg: SteppingStoneCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_history = self.cfg.num_history
        self.history_buffer = HistoryBuffer(self.num_envs, self.num_history, self.cfg.observation_space, torch.float32, self.device)
        
    def _setup_scene(self)->None:
        """
        Environment specific setup.
        Setup the robot, terrain, sensors, and lights in the scene.
        """
        super()._setup_scene()
        # update viewer
        terrain_origin = np.array([self.cfg.terrain.center_position[0], self.cfg.terrain.center_position[1], 0.0])
        camera_pos = terrain_origin
        camera_delta = np.array([0.0, -7.0, 0.5])
        self.cfg.viewer.eye = (camera_pos[0]+camera_delta[0], camera_pos[1]+camera_delta[1], camera_pos[2]+camera_delta[2])
        self.cfg.viewer.lookat = (camera_pos[0], camera_pos[1], 0.0)
    
    
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into useful form
        """
        stepping_frequency_traj = policy_action[:, :self.cfg.traj_sample] -0.1
        foot_height_traj = policy_action[:, self.cfg.traj_sample:self.cfg.traj_sample*2]
        cp_traj = policy_action[:, self.cfg.traj_sample*2:self.cfg.traj_sample*3]
        return stepping_frequency_traj, foot_height_traj, cp_traj
    
    def _apply_action(self)->None:
        """
        Actuation control loop
        **********************
        This is kind of like motor actuation loop.
        It is actually not applying self._action to articulation,
        but rather setting joint effort target.
        And this effort target is passed to actuator model to get dof torques.
        Finally, env.step method calls write_data_to_sim() to write torque to articulation.
        """
        # process rl actions
        stepping_frequency_traj, foot_height_traj, cp_traj = self._split_action(self._actions_op)
        
        # pick element from trajectory
        time_idx = (self.cfg.traj_sample*((self.mpc_ctrl_counter * self.cfg.dt)/self.cfg.policy_dt - ((self.mpc_ctrl_counter * self.cfg.dt)/self.cfg.policy_dt).int())).long()
        env_idx = torch.arange(self.num_envs, device=self.device)
        stepping_frequency = stepping_frequency_traj[env_idx, time_idx]
        foot_clearance = foot_height_traj[env_idx, time_idx]
        
        self._gait_stepping_frequency = self.cfg.nominal_gait_stepping_frequency - stepping_frequency.cpu().numpy()
        self._foot_height = self.nominal_foot_height + foot_clearance.cpu().numpy()

        self._cp1 = self.cfg.cp1_default + cp_traj[env_idx, time_idx].cpu().numpy()
        self._cp2 = self.cfg.cp2_default + cp_traj[env_idx, time_idx].cpu().numpy()

        self._desired_height = self.cfg.reference_height + (self._raycaster.data.pos_w[:, 2] + self.height_map[:, self.height_map.shape[1]//2]).cpu().numpy() # type: ignore
        
        # get proprioceptive
        self._get_state()
        
        # run mpc controller
        self._run_mpc()
        
        # get joint torque from controller
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids) # type: ignore
        
        self.visualize_marker()
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
        self._get_exteroceptive_observation()
        
        self._obs = torch.cat(
            (
                self._root_pos[:, 2:], #0:1 (only height)
                self._root_quat, #1:5
                self._root_lin_vel_b, #5:8
                self._root_ang_vel_b, #8:11
                self._desired_root_lin_vel_b, #11:13
                self._desired_root_ang_vel_b, #13:14
                self._joint_pos, #14:24
                self._joint_vel, #24:34
                self._joint_effort, #34:44
                self._accel_gyro_mpc, #44:50
                self._gait_contact, #50:52
                self._swing_phase, #52:54
                self._reibert_fps_b, # 54:58
                self._foot_pos_b, # 58:64
                self._ref_foot_pos_b, # 64:70
                self._previous_actions, #70:90
            ),
            dim=-1,
        )
        
        buffer_mask = self.history_buffer.size >= self.num_history
        if buffer_mask.any():
            reset_id = torch.nonzero(buffer_mask, as_tuple=True)[0]
            self.history_buffer.pop(reset_id)
        self.history_buffer.push(self._obs)
        
        obs = torch.cat([self.history_buffer.data_flat, self.height_map], dim=-1) # type: ignore
        observation = {"policy": obs, "critic": obs}
        return observation
    
    def _reset_robot(self, env_ids: Sequence[int])->None:
        ### set position ###
        curriculum_idx = np.floor(self.cfg.terrain_curriculum_sampler.sample(self.max_episode_length_buf.float().mean(dim=0).item(), len(env_ids)))
        self.curriculum_idx = curriculum_idx
        center_coord = self._get_sub_terrain_center(curriculum_idx)
        position = self.cfg.robot_position_sampler.sample(center_coord, len(env_ids))
        quat = self.cfg.robot_quat_sampler.sample(np.array(position)[:, :2] - center_coord, len(position))
        
        position = torch.tensor(position, device=self.device).view(-1, 3)
        quat = torch.tensor(quat, device=self.device).view(-1, 4)
        default_root_pose = torch.cat((position, quat), dim=-1)
        
        # override the default state
        self._robot_api.reset_default_pose(default_root_pose, env_ids) # type: ignore
        
        ### set joint state ###
        joint_pos = self._robot_api.default_joint_pos[:, self._joint_ids][env_ids]
        joint_vel = self._robot_api.default_joint_vel[:, self._joint_ids][env_ids]
        self._joint_pos[env_ids] = joint_pos
        self._joint_vel[env_ids] = joint_vel
        self._add_joint_offset(env_ids)
        
        ### write reset to sim ###
        self._robot_api.write_root_pose_to_sim(default_root_pose, env_ids) # type: ignore
        self._robot_api.write_root_velocity_to_sim(self._robot_api.default_root_state[env_ids, 7:], env_ids) # type: ignore
        self._robot_api.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids) # type: ignore
        
        ### reset mpc reference and gait ###
        if not self.cfg.curriculum_inference:
            twist_cmd = np.array(self.cfg.robot_target_velocity_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32) # type: ignore
            self._desired_twist_np[env_ids.cpu().numpy()] = twist_cmd # type: ignore
        
        # reset foot clearance
        self.nominal_foot_height[env_ids.cpu().numpy()] = self.cfg.robot_nominal_foot_height_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)) # type: ignore
        
        # reset gait
        self._dsp_duration[env_ids.cpu().numpy()] = np.array(self.cfg.robot_double_support_length_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32) # type: ignore
        self._ssp_duration[env_ids.cpu().numpy()] = np.array(self.cfg.robot_single_support_length_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32) # type: ignore
        self.mpc_ctrl_counter[env_ids] = 0
        for i in env_ids.cpu().numpy(): # type: ignore
            self.mpc[i].reset()
            self.mpc[i].update_gait_parameter(np.array([self._dsp_duration[i], self._dsp_duration[i]]), np.array([self._ssp_duration[i], self._ssp_duration[i]]))
        
        # reset reference trajectory
        self._ref_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._ref_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        
        if not self.cfg.inference:
            # update view port to look at the current active terrain
            camera_delta = [-1.0, -4.0, 1.0]
            self.viewport_camera_controller.update_view_location(
                eye=(center_coord[0, 0]+camera_delta[0], center_coord[0, 1]+camera_delta[1], camera_delta[2]), 
                lookat=(center_coord[0, 0], center_coord[0, 1], 0.0)) # type: ignore
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env == 0:
            stepping_frequency = self._actions_op[:, 0:self.cfg.traj_sample].mean(dim=0).cpu().numpy() + self.cfg.nominal_gait_stepping_frequency
            foot_height = self._actions_op[:, self.cfg.traj_sample:self.cfg.traj_sample*2].mean(dim=0).cpu().numpy() + self.nominal_foot_height.mean(0)
            cp = self._actions_op[:, self.cfg.traj_sample*2:self.cfg.traj_sample*3].mean(dim=0).cpu().numpy()
            cp1 = self.cfg.cp1_default + cp
            cp2 = self.cfg.cp2_default + cp

            for i in range(self.cfg.traj_sample):
                log[f"action/stepping_frequency_t{i+1}"] = stepping_frequency[i:i+1]
                log[f"action/foot_height_t{i+1}"] = foot_height[i:i+1]
                log[f"action/cp1_t{i+1}"] = cp1[i:i+1]
                log[f"action/cp2_t{i+1}"] = cp2[i:i+1]
            
        self.extras["log"].update(log)