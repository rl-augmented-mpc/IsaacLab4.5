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
        stepping_frequency_traj = policy_action[:, :10]
        foot_height_traj = policy_action[:, 10:20]
        return stepping_frequency_traj, foot_height_traj
    
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
        stepping_frequency_traj, foot_height_traj = self._split_action(self._actions_op)
        
        # pick element from trajectory
        time_idx = (self.cfg.traj_sample*((self.mpc_ctrl_counter * self.cfg.dt)/self.cfg.policy_dt - ((self.mpc_ctrl_counter * self.cfg.dt)/self.cfg.policy_dt).int())).long()
        env_idx = torch.arange(self.num_envs, device=self.device)
        stepping_frequency = stepping_frequency_traj[env_idx, time_idx]
        foot_clearance = foot_height_traj[env_idx, time_idx]
        
        # self._gait_stepping_frequency = self.cfg.nominal_gait_stepping_frequency + stepping_frequency.cpu().numpy()
        # self._foot_height = self.nominal_foot_height + foot_clearance.cpu().numpy()
        # self._desired_height = self.cfg.reference_height + (self.cfg.reference_height + self.height_map[:, self.height_map.shape[1]//2]).cpu().numpy() # type: ignore
        
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
    
    def _reset_robot(self, env_ids):
        super()._reset_robot(env_ids)
        center_coord = self._get_sub_terrain_center(self.curriculum_idx)
        if not self.cfg.inference:
            # update view port to look at the current active terrain
            camera_delta = [-2.0, -4.0, 1.0]
            self.viewport_camera_controller.update_view_location(
                eye=(center_coord[0, 0]+camera_delta[0], center_coord[0, 1]+camera_delta[1], camera_delta[2]), 
                lookat=(center_coord[0, 0], center_coord[0, 1], 0.0)) # type: ignore
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            stepping_frequency = self._actions_op[:, 0:10].mean(dim=0).cpu().numpy()
            foot_height = self._actions_op[:, 10:20].mean(dim=0).cpu().numpy()

            log["action/stepping_frequency_t1"] = stepping_frequency[0:1]
            log["action/stepping_frequency_t2"] = stepping_frequency[1:2]
            log["action/stepping_frequency_t3"] = stepping_frequency[2:3]
            log["action/stepping_frequency_t4"] = stepping_frequency[3:4]
            log["action/stepping_frequency_t5"] = stepping_frequency[4:5]
            
            log["action/foot_height_t1"] = foot_height[0:1]
            log["action/foot_height_t2"] = foot_height[1:2]
            log["action/foot_height_t3"] = foot_height[2:3]
            log["action/foot_height_t4"] = foot_height[3:4]
            log["action/foot_height_t5"] = foot_height[4:5]
            
        self.extras["log"].update(log)