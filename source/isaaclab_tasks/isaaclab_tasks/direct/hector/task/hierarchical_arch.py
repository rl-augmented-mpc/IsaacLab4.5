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
from isaaclab_tasks.direct.hector.common.robot_core import RobotCore
from isaaclab_tasks.direct.hector.common.visualization_marker import FootPlacementVisualizer

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg, HierarchicalArchPrimeCfg, HierarchicalArchAccelPFCfg

# Base class
from isaaclab_tasks.direct.hector.task.base_arch import BaseArch
    

class HierarchicalArch(BaseArch):
    cfg: HierarchicalArchCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.curriculum_idx = np.zeros(self.num_envs)
    
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
        
        # visualization marker
        self.foot_placement_visualizer = FootPlacementVisualizer("/Visuals/foot_placement")
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.8433914, 0.0, 0.5372996, 0.0))
    
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
            positive_mask = (self._actions > 0).to(torch.float32)
            self._actions_op = positive_mask * self.action_ub * self._actions + (1-positive_mask) * self.action_lb * (-self._actions)
        else:
            self._actions_op = self._actions.clone()
        
        # update reference and state of mpc controller
        self._update_mpc_input()
    
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
        
        self._get_state()
        for i in range(len(self.mpc)):
            self.mpc[i].set_swing_parameters(stepping_frequency=self._gait_stepping_frequency[i], foot_height=self._foot_height[i])
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
        
        self._accel_gyro_mpc = torch.from_numpy(np.array(accel_gyro)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self._grfm_mpc = torch.from_numpy(np.array(grfm)).to(self.device).view(self.num_envs, 12).to(torch.float32)
        self._gait_contact = torch.from_numpy(np.array(gait_contact)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._swing_phase = torch.from_numpy(np.array(swing_phase)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._reibert_fps = torch.from_numpy(np.array(reibert_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)
        self._augmented_fps = torch.from_numpy(np.array(augmented_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)
    
    def _split_action(self, policy_action:torch.Tensor)->torch.Tensor:
        """
        Split policy action into centroidal acceleration,
        """
        centroidal_acceleration = policy_action[:, :3]
        return centroidal_acceleration
    
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
        centroidal_accel = self._split_action(self._actions_op)
        # centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        
        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_foot_placement()
    
    def visualize_foot_placement(self):
        if self.common_step_counter % (self.cfg.rendering_interval/self.cfg.decimation) == 0:
            reibert_fps = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
            augmented_fps = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
            default_position = self._robot_api.default_root_state[:, :3]
            default_position[:, 2] = 0.0
            reibert_fps[:, 0, :2] = self._reibert_fps[:, :2]
            reibert_fps[:, 1, :2] = self._reibert_fps[:, 2:]
            augmented_fps[:, 0, :2] = self._augmented_fps[:, :2]
            augmented_fps[:, 1, :2] = self._augmented_fps[:, 2:]
            
            reibert_fps[:, 0, :] = (torch.bmm(self._init_rot_mat, reibert_fps[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position + self.scene.env_origins)
            reibert_fps[:, 1, :] = torch.bmm(self._init_rot_mat, reibert_fps[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position + self.scene.env_origins
            augmented_fps[:, 0, :] = torch.bmm(self._init_rot_mat, augmented_fps[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position + self.scene.env_origins
            augmented_fps[:, 1, :] = torch.bmm(self._init_rot_mat, augmented_fps[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position + self.scene.env_origins
            
            # apply swing mask
            reibert_fps[:, 0, :] = reibert_fps[:, 0, :] * (1-self._gait_contact[:, 0].unsqueeze(-1))
            reibert_fps[:, 1, :] = reibert_fps[:, 1, :] * (1-self._gait_contact[:, 1].unsqueeze(-1))
            augmented_fps[:, 0, :] = augmented_fps[:, 0, :] * (1-self._gait_contact[:, 0].unsqueeze(-1))
            augmented_fps[:, 1, :] = augmented_fps[:, 1, :] * (1-self._gait_contact[:, 1].unsqueeze(-1))
            
            self.foot_placement_visualizer.visualize(reibert_fps, augmented_fps)
    
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
                self._previous_actions, #44:47
                self._accel_gyro_mpc[:, :3], #47:50
                self._gait_contact, #50:52
                self._gt_contact, #52:54
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        print(f"[INFO] Reset environment {env_ids} at step {self.episode_length_buf[env_ids]}")
        # print("[INFO] Robot desired velocity: ", self._desired_twist_np.tolist())
        super()._reset_idx(env_ids)
        
        self._reset_robot(env_ids)
        
        # reset terrain parameters if one of envs is reset
        self._reset_terrain(env_ids)
        
        # logging
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for key, value in self.episode_sums.items():
            self.extras["log"].update({f"Reward_episode/{key}": value})
            self.episode_sums[key] = 0.0
    
    def _reset_terrain(self, env_ids: Sequence[int])->None:
        pass
        
    
    def _reset_robot(self, env_ids: Sequence[int])->None:
        num_curriculum_x = int(self.cfg.terrain.terrain_generator.num_cols/self.cfg.terrain.friction_group_patch_num)
        num_curriculum_y = int(self.cfg.terrain.terrain_generator.num_rows/self.cfg.terrain.friction_group_patch_num)
        curriculum_idx = np.floor(self.cfg.terrain_curriculum_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)))
        self.curriculum_idx = curriculum_idx
        
        center_coord = np.stack([self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx//num_curriculum_x), 
                                 self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx%num_curriculum_y)], axis=-1)
        position = self.cfg.robot_position_sampler.sample(center_coord, len(env_ids))
        quat = self.cfg.robot_quat_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(position))
        
        position = torch.tensor(position, device=self.device).view(-1, 3)
        quat = torch.tensor(quat, device=self.device).view(-1, 4)
        default_root_pose = torch.cat((position, quat), dim=-1)
        
        # override the default state
        self._robot_api.reset_default_pose(default_root_pose, env_ids)
        
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self._robot_api.default_root_state[env_ids, 7:]
        
        # reset joint position
        joint_pos = self._robot_api.default_joint_pos[:, self._joint_ids][env_ids]
        joint_vel = self._robot_api.default_joint_vel[:, self._joint_ids][env_ids]
        self._joint_pos[env_ids] = joint_pos
        self._joint_vel[env_ids] = joint_vel
        self._add_joint_offset(env_ids)
        
        # write to sim
        self._robot_api.write_root_pose_to_sim(default_root_pose, env_ids)
        self._robot_api.write_root_velocity_to_sim(default_root_vel, env_ids)
        self._robot_api.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids)
        
        # reset mpc reference
        self._desired_twist_np[env_ids.cpu().numpy()] = np.array(self.cfg.robot_target_velocity_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32)
        self._dsp_duration[env_ids.cpu().numpy()] = np.array(self.cfg.robot_double_support_length_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32)
        self._ssp_duration[env_ids.cpu().numpy()] = np.array(self.cfg.robot_single_support_length_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32)
        self.mpc_ctrl_counter[env_ids] = 0
        for i in env_ids.cpu().numpy():
            self.mpc[i].reset()
            self.mpc[i].update_gait_parameter(np.array([self._dsp_duration[i], self._dsp_duration[i]]), np.array([self._ssp_duration[i], self._ssp_duration[i]]))
        
        # reset reference 
        self._ref_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._ref_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        
        # update view port to look at the current active terrain
        # if self.is_rendering:
        self.viewport_camera_controller.update_view_location(eye=(center_coord[0, 0], center_coord[0, 1]-8.0, 3.0), lookat=(center_coord[0, 0], center_coord[0, 1], 0.0))
    
    def _get_rewards(self)->torch.Tensor:
        # reward
        self.height_reward, self.lin_vel_reward, self.ang_vel_reward = \
            self.cfg.reward_parameter.compute_reward(
            self._root_pos,
            self._root_lin_vel_b,
            self._root_ang_vel_b,
            self.cfg.reference_height,
            self._desired_root_lin_vel_b,
            self._desired_root_ang_vel_b,
        )
        
        self.position_reward, self.yaw_reward = self.cfg.pose_tracking_reward_parameter.compute_reward(
            self._root_pos[:, :2], # x, y
            self._root_yaw,
            self._ref_pos[:, :2], # x, y
            self._ref_yaw
        )
            
        self.alive_reward = self.cfg.alive_reward_parameter.compute_reward(self.reset_terminated, self.episode_length_buf, self.max_episode_length)
        
        # penalty
        self.roll_penalty, self.pitch_penalty, self.action_penalty, self.energy_penalty, self.foot_energy_penalty = \
            self.cfg.penalty_parameter.compute_penalty(
            self._root_quat,
            self._actions,
            self._previous_actions
        )
            
        self.vx_penalty, self.vy_penalty, self.wz_penalty = \
            self.cfg.twist_penalty_parameter.compute_penalty(
            self._root_lin_vel_b,
            self._root_ang_vel_b
            )
        
        self.foot_slide_penalty = self.cfg.foot_slide_penalty_parameter.compute_penalty(self._robot_api.body_lin_vel_w[:, -2:], self._gt_contact) # ankle at last 2 body index
        self.action_saturation_penalty = self.cfg.action_saturation_penalty_parameter.compute_penalty(self._actions)
        
        # scale rewards and penalty with time step (following reward manager in manager based rl env)
        self.height_reward = self.height_reward * self.step_dt
        self.lin_vel_reward = self.lin_vel_reward * self.step_dt
        self.ang_vel_reward = self.ang_vel_reward * self.step_dt
        self.alive_reward = self.alive_reward * self.step_dt
        self.position_reward = self.position_reward * self.step_dt
        self.yaw_reward = self.yaw_reward * self.step_dt
        
        self.roll_penalty = self.roll_penalty * self.step_dt
        self.pitch_penalty = self.pitch_penalty * self.step_dt
        self.action_penalty = self.action_penalty * self.step_dt
        self.energy_penalty = self.energy_penalty * self.step_dt
        self.foot_energy_penalty = self.foot_energy_penalty * self.step_dt
        self.vx_penalty = self.vx_penalty * self.step_dt
        self.vy_penalty = self.vy_penalty * self.step_dt
        self.wz_penalty = self.wz_penalty * self.step_dt
        self.foot_slide_penalty = self.foot_slide_penalty * self.step_dt
        self.action_saturation_penalty = self.action_saturation_penalty * self.step_dt
         
        reward = self.height_reward + self.lin_vel_reward + self.ang_vel_reward + self.alive_reward + self.position_reward + self.yaw_reward
        penalty = self.roll_penalty + self.pitch_penalty + self.action_penalty + self.energy_penalty + \
            self.foot_energy_penalty + self.vx_penalty + self.vy_penalty + self.wz_penalty + \
            self.foot_slide_penalty + self.action_saturation_penalty
         
        total_reward = reward - penalty
        
        # push logs to extras
        self.extras["log"] = dict()
        self.log_state()
        self.log_action()
        self.log_curriculum()
        self.log_episode_reward()
            
        return total_reward
    
    def _get_dones(self)->tuple[torch.Tensor, torch.Tensor]:
        # timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        roll = torch.atan2(2*(self._root_quat[:, 0]*self._root_quat[:, 1] + self._root_quat[:, 2]*self._root_quat[:, 3]), 1 - 2*(self._root_quat[:, 1]**2 + self._root_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        
        pitch = torch.asin(2*(self._root_quat[:, 0]*self._root_quat[:, 2] - self._root_quat[:, 3]*self._root_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        
        # base angle and base height violation
        roll_reset = torch.abs(roll) > self.cfg.roll_limit
        pitch_reset = torch.abs(pitch) > self.cfg.pitch_limit
        height_reset = (self._root_pos[:, 2] < self.cfg.min_height) | (self._root_pos[:, 2] > self.cfg.max_height)
        
        reset = roll_reset | pitch_reset
        reset = reset | height_reset
        
        return reset, time_out
    
    ### specific to the architecture ###
    def log_episode_reward(self)->None:
        self.episode_sums["height_reward"] += self.height_reward[0].item()
        self.episode_sums["lin_vel_reward"] += self.lin_vel_reward[0].item()
        self.episode_sums["ang_vel_reward"] += self.ang_vel_reward[0].item()
        self.episode_sums["alive_reward"] += self.alive_reward[0].item()
        self.episode_sums["position_reward"] += self.position_reward[0].item()
        self.episode_sums["yaw_reward"] += self.yaw_reward[0].item()
        
        self.episode_sums["roll_penalty"] += self.roll_penalty[0].item()
        self.episode_sums["pitch_penalty"] += self.pitch_penalty[0].item()
        self.episode_sums["action_penalty"] += self.action_penalty[0].item()
        self.episode_sums["energy_penalty"] += self.energy_penalty[0].item()
        self.episode_sums["foot_energy_penalty"] += self.foot_energy_penalty[0].item()
        self.episode_sums["vx_penalty"] += self.vx_penalty[0].item()
        self.episode_sums["vy_penalty"] += self.vy_penalty[0].item()
        self.episode_sums["wz_penalty"] += self.wz_penalty[0].item()
        self.episode_sums["feet_slide_penalty"] += self.foot_slide_penalty[0].item()
        self.episode_sums["action_saturation_penalty"] += self.action_saturation_penalty[0].item()
    
    def log_curriculum(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            curriculum_idx = self.curriculum_idx[0]
            log["curriculum/curriculum_idx"] = curriculum_idx
        self.extras["log"].update(log)
        
    def log_state(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            root_pos = self._root_pos[0].cpu().numpy()
            root_lin_vel_b = self._root_lin_vel_b[0].cpu().numpy()
            root_ang_vel_b = self._root_ang_vel_b[0].cpu().numpy()
            desired_root_lin_vel_b = self._desired_root_lin_vel_b[0].cpu().numpy()
            desired_root_ang_vel_b = self._desired_root_ang_vel_b[0].cpu().numpy()
            mpc_centroidal_accel = self._accel_gyro_mpc[0, :3].cpu().numpy()
            mpc_centroidal_ang_accel = self._accel_gyro_mpc[0, 3:].cpu().numpy()
            log["state/root_pos_x"] = root_pos[0]
            log["state/root_pos_y"] = root_pos[1]
            log["state/root_pos_z"] = root_pos[2]
            log["state/root_lin_vel_x"] = root_lin_vel_b[0]
            log["state/root_lin_vel_y"] = root_lin_vel_b[1]
            log["state/root_ang_vel_z"] = root_ang_vel_b[2]
            log["state/desired_root_lin_vel_x"] = desired_root_lin_vel_b[0]
            log["state/desired_root_lin_vel_y"] = desired_root_lin_vel_b[1]
            log["state/desired_root_ang_vel_z"] = desired_root_ang_vel_b[0]
            log["state/centroidal_accel_x"] = mpc_centroidal_accel[0]
            log["state/centroidal_accel_y"] = mpc_centroidal_accel[1]
            log["state/centroidal_accel_z"] = mpc_centroidal_accel[2]
            log["state/centroidal_ang_accel_x"] = mpc_centroidal_ang_accel[0]
            log["state/centroidal_ang_accel_y"] = mpc_centroidal_ang_accel[1]
            log["state/centroidal_ang_accel_z"] = mpc_centroidal_ang_accel[2]
        self.extras["log"].update(log)
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            # raw action
            centroidal_acceleration = self._actions[0, :3].cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            
            # clipped action
            centroidal_acceleration = self._actions_op[0, :3].cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
        self.extras["log"].update(log)



class HierarchicalArchPrime(HierarchicalArch):
    cfg: HierarchicalArchPrimeCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into centroidal acceleration and foot height.
        """
        centroidal_acceleration = policy_action[:, :3]
        centroidal_ang_acceleration = policy_action[:, 3:6]
        return centroidal_acceleration, centroidal_ang_acceleration
    
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
        centroidal_accel, centroidal_ang_accel = self._split_action(self._actions_op)
        
        # transform from local to global frame
        centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_accel = torch.bmm(self._root_rot_mat, centroidal_ang_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        self._A_residual[:, 9:12, -1] = centroidal_ang_accel.cpu().numpy()
        
        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_foot_placement()
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
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
                self._previous_actions, #44:50
                self._accel_gyro_mpc, #50:56
                self._gait_contact, #56:58
                self._swing_phase, #58:60
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    ### specific to the architecture ###
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            centroidal_acceleration = self._actions[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions[0, 3:6].cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["raw_action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["raw_action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["raw_action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            
            centroidal_acceleration = self._actions_op[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions_op[0, 3:6].cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
        self.extras["log"].update(log)
    

class HierarchicalArchAccelPF(HierarchicalArch):
    """Hierarchical Architecture with linear/angular acceleration and sagital foot placement.
    """
    cfg: HierarchicalArchAccelPFCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into centroidal acceleration and foot height.
        """
        centroidal_acceleration = policy_action[:, :3]
        centroidal_ang_acceleration = policy_action[:, 3:6]
        residual_foot_placement = policy_action[:, 6:10]
        return centroidal_acceleration, centroidal_ang_acceleration, residual_foot_placement
    
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
        centroidal_accel, centroidal_ang_accel, residual_foot_placement = self._split_action(self._actions_op)
        saggital_residual_foot_placement = residual_foot_placement[:, [0,2]] # left, right
        lateral_residual_foot_placement = residual_foot_placement[:, [1,3]] # left, right
        
        # transform from local to global frame
        centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_accel = torch.bmm(self._root_rot_mat, centroidal_ang_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        self._A_residual[:, 9:12, -1] = centroidal_ang_accel.cpu().numpy()
        self._foot_placement_residuals[:, 0] = (saggital_residual_foot_placement[:, 0] * torch.cos(self._root_yaw.squeeze()) - lateral_residual_foot_placement[:, 0] * torch.sin(self._root_yaw.squeeze())).cpu().numpy()
        self._foot_placement_residuals[:, 1] = (saggital_residual_foot_placement[:, 0] * torch.sin(self._root_yaw.squeeze()) + lateral_residual_foot_placement[:, 0] * torch.cos(self._root_yaw.squeeze())).cpu().numpy()
        self._foot_placement_residuals[:, 2] = (saggital_residual_foot_placement[:, 1] * torch.cos(self._root_yaw.squeeze()) - lateral_residual_foot_placement[:, 1] * torch.sin(self._root_yaw.squeeze())).cpu().numpy()
        self._foot_placement_residuals[:, 3] = (saggital_residual_foot_placement[:, 1] * torch.sin(self._root_yaw.squeeze()) + lateral_residual_foot_placement[:, 1] * torch.cos(self._root_yaw.squeeze())).cpu().numpy()
        
        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_foot_placement()
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
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
                self._previous_actions, #44:54
                self._accel_gyro_mpc, #54:60
                self._gait_contact, #60:62
                self._swing_phase, #62:64
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            centroidal_acceleration = self._actions[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions[0, 3:6].cpu().numpy()
            foot_placement = self._actions[0, 6:].cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["raw_action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["raw_action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["raw_action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            log["raw_action/sagital_foot_placement_left"] = foot_placement[0]
            log["raw_action/lateral_foot_placement_left"] = foot_placement[1]
            log["raw_action/sagital_foot_placement_right"] = foot_placement[2]
            log["raw_action/lateral_foot_placement_right"] = foot_placement[3]
            
            centroidal_acceleration = self._actions_op[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions_op[0, 3:6].cpu().numpy()
            foot_placement = self._actions[0, 6:].cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            log["raw_action/sagital_foot_placement_left"] = foot_placement[0]
            log["raw_action/lateral_foot_placement_left"] = foot_placement[1]
            log["raw_action/sagital_foot_placement_right"] = foot_placement[2]
            log["raw_action/lateral_foot_placement_right"] = foot_placement[3]
            
        self.extras["log"].update(log)