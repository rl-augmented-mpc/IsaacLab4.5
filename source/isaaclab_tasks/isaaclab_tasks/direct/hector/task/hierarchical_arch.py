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
from  isaaclab_tasks.direct.hector.common.utils.data_util import HistoryBuffer
from isaaclab_tasks.direct.hector.common.visualization_marker import FootPlacementVisualizer, VelocityVisualizer, SwingFootVisualizer

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg, HierarchicalArchPrimeCfg, HierarchicalArchPrimeFullCfg

# Base class
from isaaclab_tasks.direct.hector.task.base_arch import BaseArch
    

class HierarchicalArch(BaseArch):
    cfg: HierarchicalArchCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.curriculum_idx = np.zeros(self.num_envs)
        
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
        self._robot_api = RobotCore(self._robot, self.num_envs, self.cfg.foot_patch_num)
        self.scene.articulations["robot"] = self._robot
        
        # sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # ray caster
        self._raycaster = RayCaster(self.cfg.ray_caster)
        self.scene.sensors["raycaster"] = self._raycaster
        
        # base terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # visualization marker
        self.foot_placement_visualizer = FootPlacementVisualizer("/Visuals/foot_placement")
        self._velocity_visualizer = VelocityVisualizer("/Visuals/velocity_visualizer")
        self.swing_foot_visualizer = SwingFootVisualizer("/Visuals/swing_foot_visualizer")
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.0, 0.0, 0.0, 1.0))
        # self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.8433914, 0.0, 0.5372996, 0.0))
        
        # update viewer
        curriculum_idx = 0
        num_curriculum_x = int(self.cfg.terrain.terrain_generator.num_cols/self.cfg.terrain.friction_group_patch_num)
        num_curriculum_y = int(self.cfg.terrain.terrain_generator.num_rows/self.cfg.terrain.friction_group_patch_num)
        terrain_origin = np.array([self.cfg.terrain.center_position[0], self.cfg.terrain.center_position[1], 0.0])
        camera_pos = terrain_origin + np.array([self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx//num_curriculum_x), 
                                 self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx%num_curriculum_y), 0])
        camera_delta = np.array([0.0, -4.0, 0.0])
        self.cfg.viewer.eye = (camera_pos[0]+camera_delta[0], camera_pos[1]+camera_delta[1], camera_pos[2]+camera_delta[2])
        self.cfg.viewer.lookat = (camera_pos[0], camera_pos[1], 0.0)
    
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
        
        if self.cfg.curriculum_inference:
            # update reference and state of mpc controller
            env_ids = self._robot._ALL_INDICES
            robot_twist = np.array(self.cfg.robot_target_velocity_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32)
            self._desired_twist_np[env_ids.cpu().numpy()] = robot_twist # type: ignore
        self._update_mpc_input()
    
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
        
        # get proprioceptive
        self._get_state()

        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_marker()
    
    def visualize_marker(self):
        reibert_fps = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
        augmented_fps = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
        default_position = self._robot_api.default_root_state[:, :3]
        default_position[:, 2] = self._robot_api.root_pos_w[:, 2] - self._root_pos[:, 2]
        reibert_fps[:, 0, :2] = self._reibert_fps[:, :2]
        reibert_fps[:, 1, :2] = self._reibert_fps[:, 2:]
        augmented_fps[:, 0, :2] = self._augmented_fps[:, :2]
        augmented_fps[:, 1, :2] = self._augmented_fps[:, 2:]
        
        # convert local foot placement to simulation global frame
        reibert_fps[:, 0, :] = torch.bmm(self._init_rot_mat, reibert_fps[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position
        reibert_fps[:, 1, :] = torch.bmm(self._init_rot_mat, reibert_fps[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position 
        augmented_fps[:, 0, :] = torch.bmm(self._init_rot_mat, augmented_fps[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position
        augmented_fps[:, 1, :] = torch.bmm(self._init_rot_mat, augmented_fps[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position
        
        # hide foot placement marker when foot is in contact
        reibert_fps[:, 0, 2] -= self._gait_contact[:, 0] * 5.0
        reibert_fps[:, 1, 2] -= self._gait_contact[:, 1] * 5.0
        augmented_fps[:, 0, 2] -= self._gait_contact[:, 0] * 5.0
        augmented_fps[:, 1, 2] -= self._gait_contact[:, 1] * 5.0
        
        
        # swing foot
        left_swing = (self._root_rot_mat @ self._ref_foot_pos_b[:, :3].unsqueeze(2)).squeeze(2) + self._root_pos
        right_swing = (self._root_rot_mat @ self._ref_foot_pos_b[:, 3:].unsqueeze(2)).squeeze(2) + self._root_pos
        
        left_swing = (self._init_rot_mat @ left_swing.unsqueeze(-1)).squeeze(-1) + default_position
        right_swing = (self._init_rot_mat @ right_swing.unsqueeze(-1)).squeeze(-1) + default_position
        swing_reference = torch.stack((left_swing, right_swing), dim=1)
        
        orientation = self._robot_api.root_quat_w.repeat(4, 1)
        
        self.foot_placement_visualizer.visualize(reibert_fps, augmented_fps, orientation)
        self._velocity_visualizer.visualize(self._robot_api.root_pos_w, self._robot_api.root_quat_w, self._robot_api.root_lin_vel_b)
        self.swing_foot_visualizer.visualize(swing_reference)
    
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
        observation = {"policy": self._obs, "critic": self._obs}
        return observation
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # log reset episode length
        self.max_episode_length_buf[env_ids] = self.episode_length_buf[env_ids]
        super()._reset_idx(env_ids)
        
        self._reset_robot(env_ids)
        self._reset_terrain(env_ids)
        
        # log
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.log_curriculum()
        
        # log episode reward
        for key in self.episode_reward_sums.keys():
            episode_sums = torch.mean(self.episode_reward_sums[key][env_ids])
            self.extras["log"].update({f"Episode_reward/{key}": episode_sums/self.max_episode_length_s})
            self.episode_reward_sums[key][env_ids] = 0.0
        for key in self.episode_penalty_sums.keys():
            episode_sums = torch.mean(self.episode_penalty_sums[key][env_ids])
            self.extras["log"].update({f"Episode_penalty/{key}": episode_sums/self.max_episode_length_s})
            self.episode_penalty_sums[key][env_ids] = 0.0
    
    def _reset_terrain(self, env_ids: Sequence[int])->None:
        pass
    
    def _get_sub_terrain_center(self, curriculum_idx:np.ndarray):
        if self.cfg.terrain.terrain_type == "generator":
            # terrain left bottom corner is at the origin
            terrain_size_x = self.cfg.terrain.terrain_generator.size[0]*self.cfg.terrain.terrain_generator.num_cols
            terrain_size_y = self.cfg.terrain.terrain_generator.size[1]*self.cfg.terrain.terrain_generator.num_rows
            
            num_tiles_per_curriculum = self.cfg.terrain.terrain_generator.num_cols// int(math.sqrt(self.cfg.terrain.num_curriculums))
            nx = self.cfg.terrain.terrain_generator.num_cols // num_tiles_per_curriculum # number of sub-terrain in x direction
            ny = self.cfg.terrain.terrain_generator.num_rows // num_tiles_per_curriculum # number of sub-terrain in y direction
            center_coord = \
                np.stack([(self.cfg.terrain.terrain_generator.size[0] * num_tiles_per_curriculum)/2 + \
                    (self.cfg.terrain.terrain_generator.size[0] * num_tiles_per_curriculum)*(curriculum_idx//nx) - terrain_size_x//2, 
                        (self.cfg.terrain.terrain_generator.size[1] * num_tiles_per_curriculum)/2 + \
                            (self.cfg.terrain.terrain_generator.size[1] * num_tiles_per_curriculum)*(curriculum_idx%ny) - terrain_size_y//2], 
                        axis=-1)
        elif self.cfg.terrain.terrain_type == "patched":
            # terrain center is at the origin
            num_curriculum_x = int(self.cfg.terrain.terrain_generator.num_cols/self.cfg.terrain.friction_group_patch_num)
            num_curriculum_y = int(self.cfg.terrain.terrain_generator.num_rows/self.cfg.terrain.friction_group_patch_num)
            
            final_curriculum_mask = self.common_step_counter//self.cfg.num_steps_per_env >= 4000
            curriculum_idx[final_curriculum_mask] = (self.cfg.terrain.num_curriculums - 1) * np.ones_like(curriculum_idx[final_curriculum_mask])
            
            center_coord = np.stack([
                self.cfg.terrain.terrain_generator.size[0]*(self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx//num_curriculum_x)), 
                self.cfg.terrain.terrain_generator.size[1]*(self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx%num_curriculum_y))], 
                                    axis=-1)
        else:
            center_coord = np.zeros((len(curriculum_idx), 2))
        return center_coord
    
    def _reset_robot(self, env_ids: Sequence[int])->None:
        ### set position ###
        curriculum_idx = np.floor(self.cfg.terrain_curriculum_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)))
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
            camera_delta = [0, -7.0, 1.0]
            self.viewport_camera_controller.update_view_location(
                eye=(center_coord[0, 0]+camera_delta[0], center_coord[0, 1]+camera_delta[1], camera_delta[2]), 
                lookat=(center_coord[0, 0], center_coord[0, 1], 0.0)) # type: ignore
    
    def _get_rewards(self)->torch.Tensor:
        # reward
        self.height_reward, self.lin_vel_reward, self.ang_vel_reward = \
            self.cfg.reward_parameter.compute_reward(
            self._root_pos,
            self._root_lin_vel_b,
            self._root_ang_vel_b,
            torch.from_numpy(self._desired_height).to(self.device),
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
        
        self.swing_foot_tracking_reward = self.cfg.swing_foot_tracking_reward_parameter.compute_reward(
            (self._foot_pos_b.reshape(-1, 2, 3) * (1-self._gait_contact).reshape(-1, 2, 1)).reshape(-1, 6), 
            (self._ref_foot_pos_b.reshape(-1, 2, 3) * (1-self._gait_contact).reshape(-1, 2, 1)).reshape(-1, 6),
        )
            
        
        # penalty
        self.roll_penalty, self.pitch_penalty = self.cfg.orientation_penalty_parameter.compute_penalty(self._root_quat)
        self.velocity_penalty = self.cfg.velocity_penalty_parameter.compute_penalty(self._root_lin_vel_b)
        self.ang_velocity_penalty = self.cfg.angular_velocity_penalty_parameter.compute_penalty(self._root_ang_vel_b)
        
        self.foot_slide_penalty = self.cfg.foot_slide_penalty_parameter.compute_penalty(self._robot_api.body_lin_vel_w[:, -2:, :2], self._gt_contact) # ankle at last 2 body index
        self.foot_distance_penalty = self.cfg.foot_distance_penalty_parameter.compute_penalty(self._foot_pos_b[:, :3], self._foot_pos_b[:, 3:])

        self.toe_left_joint_penalty = self.cfg.toe_left_joint_penalty_parameter.compute_penalty(self._robot_api.joint_pos[:, -2])
        self.toe_right_joint_penalty = self.cfg.toe_right_joint_penalty_parameter.compute_penalty(self._robot_api.joint_pos[:, -1])
        
        self.action_penalty, self.energy_penalty = self.cfg.action_penalty_parameter.compute_penalty(
            self._actions, 
            self._previous_actions, 
            self.common_step_counter//self.cfg.num_steps_per_env)
        
        self.torque_penalty = self.cfg.torque_penalty_parameter.compute_penalty(self._joint_actions, self.common_step_counter//self.cfg.num_steps_per_env)
        self.action_saturation_penalty = self.cfg.action_saturation_penalty_parameter.compute_penalty(self._actions)
        
        # scale rewards and penalty with time step (following reward manager in manager based rl env)
        self.height_reward = self.height_reward * self.step_dt
        self.lin_vel_reward = self.lin_vel_reward * self.step_dt
        self.ang_vel_reward = self.ang_vel_reward * self.step_dt
        self.alive_reward = self.alive_reward * self.step_dt
        self.position_reward = self.position_reward * self.step_dt
        self.yaw_reward = self.yaw_reward * self.step_dt
        self.swing_foot_tracking_reward = self.swing_foot_tracking_reward * self.step_dt
        
        self.roll_penalty = self.roll_penalty * self.step_dt
        self.pitch_penalty = self.pitch_penalty * self.step_dt

        self.velocity_penalty = self.velocity_penalty * self.step_dt
        self.ang_velocity_penalty = self.ang_velocity_penalty * self.step_dt

        self.foot_slide_penalty = self.foot_slide_penalty * self.step_dt
        self.foot_distance_penalty = self.foot_distance_penalty * self.step_dt

        self.toe_left_joint_penalty = self.toe_left_joint_penalty * self.step_dt
        self.toe_right_joint_penalty = self.toe_right_joint_penalty * self.step_dt

        self.action_penalty = self.action_penalty * self.step_dt
        self.energy_penalty = self.energy_penalty * self.step_dt
        self.action_saturation_penalty = self.action_saturation_penalty * self.step_dt
        self.torque_penalty = self.torque_penalty * self.step_dt
         
        reward = self.height_reward + self.lin_vel_reward + self.ang_vel_reward + self.position_reward + self.yaw_reward + \
            self.swing_foot_tracking_reward + self.alive_reward
        
        penalty = self.roll_penalty + self.pitch_penalty + self.action_penalty + self.energy_penalty + \
            self.foot_slide_penalty + self.action_saturation_penalty + self.foot_distance_penalty + \
            self.toe_left_joint_penalty + self.toe_right_joint_penalty + \
                self.torque_penalty + self.velocity_penalty + self.ang_velocity_penalty
        
        total_reward = reward - penalty
        
        # push logs to extras
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.log_state()
        self.log_action()
        self.log_episode_return()
        self.log_reward()
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
    
    def log_episode_return(self)->None:
        self.episode_reward_sums["height_reward"] += self.height_reward
        self.episode_reward_sums["lin_vel_reward"] += self.lin_vel_reward
        self.episode_reward_sums["ang_vel_reward"] += self.ang_vel_reward
        self.episode_reward_sums["alive_reward"] += self.alive_reward
        self.episode_reward_sums["position_reward"] += self.position_reward
        self.episode_reward_sums["yaw_reward"] += self.yaw_reward
        self.episode_reward_sums["swing_foot_tracking_reward"] += self.swing_foot_tracking_reward
        
        self.episode_penalty_sums["roll_penalty"] += self.roll_penalty
        self.episode_penalty_sums["pitch_penalty"] += self.pitch_penalty
        self.episode_penalty_sums["velocity_penalty"] += self.velocity_penalty
        self.episode_penalty_sums["ang_velocity_penalty"] += self.ang_velocity_penalty
        self.episode_penalty_sums["feet_slide_penalty"] += self.foot_slide_penalty
        self.episode_penalty_sums["foot_distance_penalty"] += self.foot_distance_penalty
        self.episode_penalty_sums["action_penalty"] += self.action_penalty
        self.episode_penalty_sums["energy_penalty"] += self.energy_penalty
        self.episode_penalty_sums["action_saturation_penalty"] += self.action_saturation_penalty
        self.episode_penalty_sums["torque_penalty"] += self.torque_penalty
        self.episode_penalty_sums["toe_left_joint_penalty"] += self.toe_left_joint_penalty
        self.episode_penalty_sums["toe_right_joint_penalty"] += self.toe_right_joint_penalty
        
    
    def log_curriculum(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            curriculum_idx = self.curriculum_idx[0]
            log["curriculum/curriculum_idx"] = curriculum_idx
        self.extras["log"].update(log)
    
    def log_reward(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            log["reward/height_reward"] = self.height_reward.mean().item()
            log["reward/lin_vel_reward"] = self.lin_vel_reward.mean().item()
            log["reward/ang_vel_reward"] = self.ang_vel_reward.mean().item()
            log["reward/alive_reward"] = self.alive_reward.mean().item()
            log["reward/position_reward"] = self.position_reward.mean().item()
            log["reward/yaw_reward"] = self.yaw_reward.mean().item()
            log["reward/swing_foot_tracking_reward"] = self.swing_foot_tracking_reward.mean().item()
            
            log["penalty/roll_penalty"] = self.roll_penalty.mean().item()
            log["penalty/pitch_penalty"] = self.pitch_penalty.mean().item()
            log["penalty/velocity_penalty"] = self.velocity_penalty.mean().item()
            log["penalty/ang_velocity_penalty"] = self.ang_velocity_penalty.mean().item()
            log["penalty/feet_slide_penalty"] = self.foot_slide_penalty.mean().item()
            log["penalty/foot_distance_penalty"] = self.foot_distance_penalty.mean().item()
            log["penalty/action_penalty"] = self.action_penalty.mean().item()
            log["penalty/energy_penalty"] = self.energy_penalty.mean().item()
            log["penalty/action_saturation_penalty"] = self.action_saturation_penalty.mean().item()
            log["penalty/torque_penalty"] = self.torque_penalty.mean().item()
            log["penalty/toe_left_joint_penalty"] = self.toe_left_joint_penalty.mean().item()
            log["penalty/toe_right_joint_penalty"] = self.toe_right_joint_penalty.mean().item()

        self.extras["log"].update(log)
    
    def log_state(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            root_pos = self._root_pos.mean(dim=0).cpu().numpy()
            root_lin_vel_b = self._root_lin_vel_b.mean(dim=0).cpu().numpy()
            root_ang_vel_b = self._root_ang_vel_b.mean(dim=0).cpu().numpy()
            desired_root_lin_vel_b = self._desired_root_lin_vel_b.mean(dim=0).cpu().numpy()
            desired_root_ang_vel_b = self._desired_root_ang_vel_b.mean(dim=0).cpu().numpy()
            mpc_centroidal_accel = self._accel_gyro_mpc[:, :3].mean(dim=0).cpu().numpy()
            mpc_centroidal_ang_accel = self._accel_gyro_mpc[:, 3:].mean(dim=0).cpu().numpy()

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
            centroidal_acceleration = self._actions[:, :3].mean(dim=0).cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            
            # clipped action
            centroidal_acceleration = self._actions_op[:, 3:].mean(dim=0).cpu().numpy()
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
        
        # get proprioceptive
        self._get_state()

        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_marker()
    
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
        observation = {"policy": self._obs, "critic": self._obs}
        return observation
    
    ### specific to the architecture ###
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            centroidal_acceleration = self._actions[:, :3].mean(dim=0).cpu().numpy()
            centroidal_ang_acceleration = self._actions[:, 3:6].mean(dim=0).cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["raw_action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["raw_action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["raw_action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            
            centroidal_acceleration = self._actions_op[:, :3].mean(dim=0).cpu().numpy()
            centroidal_ang_acceleration = self._actions_op[:, 3:6].mean(dim=0).cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
        self.extras["log"].update(log)


class HierarchicalArchPrimeFull(HierarchicalArch):
    cfg: HierarchicalArchPrimeFullCfg
    def __init__(self, cfg: HierarchicalArchPrimeFullCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_history = self.cfg.num_history
        self.history_buffer = HistoryBuffer(self.num_envs, self.num_history, self.cfg.observation_space, torch.float32, self.device)
        
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into centroidal acceleration and foot height.
        """
        centroidal_acceleration = policy_action[:, :3]
        centroidal_ang_acceleration = policy_action[:, 3:6]
        added_mass_inv1 = policy_action[:, 6:9]
        added_mass_inv2 = policy_action[:, 9:12]
        added_inertia_inv1 = policy_action[:, 12:15]
        added_inertia_inv2 = policy_action[:, 15:18]
        return centroidal_acceleration, centroidal_ang_acceleration, added_mass_inv1, added_mass_inv2, added_inertia_inv1, added_inertia_inv2
    
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
        centroidal_accel, centroidal_ang_accel, added_mass_inv1, added_mass_inv2, added_inertia_inv1, added_inertia_inv2 = self._split_action(self._actions_op)

        # form residual dynamics matrix
        centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_accel = torch.bmm(self._root_rot_mat, centroidal_ang_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        self._A_residual[:, 9:12, -1] = centroidal_ang_accel.cpu().numpy()
        self._B_residual[:, 6:9, 6:9] = torch.diag_embed(added_inertia_inv1).cpu().numpy()
        self._B_residual[:, 6:9, 9:12] = torch.diag_embed(added_inertia_inv2).cpu().numpy()
        self._B_residual[:, 9:12, 0:3] = torch.diag_embed(added_mass_inv1).cpu().numpy()
        self._B_residual[:, 9:12, 3:6] = torch.diag_embed(added_mass_inv2).cpu().numpy()

        # get proprioceptive
        self._get_state()
        
        # run mpc controller
        self._run_mpc()

        # run low level control with updated GRFM
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
                self._accel_gyro_mpc, #62:68
                self._gait_contact, #68:70
                self._swing_phase, #70:72
                self._previous_actions, #44:62
            ),
            dim=-1,
        )
        buffer_mask = self.history_buffer.size >= self.num_history
        if buffer_mask.any():
            reset_id = torch.nonzero(buffer_mask, as_tuple=True)[0]
            self.history_buffer.pop(reset_id)
        self.history_buffer.push(self._obs)
        # observation = {"policy": self._obs}
        obs = self.history_buffer.data_flat
        observation = {"policy": obs, "critic": obs}
        return observation
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            centroidal_acceleration = self._actions[:, :3].mean(dim=0).cpu().numpy()
            centroidal_ang_acceleration = self._actions[:, 3:6].mean(dim=0).cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["raw_action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["raw_action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["raw_action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]

            centroidal_acceleration = self._actions_op[:, :3].mean(dim=0).cpu().numpy()
            centroidal_ang_acceleration = self._actions_op[:, 3:6].mean(dim=0).cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
        self.extras["log"].update(log)