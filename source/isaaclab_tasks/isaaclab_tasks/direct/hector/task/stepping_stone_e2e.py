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
from isaaclab_tasks.direct.hector.common.visualization_marker import FootPlacementVisualizer, VelocityVisualizer, SwingFootVisualizer

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.stepping_stone_e2e_cfg import SteppingStoneE2ECfg

# Base class
from isaaclab_tasks.direct.hector.task.base_arch_e2e import BaseArchE2E


class SteppingStoneE2E(BaseArchE2E):
    cfg: SteppingStoneE2ECfg
    def __init__(self, cfg: SteppingStoneE2ECfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_history = self.cfg.num_history
        self.history_buffer = HistoryBuffer(self.num_envs, self.num_history, self.cfg.observation_space, torch.float32, self.device)
        self.roughness_at_fps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
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
        
        # base terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.0, 0.0, 0.0, 1.0))
        
        # update viewer
        terrain_origin = np.array([self.cfg.terrain.center_position[0], self.cfg.terrain.center_position[1], 0.0])
        camera_pos = terrain_origin
        camera_delta = np.array([0.0, -7.0, 0.5])
        self.cfg.viewer.eye = (camera_pos[0]+camera_delta[0], camera_pos[1]+camera_delta[1], camera_pos[2]+camera_delta[2])
        self.cfg.viewer.lookat = (camera_pos[0], camera_pos[1], 0.0)
        
        # visualization marker
        self._velocity_visualizer = VelocityVisualizer("/Visuals/velocity_visualizer")
    
    
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into useful form
        """
        stepping_frequency_traj = policy_action[:, :self.cfg.traj_sample]
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
        # """
        # swing_foot_pos = self._foot_pos_b.reshape(self.num_envs, 2, 3)[self._gt_contact==0]
        # swing_foot_pos[:, 0] += 0.05 # track toe pos
        # px = (swing_foot_pos[:, 0]//0.05).long() + int(1.0//0.05+1)/2
        # py = -(swing_foot_pos[:, 1]//0.05).long() + int(1.0//0.05+1)/2
        # indices = (int(1.0/0.05 + 1)*px + py).long()
        # self._desired_height = self.cfg.reference_height + (self._raycaster.data.pos_w[:, 2] + self.height_map[torch.arange(self.num_envs), indices]).cpu().numpy() # type: ignore
        super()._apply_action()
        self.visualize_marker()
    
    def visualize_marker(self):
        self._velocity_visualizer.visualize(self._robot_api.root_pos_w, self._robot_api.root_quat_w, self._robot_api.root_lin_vel_b)
    
    # def _get_ground_gradient_at_fps(self):
    #     """
    #     get neighboring 3x3 pixel of ground gradient at foot placement
    #     """
    #     foot_placement = self._augmented_fps_b.reshape(self.num_envs, 2, 2)[self._gait_contact==0]
    #     height, width = self.height_map_2d_grad.shape[1:3]
        
    #     row_index = (-(foot_placement[:, 1]//0.05).long() + int((height-1)/2)).long()
    #     row_index = torch.clamp(row_index, 0, height-1).unsqueeze(1).unsqueeze(2).repeat(1, 3, 3)
    #     col_index = ((foot_placement[:, 0]//0.05).long() + int((width-1)/2)).long()
    #     col_index = torch.clamp(col_index, 0, width-1).unsqueeze(1).unsqueeze(2).repeat(1, 3, 3)
        
    #     row_index[:, :, 0] = row_index[:, :, 1] - 1
    #     row_index[:, :, 2] = row_index[:, :, 1] + 1
    #     col_index[:, 0, :] = col_index[:, 1, :] - 1
    #     col_index[:, 2, :] = col_index[:, 1, :] + 1
        
    #     indices = (width*col_index.view(self.num_envs, -1) + row_index.view(self.num_envs, -1)).long() # (num_envs, 9)
    #     indices = torch.clamp(indices, 0, height*width-1).view(-1)
        
    #     env_ids = torch.arange(self.num_envs).view(-1, 1).repeat(1,9).view(-1)
    #     ground_gradient = self.height_map_2d_grad.reshape(self.num_envs, -1).repeat(9, 1)[env_ids, indices] # type: ignore
    #     self.roughness_at_fps = ground_gradient.view(self.num_envs, 9).mean(dim=1)
        
        
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
        self._get_exteroceptive_observation()
        # self._get_ground_gradient_at_fps()
        
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
    
    
    def _get_rewards(self)->torch.Tensor:
        # reward
        self.height_reward, self.lin_vel_reward, self.ang_vel_reward = \
            self.cfg.reward_parameter.compute_reward(
            self._root_pos,
            self._root_lin_vel_b,
            self._root_ang_vel_b,
            self._desired_height,
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
        self.roll_penalty, self.pitch_penalty = self.cfg.orientation_penalty_parameter.compute_penalty(self._root_quat)
        self.velocity_penalty = self.cfg.velocity_penalty_parameter.compute_penalty(self._root_lin_vel_b)
        self.ang_velocity_penalty = self.cfg.angular_velocity_penalty_parameter.compute_penalty(self._root_ang_vel_b)
        
        self.foot_slide_penalty = self.cfg.foot_slide_penalty_parameter.compute_penalty(self._robot_api.body_lin_vel_w[:, -2:, :2], self._gt_contact) # ankle at last 2 body index
        self.foot_distance_penalty = self.cfg.foot_distance_penalty_parameter.compute_penalty(self._foot_pos_b[:, :3], self._foot_pos_b[:, 3:])

        self.toe_left_joint_penalty = self.cfg.toe_left_joint_penalty_parameter.compute_penalty(self._robot_api.joint_pos[:, -2])
        self.toe_right_joint_penalty = self.cfg.toe_right_joint_penalty_parameter.compute_penalty(self._robot_api.joint_pos[:, -1])
        
        # self.contact_location_penalty = self.cfg.contact_location_penalty.compute_penalty(self.roughness_at_fps)
        
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
        
        self.roll_penalty = self.roll_penalty * self.step_dt
        self.pitch_penalty = self.pitch_penalty * self.step_dt

        self.velocity_penalty = self.velocity_penalty * self.step_dt
        self.ang_velocity_penalty = self.ang_velocity_penalty * self.step_dt

        self.foot_slide_penalty = self.foot_slide_penalty * self.step_dt
        self.foot_distance_penalty = self.foot_distance_penalty * self.step_dt

        self.toe_left_joint_penalty = self.toe_left_joint_penalty * self.step_dt
        self.toe_right_joint_penalty = self.toe_right_joint_penalty * self.step_dt
        
        # self.contact_location_penalty = self.contact_location_penalty * self.step_dt

        self.action_penalty = self.action_penalty * self.step_dt
        self.energy_penalty = self.energy_penalty * self.step_dt
        self.action_saturation_penalty = self.action_saturation_penalty * self.step_dt
        self.torque_penalty = self.torque_penalty * self.step_dt
         
        self.reward = self.height_reward + self.lin_vel_reward + self.ang_vel_reward + self.position_reward + self.yaw_reward + self.alive_reward
        
        self.penalty = self.roll_penalty + self.pitch_penalty + self.action_penalty + self.energy_penalty + \
            self.foot_slide_penalty + self.action_saturation_penalty + self.foot_distance_penalty + \
            self.toe_left_joint_penalty + self.toe_right_joint_penalty + \
                self.torque_penalty + self.velocity_penalty + self.ang_velocity_penalty
        
        total_reward = self.reward - self.penalty
        
        # push logs to extras
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.log_state()
        self.log_action()
        self.log_episode_return()
        self.log_reward()
        return total_reward
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # log reset episode length
        self.max_episode_length_buf[env_ids] = self.episode_length_buf[env_ids]
        super()._reset_idx(env_ids)
        
        self._reset_robot(env_ids)
        
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
        curriculum_idx = np.floor(
            self.cfg.terrain_curriculum_sampler.sample(
                self.common_step_counter//self.cfg.num_steps_per_env, 
                self.max_episode_length_buf.float().mean(dim=0).item(), len(env_ids)
                )
            )
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
        
        # print(self._joint_pos.shape)
        # print(self._joint_pos[env_ids].shape)
        self._joint_pos[env_ids] = joint_pos
        self._joint_vel[env_ids] = joint_vel
        self._joint_pos[env_ids] = self._add_joint_offset(self._joint_pos[env_ids]) # type: ignore
        
        ### write reset to sim ###
        self._robot_api.write_root_pose_to_sim(default_root_pose, env_ids) # type: ignore
        self._robot_api.write_root_velocity_to_sim(self._robot_api.default_root_state[env_ids, 7:], env_ids) # type: ignore
        self._robot_api.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids) # type: ignore
        
        # set command velocity
        twist_cmd = np.array(self.cfg.robot_target_velocity_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32) # type: ignore
        self._desired_root_lin_vel_b[env_ids] = torch.tensor(twist_cmd[:, :2], device=self.device) # type: ignore
        self._desired_root_ang_vel_b[env_ids] = torch.tensor(twist_cmd[:, 2:], device=self.device) # type: ignore
        
        # reset reference trajectory
        self._ref_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._ref_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        
        if not self.cfg.inference:
            # update view port to look at the current active terrain
            camera_delta = [0.0, -10.0, 3.0]
            self.viewport_camera_controller.update_view_location(
                eye=(center_coord[0, 0]+camera_delta[0], center_coord[0, 1]+camera_delta[1], camera_delta[2]), 
                lookat=(center_coord[0, 0], center_coord[0, 1], 0.0)) # type: ignore
    
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
    
    def log_reward(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            log["reward/height_reward"] = self.height_reward.mean().item()
            log["reward/lin_vel_reward"] = self.lin_vel_reward.mean().item()
            log["reward/ang_vel_reward"] = self.ang_vel_reward.mean().item()
            log["reward/alive_reward"] = self.alive_reward.mean().item()
            log["reward/position_reward"] = self.position_reward.mean().item()
            log["reward/yaw_reward"] = self.yaw_reward.mean().item()
            log["reward/total_reward"] = self.reward.mean().item()
            
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
            log["penalty/total_penalty"] = self.penalty.mean().item()
            

        self.extras["log"].update(log)
    
    def log_state(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            root_pos = self._root_pos.mean(dim=0).cpu().numpy()
            root_lin_vel_b = self._root_lin_vel_b.mean(dim=0).cpu().numpy()
            root_ang_vel_b = self._root_ang_vel_b.mean(dim=0).cpu().numpy()
            desired_root_lin_vel_b = self._desired_root_lin_vel_b.mean(dim=0).cpu().numpy()
            desired_root_ang_vel_b = self._desired_root_ang_vel_b.mean(dim=0).cpu().numpy()

            log["state/root_pos_x"] = root_pos[0]
            log["state/root_pos_y"] = root_pos[1]
            log["state/root_pos_z"] = root_pos[2]
            log["state/root_lin_vel_x"] = root_lin_vel_b[0]
            log["state/root_lin_vel_y"] = root_lin_vel_b[1]
            log["state/root_ang_vel_z"] = root_ang_vel_b[2]
            log["state/desired_root_lin_vel_x"] = desired_root_lin_vel_b[0]
            log["state/desired_root_lin_vel_y"] = desired_root_lin_vel_b[1]
            log["state/desired_root_ang_vel_z"] = desired_root_ang_vel_b[0]
        self.extras["log"].update(log)

    def log_curriculum(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            curriculum_idx = self.curriculum_idx[0]
            log["curriculum/curriculum_idx"] = curriculum_idx
        self.extras["log"].update(log)
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env == 0:
            joint_action = self._joint_actions.mean(dim=0).cpu().numpy()
            log["action/left_hip__yaw"] = joint_action[0]
            log["action/left_hip_roll"] = joint_action[1]
            log["action/left_hip_pitch"] = joint_action[2]
            log["action/left_knee"] = joint_action[3]
            log["action/left_ankle_pitch"] = joint_action[4]
            
            log["action/right_hip_yaw"] = joint_action[5]
            log["action/right_hip_roll"] = joint_action[6]
            log["action/right_hip_pitch"] = joint_action[7]
            log["action/right_knee"] = joint_action[8]
            log["action/right_ankle_pitch"] = joint_action[9]
            
        self.extras["log"].update(log)