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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Task core
from isaaclab_tasks.direct.hector.common.task_reward import VelocityTrackingReward, AliveReward, ContactTrackingReward, PoseTrackingReward, SagittalFPSimilarityReward
from isaaclab_tasks.direct.hector.common.task_penalty import VelocityTrackingPenalty, TwistPenalty, FeetSlidePenalty, JointPenalty, ActionSaturationPenalty

from isaaclab_tasks.direct.hector.common.sampler import UniformLineSampler, UniformCubicSampler, GridCubicSampler, QuaternionSampler, CircularSampler
from isaaclab_tasks.direct.hector.common.curriculum import  CurriculumRateSampler, CurriculumLineSampler, CurriculumUniformLineSampler, CurriculumUniformCubicSampler, CurriculumQuaternionSampler
from isaaclab_tasks.direct.hector.core_cfg.terrain_cfg import CurriculumFrictionPatchTerrain, FrictionPatchTerrain

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.base_arch_cfg import BaseArchCfg

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class HierarchicalArchCfg(BaseArchCfg):
    # ================================
    # Common configurations
    # ================================
    seed = 10
    inference = False
    curriculum_inference = False
    
    dt=0.002 #500Hz 
    rendering_interval = 10 # 50Hz
    mpc_decimation = 5 # 100Hz
    decimation = 5 # 100Hz (RL)
    
    reference_height = 0.55
    nominal_gait_stepping_frequency = 1.0
    nominal_foot_height = 0.12
    
    terrain = CurriculumFrictionPatchTerrain
    
    # RL observation, action parameters
    action_space = 3
    observation_space = 54
    state_space = 54
    num_joint_actions = 10 # joint torque
    num_states = 33 # mpc state numbers
    
    # action space
    action_lb = [-1.0,-1.0,-6.0]
    action_ub = [1.0,  1.0, 6.0]
    observation_lb = -50.0
    observation_ub = 50.0
    clip_action = True # clip to -1 to 1
    scale_action = True # scale max value to action_ub
    
    # scene 
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=0.0,
        )
    
    # Curriculum sampler
    curriculum_max_steps = 10000
    center = (terrain.terrain_generator.num_cols*terrain.terrain_generator.size[0]/2, terrain.terrain_generator.num_rows*terrain.terrain_generator.size[1]/2, 0.0)
    robot_position_sampler = CircularSampler(radius=2.0, z_range=(0.55, 0.55))
    robot_quat_sampler = CurriculumQuaternionSampler(
        x_range_start=(0.0, math.pi/3), x_range_end=(0.0, 2*math.pi),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=24*10000)
    )
    
    terrain_curriculum_sampler = CurriculumLineSampler(
        x_start=0, x_end=terrain.num_curriculums-1,
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
    )
    robot_target_velocity_sampler = CurriculumUniformCubicSampler(
        x_range_start=(0.1, 0.3), x_range_end=(0.3, 0.5),
        y_range_start=(0.0, 0.0), y_range_end=(0.0, 0.0),
        z_range_start=(0.0, 0.0), z_range_end=(-0.5, 0.5),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
    )
    
    # robot_double_support_length_sampler = CurriculumUniformLineSampler(
    #     x_range_start=(int(0.0/dt), int(0.05/dt)), x_range_end=(int(0.0/dt), int(0.15/dt)),
    #     rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
    # )
    # robot_single_support_length_sampler = CurriculumUniformLineSampler(
    #     x_range_start=(int(0.3/dt), int(0.3/dt)), x_range_end=(int(0.4/dt), int(0.45/dt)),
    #     rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
    # )
    robot_double_support_length_sampler = CurriculumUniformLineSampler(
        x_range_start=(int(0.0/dt), int(0.0/dt)), x_range_end=(int(0.0/dt), int(0.0/dt)),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
    )
    robot_single_support_length_sampler = CurriculumUniformLineSampler(
        x_range_start=(int(0.2/dt), int(0.2/dt)), x_range_end=(int(0.2/dt), int(0.2/dt)),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
    )
    
    # reward parameters
    reward_parameter: VelocityTrackingReward = VelocityTrackingReward(height_similarity_weight=0.66, 
                                                            lin_vel_similarity_weight=0.66,
                                                            ang_vel_similarity_weight=0.66,
                                                            height_similarity_coeff=0.5, 
                                                            lin_vel_similarity_coeff=0.5,
                                                            ang_vel_similarity_coeff=0.5,
                                                            height_reward_mode="exponential",
                                                            lin_vel_reward_mode="exponential",
                                                            ang_vel_reward_mode="exponential")
    pose_tracking_reward_parameter: PoseTrackingReward = PoseTrackingReward(
        position_weight=0.66, yaw_weight=0.66, 
        position_coeff=0.8, yaw_coeff=0.8,
        position_reward_mode="exponential", yaw_reward_mode="exponential")
    alive_reward_parameter: AliveReward = AliveReward(alive_weight=0.33)
    
    # penalty parameters
    penalty_parameter: VelocityTrackingPenalty = VelocityTrackingPenalty(roll_deviation_weight=0.66, 
                                                               pitch_deviation_weight=0.66, 
                                                               action_penalty_weight=0.05, 
                                                               energy_penalty_weight=0.05,
                                                               foot_energy_penalty_weight=0.00)
    twist_penalty_parameter: TwistPenalty = TwistPenalty(vx_bound=(0.25, 0.5), 
                                                         vy_bound=(0.1, 0.5), 
                                                         wz_bound=(0.15, 0.5), 
                                                         vx_penalty_weight=0.1,
                                                         vy_penalty_weight=0.1,
                                                         wz_penalty_weight=0.1)
    action_saturation_penalty_parameter: ActionSaturationPenalty = ActionSaturationPenalty(action_penalty_weight=0.66, action_bound=(0.9, 1.0))
    foot_slide_penalty_parameter: FeetSlidePenalty = FeetSlidePenalty(feet_slide_weight=0.1)


@configclass
class HierarchicalArchPrimeCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 60
    state_space = 60
    action_space = 6
    
    # action bound hyper parameter
    action_lb = [-6.0, -6.0, -6.0] + [-2.0, -2.0, -2.0]
    action_ub = [6.0,  6.0, 6.0] + [2.0, 2.0, 2.0]

@configclass
class HierarchicalArchAccelPFCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 64
    state_space = 64
    action_space = 10
    
    # action bound hyper parameter
    action_lb = [-6.0, -6.0, -6.0] + [-2.0, -2.0, -2.0] + [-0.02, -0.02, -0.01, -0.01]
    action_ub = [6.0,  6.0, 6.0] + [2.0, 2.0, 2.0] + [0.1, 0.1, 0.01, 0.01]


@configclass
class HierarchicalArchPrimeFullCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 72
    state_space = 72
    action_space = 18
    # action bound hyper parameter
    g= 9.81
    lin_accel_scale = [0.3*g, 0.3*g, 0.5*g]
    # ang_accel_scale = [27.62, 5.23, 5.23]
    ang_accel_scale = [0.5*27.62, 0.5*5.23, 0.5*5.23]
    # +- 20% uncertainty of original mass and inertia

    inv_mass_scale = [0.2/13.856, 0.2/13.856, 0.2/13.856]
    inv_inertia_scale = [0.2/0.5413, 0.2/0.52, 0.2/0.0691]

    action_lb = [-v for v in lin_accel_scale] + [-v for v in ang_accel_scale] + \
        [-v for v in inv_mass_scale] + [-v for v in inv_mass_scale] + \
        [-v for v in inv_inertia_scale] + [-v for v in inv_inertia_scale]
    action_ub = lin_accel_scale + ang_accel_scale + inv_mass_scale + inv_mass_scale + \
        inv_inertia_scale + inv_inertia_scale