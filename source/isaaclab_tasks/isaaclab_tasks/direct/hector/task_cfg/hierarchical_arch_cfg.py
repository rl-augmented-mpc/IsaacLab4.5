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
from isaaclab_tasks.direct.hector.common.task_reward import VelocityTrackingReward, AliveReward, ContactTrackingReward, PoseTrackingReward, \
    SagittalFPSimilarityReward, SwingFootTrackingReward
from isaaclab_tasks.direct.hector.common.task_penalty import OrientationRegularizationPenalty, ActionRegularizationPenalty, \
    TwistPenalty, FeetSlidePenalty, JointPenalty, ActionSaturationPenalty, TerminationPenalty, CurriculumActionRegularizationPenalty, \
        FootDistanceRegularizationPenalty, CurriculumTorqueRegularizationPenalty, VelocityPenalty, AngularVelocityPenalty

# Task cfg
from isaaclab_tasks.direct.hector.common.sampler import UniformLineSampler, UniformCubicSampler, GridCubicSampler, QuaternionSampler, CircularSampler, CircularOrientationSampler
from isaaclab_tasks.direct.hector.common.curriculum import  CurriculumRateSampler, CurriculumLineSampler, CurriculumUniformLineSampler, CurriculumUniformCubicSampler, CurriculumQuaternionSampler
from isaaclab_tasks.direct.hector.core_cfg import terrain_cfg

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
    
    dt=1/500 #500Hz 
    rendering_interval = 10 # 50Hz
    mpc_decimation = 5 # 100Hz
    decimation = 5 # 100Hz (RL)
    
    reference_height = 0.55
    nominal_gait_stepping_frequency = 1.0
    nominal_foot_height = 0.12
    
    terrain = terrain_cfg.CurriculumFrictionPatchTerrain
    
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
    curriculum_max_steps = 3000
    center = (terrain.center_position[0], terrain.center_position[1], 0.0)

    robot_position_sampler = CircularSampler(radius=0.6, z_range=(0.55, 0.55))
    robot_quat_sampler = CircularOrientationSampler(
        x_range=(-math.pi/18, math.pi/18),
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
    
    ############################
    ## Reward configurations ###
    ############################
    
    # reward
    reward_parameter: VelocityTrackingReward = VelocityTrackingReward(height_similarity_weight=0.66, 
                                                            lin_vel_similarity_weight=0.66,
                                                            ang_vel_similarity_weight=0.66,
                                                            height_similarity_coeff=4.0, 
                                                            lin_vel_similarity_coeff=4.0,
                                                            ang_vel_similarity_coeff=4.0,
                                                            # height_similarity_coeff=0.5, 
                                                            # lin_vel_similarity_coeff=0.5,
                                                            # ang_vel_similarity_coeff=0.5,
                                                            # height_reward_mode="gaussian",
                                                            # lin_vel_reward_mode="gaussian",
                                                            # ang_vel_reward_mode="gaussian"
                                                            height_reward_mode="exponential",
                                                            lin_vel_reward_mode="exponential",
                                                            ang_vel_reward_mode="exponential"
                                                            )
    pose_tracking_reward_parameter: PoseTrackingReward = PoseTrackingReward(
        position_weight=0.0, # disable 
        yaw_weight=0.0, # disable
        position_coeff=1.0, 
        yaw_coeff=1.0,
        # position_reward_mode="gaussian", 
        # yaw_reward_mode="gaussian"
        position_reward_mode="exponential", 
        yaw_reward_mode="exponential"
        )
    alive_reward_parameter: AliveReward = AliveReward(alive_weight=0.66)
    swing_foot_tracking_reward_parameter: SwingFootTrackingReward = SwingFootTrackingReward(
        swing_foot_weight=0.0, # disable
        swing_foot_coeff=2.0,
        swing_foot_reward_mode="gaussian"
    )
    
    # penalty
    orientation_penalty_parameter: OrientationRegularizationPenalty = OrientationRegularizationPenalty(
        roll_penalty_weight=0.0, 
        pitch_penalty_weight=0.0,
        roll_range=(-math.pi/18, math.pi/18),
        pitch_range=(-math.pi/18, math.pi/18),
    )
    
    velocity_penalty_parameter: VelocityPenalty = VelocityPenalty(
        velocity_penalty_weight=0.66, 
    )
    angular_velocity_penalty_parameter: AngularVelocityPenalty = AngularVelocityPenalty(
        ang_velocity_penalty_weight=0.66,
    )
    
    action_penalty_parameter: CurriculumActionRegularizationPenalty = CurriculumActionRegularizationPenalty(
        action_penalty_weight_start=5e-4, 
        action_penalty_weight_end=5e-4,
        energy_penalty_weight_start=5e-4,
        energy_penalty_weight_end=5e-4, 
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps),
        )
    torque_penalty_parameter: CurriculumTorqueRegularizationPenalty = CurriculumTorqueRegularizationPenalty(
        torque_penalty_weight_start=1e-5,
        torque_penalty_weight_end=1e-5, 
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps),
    )
    
    foot_distance_penalty_parameter: FootDistanceRegularizationPenalty = FootDistanceRegularizationPenalty(foot_distance_penalty_weight=0.66, foot_distance_bound=(0.3, 0.5))
    foot_slide_penalty_parameter: FeetSlidePenalty = FeetSlidePenalty(feet_slide_weight=0.1)
    action_saturation_penalty_parameter: ActionSaturationPenalty = ActionSaturationPenalty(action_penalty_weight=0.66, action_bound=(0.9, 1.0))


@configclass
class HierarchicalArchPrimeCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 60
    state_space = 60
    action_space = 6
    
    # action bound hyper parameter
    action_lb = [-6.0, -6.0, -6.0] + [-0.2, -2.0, -2.0]
    action_ub = [6.0,  6.0, 6.0] + [0.2, 2.0, 2.0]

@configclass
class HierarchicalArchPrimeFullCfg(HierarchicalArchCfg):
    episode_length_s =20
    num_history = 1
    
    observation_space = 72
    state_space = 72
    action_space = 18

    # action bound hyper parameter
    lin_accel_scale = [6.0, 6.0, 6.0]
    ang_accel_scale = [0.2, 2.0, 2.0]

    # +- 10% uncertainty of original mass and inertia
    inv_mass_scale_lb = [-0.1/13.856, -0.1/13.856, -0.1/13.856]
    inv_mass_scale_ub = [0.0/13.856, 0.0/13.856, 0.0/13.856]
    inv_inertia_scale = [0.1/0.5413, 0.1/0.52, 0.1/0.0691]

    action_lb = [-v for v in lin_accel_scale] + [-v for v in ang_accel_scale] + \
        [v for v in inv_mass_scale_lb] + [v for v in inv_mass_scale_lb] + \
        [-v for v in inv_inertia_scale] + [-v for v in inv_inertia_scale]
    action_ub = lin_accel_scale + ang_accel_scale + inv_mass_scale_ub + inv_mass_scale_ub + \
        inv_inertia_scale + inv_inertia_scale

@configclass
class HierarchicalArchAccelPFCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 64
    state_space = 64
    action_space = 10
    
    # action bound hyper parameter
    action_lb = [-6.0, -6.0, -6.0] + [-0.2, -2.0, -2.0] + [-0.02, -0.02, -0.01, -0.01]
    action_ub = [6.0,  6.0, 6.0] + [0.2, 2.0, 2.0] + [0.1, 0.1, 0.01, 0.01]