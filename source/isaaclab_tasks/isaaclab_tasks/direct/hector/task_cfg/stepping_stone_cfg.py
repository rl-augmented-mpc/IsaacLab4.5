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
from isaaclab_tasks.direct.hector.common.sampler import CircularSamplerWithLimit, BinaryOrientationSampler
from isaaclab_tasks.direct.hector.common.curriculum import  CurriculumRateSampler, CurriculumLineSampler, CurriculumUniformLineSampler, \
    CurriculumUniformCubicSampler, CurriculumQuaternionSampler, PerformanceCurriculumLineSampler
from isaaclab_tasks.direct.hector.core_cfg import terrain_cfg

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"



@configclass
class SteppingStoneCfg(HierarchicalArchCfg):
    episode_length_s = 10
    seed = 42
    num_steps_per_env = 32
    inference = False
    curriculum_inference = False

    clip_action = False
    scale_action = True

    # ========================
    # Observation/Action space
    # ========================
    dt=1/500 # physics dt
    policy_dt = 0.01 # RL policy dt
    traj_sample = int(policy_dt/dt)
    decimation = int(policy_dt/dt)

    action_space = traj_sample*3
    observation_space = 70+action_space
    state_space = 70+action_space
    num_history = 1
    num_extero_observations = int((1.0/0.05 + 1)*(1.0/0.05 + 1))

    action_lb = [-0.5]*traj_sample + [-0.03]*traj_sample + [-0.5]*traj_sample
    action_ub = [0.5]*traj_sample + [0.15]*traj_sample + [0.5]*traj_sample


    # ================================
    # Environment configurations
    # ================================
    # terrain = terrain_cfg.SteppingStoneTerrain
    terrain = terrain_cfg.CurriculumSteppingStoneTerrain

    # ================================
    # Task configurations
    # ================================

    # termination conditions
    roll_limit = (30/180)*math.pi
    pitch_limit = (30/180)*math.pi
    min_height = 0.55-0.3
    max_height = 0.55+0.3

    # terrain curriculum
    terrain_curriculum_sampler = PerformanceCurriculumLineSampler(
        x_start=0, x_end=terrain.num_curriculums-1,
        num_curriculums=terrain.num_curriculums,
        update_frequency=5,
        maximum_episode_length=int(episode_length_s/(dt*decimation)),
        ratio=0.9
    )
    
    # gait parameters
    robot_nominal_foot_height_sampler = CurriculumUniformLineSampler(
        x_range_start=(0.15, 0.15),
        x_range_end=(0.15, 0.15),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1)
    )

    # robot spawner
    robot_position_sampler = CircularSamplerWithLimit(radius=2.0, z_range=(0.56, 0.56))
    robot_quat_sampler = BinaryOrientationSampler()
    robot_target_velocity_sampler = CurriculumUniformCubicSampler(
        x_range_start=(0.5, 0.7), x_range_end=(0.5, 0.7),
        y_range_start=(0.0, 0.0), y_range_end=(0.0, 0.0),
        z_range_start=(-0.0, 0.0), z_range_end=(-0.0, 0.0),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1)
    )
    
    
    # =====================
    # Reward configurations
    # =====================
    
    # reward
    reward_parameter: VelocityTrackingReward = VelocityTrackingReward(height_similarity_weight=0.3, 
                                                            lin_vel_similarity_weight=0.3,
                                                            ang_vel_similarity_weight=0.3,
                                                            height_similarity_coeff=0.5, 
                                                            lin_vel_similarity_coeff=0.5,
                                                            ang_vel_similarity_coeff=0.5,
                                                            height_reward_mode="gaussian",
                                                            lin_vel_reward_mode="gaussian",
                                                            ang_vel_reward_mode="gaussian"
                                                            )
    
    pose_tracking_reward_parameter: PoseTrackingReward = PoseTrackingReward(
        position_weight=0.0, # disable 
        yaw_weight=0.0, # disable
        position_coeff=0.5, 
        yaw_coeff=0.5,
        position_reward_mode="gaussian", 
        yaw_reward_mode="gaussian"
        )
    
    alive_reward_parameter: AliveReward = AliveReward(alive_weight=0.01)

    swing_foot_tracking_reward_parameter: SwingFootTrackingReward = SwingFootTrackingReward(
        swing_foot_weight=0.2, 
        swing_foot_coeff=0.5,
        swing_foot_reward_mode="gaussian"
    )
    
    # penalty
    orientation_penalty_parameter: OrientationRegularizationPenalty = OrientationRegularizationPenalty(
        roll_penalty_weight=0.2, 
        pitch_penalty_weight=0.2, 
        roll_range=(torch.pi/6, torch.pi/3), 
        pitch_range=(torch.pi/6, torch.pi/3))
    
    velocity_penalty_parameter: VelocityPenalty = VelocityPenalty(
        velocity_penalty_weight=1.0, 
    )

    angular_velocity_penalty_parameter: AngularVelocityPenalty = AngularVelocityPenalty(
        ang_velocity_penalty_weight=0.01,
    )
    
    foot_distance_penalty_parameter: FootDistanceRegularizationPenalty = FootDistanceRegularizationPenalty(
        foot_distance_penalty_weight=0.5, 
        foot_distance_bound=(0.3, 0.5))
    
    foot_slide_penalty_parameter: FeetSlidePenalty = FeetSlidePenalty(
        feet_slide_weight=0.2
        )

    toe_left_joint_penalty_parameter: JointPenalty = JointPenalty(
        joint_penalty_weight=2.0, 
        joint_pos_bound=(torch.pi/18, torch.pi/6),
    )
    toe_right_joint_penalty_parameter: JointPenalty = JointPenalty(
        joint_penalty_weight=2.0, 
        joint_pos_bound=(torch.pi/18, torch.pi/6),
    )
    
    action_penalty_parameter: CurriculumActionRegularizationPenalty = CurriculumActionRegularizationPenalty(
        action_penalty_weight_start=5e-4, 
        action_penalty_weight_end=5e-4,
        energy_penalty_weight_start=5e-4,
        energy_penalty_weight_end=5e-4, 
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1),
        )
    torque_penalty_parameter: CurriculumTorqueRegularizationPenalty = CurriculumTorqueRegularizationPenalty(
        torque_penalty_weight_start=1e-4,
        torque_penalty_weight_end=1e-4, 
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1),
    )
    action_saturation_penalty_parameter: ActionSaturationPenalty = ActionSaturationPenalty(action_penalty_weight=0.0, action_bound=(0.9, 1.0))