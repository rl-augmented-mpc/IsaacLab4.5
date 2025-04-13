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
    CurriculumUniformCubicSampler, CurriculumQuaternionSampler
from isaaclab_tasks.direct.hector.core_cfg import terrain_cfg

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"



@configclass
class SteppingStoneCfg(HierarchicalArchCfg):
    episode_length_s =20
    seed = 42
    num_steps_per_env = 32
    inference = False
    curriculum_inference = False

    # ========================
    # Observation/Action space
    # ========================
    dt=1/500 # physics dt
    policy_dt = 0.01 # RL policy dt
    traj_sample = int(policy_dt/dt)
    decimation = int(policy_dt/dt)

    action_space = traj_sample*2
    observation_space = 70+action_space
    state_space = 70+action_space
    num_history = 3
    num_extero_observations = int((1.0/0.05 + 1)*(1.0/0.05 + 1))

    action_lb = [-0.5]*traj_sample + [-0.01]*traj_sample
    action_ub = [0.5]*traj_sample + [0.15]*traj_sample


    # ================================
    # Environment configurations
    # ================================
    terrain = terrain_cfg.SteppingTerrain

    # ================================
    # Task configurations
    # ================================

    # termination conditions
    roll_limit = (60/180)*math.pi
    pitch_limit = (60/180)*math.pi
    min_height = 0.55-0.4
    max_height = 0.55+0.4
    
    # gait parameters
    robot_nominal_foot_height_sampler = CurriculumUniformLineSampler(
        x_range_start=(0.12, 0.12),
        x_range_end=(0.12, 0.12),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1)
    )

    # robot spawner
    robot_position_sampler = CircularSamplerWithLimit(radius=1.5, z_range=(0.56, 0.56))
    robot_quat_sampler = BinaryOrientationSampler()
    terrain_curriculum_sampler = CurriculumLineSampler(
        x_start=0, x_end=terrain.num_curriculums-1,
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1)
    )
    robot_target_velocity_sampler = CurriculumUniformCubicSampler(
        x_range_start=(0.3, 0.5), x_range_end=(0.3, 0.5),
        y_range_start=(0.0, 0.0), y_range_end=(0.0, 0.0),
        z_range_start=(-0.0, 0.0), z_range_end=(-0.0, 0.0),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=1)
    )
    
    
    # =====================
    # Reward configurations
    # =====================
    
    # reward
    reward_parameter: VelocityTrackingReward = VelocityTrackingReward(height_similarity_weight=0.33, 
                                                            lin_vel_similarity_weight=0.33,
                                                            ang_vel_similarity_weight=0.33,
                                                            height_similarity_coeff=4.0, 
                                                            lin_vel_similarity_coeff=4.0,
                                                            ang_vel_similarity_coeff=4.0,
                                                            height_reward_mode="gaussian",
                                                            lin_vel_reward_mode="gaussian",
                                                            ang_vel_reward_mode="gaussian"
                                                            )
    pose_tracking_reward_parameter: PoseTrackingReward = PoseTrackingReward(
        position_weight=0.0, # disable 
        yaw_weight=0.0, # disable
        position_coeff=1.0, 
        yaw_coeff=1.0,
        position_reward_mode="gaussian", 
        yaw_reward_mode="gaussian"
        )
    alive_reward_parameter: AliveReward = AliveReward(alive_weight=0.66)
    swing_foot_tracking_reward_parameter: SwingFootTrackingReward = SwingFootTrackingReward(
        swing_foot_weight=0.33, 
        swing_foot_coeff=4.0,
        swing_foot_reward_mode="gaussian"
    )
    
    # penalty
    orientation_penalty_parameter: OrientationRegularizationPenalty = OrientationRegularizationPenalty(
        roll_penalty_weight=0.33, 
        pitch_penalty_weight=0.33, 
        roll_range=(torch.pi/6, torch.pi/3), 
        pitch_range=(torch.pi/6, torch.pi/3))
    
    velocity_penalty_parameter: VelocityPenalty = VelocityPenalty(
        velocity_penalty_weight=2.0, 
    )
    angular_velocity_penalty_parameter: AngularVelocityPenalty = AngularVelocityPenalty(
        ang_velocity_penalty_weight=0.01,
    )
    
    foot_distance_penalty_parameter: FootDistanceRegularizationPenalty = FootDistanceRegularizationPenalty(foot_distance_penalty_weight=0.5, foot_distance_bound=(0.3, 0.5))
    foot_slide_penalty_parameter: FeetSlidePenalty = FeetSlidePenalty(
        feet_slide_weight=0.1
        )

    toe_left_joint_penalty_parameter: JointPenalty = JointPenalty(
        joint_penalty_weight=1.0, 
        joint_pos_bound=(torch.pi/18, torch.pi/6),
    )
    toe_right_joint_penalty_parameter: JointPenalty = JointPenalty(
        joint_penalty_weight=1.0, 
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