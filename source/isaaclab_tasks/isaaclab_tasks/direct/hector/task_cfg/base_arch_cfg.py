# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import math
import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence
import carb

# IsaacLab core
from isaaclab.envs.common import ViewerCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

# Task core
from isaaclab_assets.hector import HECTOR_CFG
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from isaaclab_tasks.direct.hector.common.task_reward import VelocityTrackingReward, AliveReward, ContactTrackingReward, PoseTrackingReward
from isaaclab_tasks.direct.hector.common.task_penalty import VelocityTrackingPenalty, TwistPenalty, FeetSlidePenalty, JointPenalty, ActionSaturationPenalty
from isaaclab_tasks.direct.hector.common.sampler import UniformLineSampler, UniforPlaneSampler, UniformCubicSampler, QuaternionSampler
from isaaclab_tasks.direct.hector.common.curriculum import CurriculumRateSampler, CurriculumUniformCubicSampler, CurriculumQuaternionSampler
from isaaclab_tasks.direct.hector.core_cfg.terrain_cfg import ThickPatchTerrain

# Macros
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class BaseArchCfg(DirectRLEnvCfg):
    # ================================
    # Common configurations
    # ================================
    seed = 42
    episode_length_s = 10.0
    num_steps_per_env = 24 # see rsl cfg (horizon for rollout)
    
    dt=0.002 #500Hz 
    rendering_interval = 10 # 50Hz
    mpc_decimation = 5 # 100Hz
    decimation = 5 # 100Hz (RL)
    
    # TODO: override this in the derived class
    observation_space = MISSING # actor observation space
    state_space = MISSING # critic observation space
    action_space = MISSING # action space
    num_joint_actions:int = None
    num_states:int = None
    
    # TODO: override this in the derived class
    # observation, action settings
    action_lb:list[float] = None
    action_ub:list[float] = None
    observation_lb:float = None
    observation_ub:float = None
    clip_action:bool = True # clip to -1 to 1 with tanh
    scale_action:bool = True # scale max value to action_ub
    
    # MPC parameters
    gait_change_cutoff = 200
    ssp_duration = int(0.2/dt) # single support 0.2s
    dsp_duration = int(0.0/dt) # double support
    iteration_between_mpc = 10 # mpc time step discritization (dt_mpc = dt*iteration_between_mpc)
    horizon_length = 10
    reference_height = 0.55
    nominal_gait_stepping_frequency = 1.0
    nominal_foot_height = 0.12
    
    # ================================
    # Environment configurations
    # ================================
    
    # simulation cfg
    # Not sure what physics_material is for
    sim: SimulationCfg = SimulationCfg(
        dt=dt, 
        render_interval=rendering_interval, 
        gravity=(0, 0, -9.81), 
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=static_friction,
        #     dynamic_friction=dynamic_friction,
        # ),
        disable_contact_processing=True, 
        )
    
    # terrain
    terrain = ThickPatchTerrain
    
    # robot
    robot: ArticulationCfg = HECTOR_CFG
    robot.prim_path = f"{ENV_REGEX_NS}/Robot"
    
    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Robot/[L|R]_toe", 
        # filter_prim_paths_expr = ["/World/ground"],
        history_length=3, 
        update_period=dt*decimation, 
        track_air_time=False
    )
    contact_sensor.visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ContactSensor",
            markers={
                "contact": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/arrow_z.usd",
                    scale=(0.5, 0.05, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                "no_contact": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/arrow_z.usd",
                    scale=(0.5, 0.05, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), 
                    visible=False,
                ),
            },
        )
    
    # LiDAR
    ray_caster = RayCasterCfg(
        prim_path=f"{ENV_REGEX_NS}/Robot/base",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0)),
        mesh_prim_paths=["/World/ground"],
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
        ),
        max_distance=100,
    )
    
    # light
    light_type = "distant"
    
    if light_type == "dome":
        light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=1800.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
    elif light_type == "distant":
        light = AssetBaseCfg(
            prim_path="/World/distantLight",
            spawn=sim_utils.DistantLightCfg(
                intensity=3000.0,
            )
        )
    
    # scene 
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=1.5,
        )
    
    # ================================
    # Task configurations
    # ================================
    
    # termination conditions
    roll_limit = math.pi/3
    pitch_limit = math.pi/4
    min_height = reference_height-0.2 #-0.15m from nominal height
    max_height = reference_height+0.2 #+0.15m from nominal height
    
    # robot spawner
    center = (terrain.center_position[0], terrain.center_position[1], 0.0)
    max_x = 2.0
    max_y = 2.0
    robot_position_sampler = UniformCubicSampler(
        x_range=(center[0]-max_x, center[0]+max_x), 
        y_range=(center[1]-max_y, center[1]+max_y), 
        z_range=(0.55, 0.55))
    robot_quat_sampler = CurriculumQuaternionSampler(
        x_range_start=(0.0, math.pi/3), x_range_end=(0.0, 2*math.pi),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=24*10000)
    )
    robot_target_velocity_sampler = CurriculumUniformCubicSampler(
        x_range_start=(0.0, 0.0), x_range_end=(0.0, 0.0),
        y_range_start=(0.0, 0.0), y_range_end=(0.0, 0.0),
        z_range_start=(0.0, 0.0), z_range_end=(0.0, 0.0),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=24*10000)
    )
    
    # reward parameters
    reward_parameter: VelocityTrackingReward = VelocityTrackingReward(height_similarity_weight=0.66, 
                                                            lin_vel_similarity_weight=0.66,
                                                            ang_vel_similarity_weight=0.66,
                                                            height_reward_mode="exponential",
                                                            lin_vel_reward_mode="exponential",
                                                            ang_vel_reward_mode="exponential")
    pose_tracking_reward_parameter: PoseTrackingReward = PoseTrackingReward(position_weight=0.66, yaw_weight=0.66, position_reward_mode="exponential", yaw_reward_mode="exponential")
    alive_reward_parameter: AliveReward = AliveReward(alive_weight=1.0)
    contact_tracking_reward_parameter: ContactTrackingReward = ContactTrackingReward(contact_similarity_weight=0.0, contact_reward_mode="square")
    
    # penalty parameters
    penalty_parameter: VelocityTrackingPenalty = VelocityTrackingPenalty(roll_deviation_weight=0.33, 
                                                               pitch_deviation_weight=0.33, 
                                                               action_penalty_weight=0.03, 
                                                               energy_penalty_weight=0.03,
                                                               foot_energy_penalty_weight=0.00)
    twist_penalty_parameter: TwistPenalty = TwistPenalty(vx_bound=(0.3, 0.5), 
                                                         vy_bound=(0.3, 0.5), 
                                                         wz_bound=(0.5, 1.0), 
                                                         vx_penalty_weight=0.1,
                                                         vy_penalty_weight=0.1,
                                                         wz_penalty_weight=0.1)
    foot_slide_penalty_parameter: FeetSlidePenalty = FeetSlidePenalty(feet_slide_weight=0.1)
    left_hip_roll_joint_penalty_parameter: JointPenalty = JointPenalty(joint_penalty_weight=0.1, joint_pos_bound=(torch.pi/10, torch.pi/6))
    right_hip_roll_joint_penalty_parameter: JointPenalty = JointPenalty(joint_penalty_weight=0.1, joint_pos_bound=(torch.pi/10, torch.pi/6))
    hip_pitch_joint_deviation_penalty_parameter: JointPenalty = JointPenalty(joint_penalty_weight=0.1, joint_pos_bound=(torch.pi/6, torch.pi/2))
    action_saturation_penalty_parameter: ActionSaturationPenalty = ActionSaturationPenalty(action_penalty_weight=0.2, action_bound=(0.9, 1.0))
    
    
    # joint orders
    joint_names = ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint',
            'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']
    body_names = ['base', 'trunk', 'L_hip', 'R_hip', 
                  'imu_link', 'L_hip2', 'L_hip_trans', 'R_hip2', 'R_hip_trans', 
                  'L_hip2_trans', 'L_thigh', 'R_hip2_trans', 'R_thigh', 
                  'L_calf', 'L_thigh1_trans', 'L_thigh2_trans', 
                  'R_calf', 'R_thigh1_trans', 'R_thigh2_trans', 
                  'L_toe', 'R_toe', 
                  'L_foot_elem_00', 'L_foot_elem_01', 'L_foot_elem_02', 'L_foot_elem_03', 'L_foot_elem_04', 'L_foot_elem_05', 
                  'R_foot_elem_00', 'R_foot_elem_01', 'R_foot_elem_02', 'R_foot_elem_03', 'R_foot_elem_04', 'R_foot_elem_05']
    foot_patch_num = 2
    
    # simulation viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=center, 
        lookat=center,
        resolution=(1920, 1080)
        )