# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp


@configclass 
class HECTOREventCfg(EventCfg):
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = None

    # reset
    base_external_force_torque = None
    
    # reset terrain type
    reset_terrain_type = EventTerm(
        func=hector_mdp.reset_terrain_type, # type: ignore
        mode="reset",
    )
    
    # random initial noise added to default state defined in articulation cfg
    reset_base = EventTerm(
        # func=hector_mdp.reset_root_state_orthogonal, 
        func=mdp.reset_root_state_uniform, # type: ignore
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-5.0, -5.0), 
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),  
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, # type: ignore
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = None
    
    # reset camera
    # reset_camera = EventTerm(
    #     func=hector_mdp.reset_camera, # type: ignore
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    
    # # reset particle mass
    # reset_particle_mass = EventTerm(
    #     func=hector_mdp.reset_particle_mass, # type: ignore
    #     mode="reset",
    #     params={
    #         "mass_range": {
    #             "x": (1.0, 3.0)
    #             },
    #         "asset_cfg": SceneEntityCfg("gravel"),
    #     },
    # )
    
@configclass 
class HECTORSlipEventCfg(HECTOREventCfg):
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, # type: ignore
        mode="reset",
        params={
            # "pose_range": {
            #     "x": (-3.0, 3.0), 
            #     "y": (-3.0, 3.0),
            #     "z": (0.0, 0.0),
            #     "roll": (0.0, 0.0),
            #     "pitch": (0.0, 0.0),
            #     "yaw": (-math.pi, math.pi),
            #     },
            "pose_range": {
                "x": (-0.0, 0.0), 
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0, 0),
                },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),  
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )
    
    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity, # type: ignore
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )
    # push_robot = EventTerm(
    #     func=hector_mdp.apply_tangential_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(2.0, 5.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "force_range": (-15.0, 15.0), # max 1m/s^2
    #         "torque_range": (0.0, 0.0), 
    #         },
    # )