# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Digit implementation made by LIDAR Gatech

"""Configuration for Agility robots.
The following configurations are available:
* :obj:`DIGITV4_CFG`: Agility Cassie robot with simple PD controller for the legs
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

import os
import torch

torch.cuda.empty_cache()

full_path = os.path.dirname(os.path.realpath(__file__))


HECTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_oct8.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_flat_foot_bounding_cube.usd", # flat foot
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_flat_foot_small_bounding_cube.usd", # flat foot without cover
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_flat_foot_small_bounding_cube_toe_sensor.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_flat_foot_small_bounding_cube_toe_sensor_convex_hull.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_original_foot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_vel={".*": 0.0},
        joint_pos={
            'L_hip_joint': 0.0,
            'L_hip2_joint': 0.0,
            'L_thigh_joint': 0.0,
            'L_calf_joint': 0.0,
            'L_toe_joint': 0.0,
            'R_hip_joint': 0.0,
            'R_hip2_joint': 0.0,
            'R_thigh_joint': 0.0,
            'R_calf_joint': 0.0,
            'R_toe_joint': 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_hip_joint",
                "L_hip2_joint",
                "L_thigh_joint",
                "L_calf_joint",
                "R_hip_joint",
                "R_hip2_joint",
                "R_thigh_joint",
                "R_calf_joint",
            ],
            effort_limit=67.0,
            velocity_limit=30.0,
            stiffness={
                "L_hip_joint": 0,
                "L_hip2_joint": 0,
                "L_thigh_joint": 0,
                "L_calf_joint": 0,
                "R_hip_joint": 0,
                "R_hip2_joint": 0,
                "R_thigh_joint": 0,
                "R_calf_joint": 0,
            },
            damping={
                "L_hip_joint": 0.0,
                "L_hip2_joint": 0.0,
                "L_thigh_joint": 0.0,
                "L_calf_joint": 0.0,
                "R_hip_joint": 0.0,
                "R_hip2_joint": 0.0,
                "R_thigh_joint": 0.0,
                "R_calf_joint": 0.0,
            },
            armature={
                "L_hip_joint": 0,
                "L_hip2_joint": 0,
                "L_thigh_joint": 0,
                "L_calf_joint": 0,
                "R_hip_joint": 0,
                "R_hip2_joint": 0,
                "R_thigh_joint": 0,
                "R_calf_joint": 0,
            },
            # min_delay=0,  # physics time steps (min: 1.0*0=0.0ms)
            # max_delay=8,  # physics time steps (max: 1.0*8=8.0ms)
        ),
        "toes": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_toe_joint",
                "R_toe_joint",
            ],
            effort_limit=33.5,
            velocity_limit=30.0,
            stiffness={
                "L_toe_joint": 0,
                "R_toe_joint": 0,
            },
            damping={
                "L_toe_joint": 0.0,
                "R_toe_joint": 0.0,
            },
            armature={
                "L_toe_joint": 0.000,
                "R_toe_joint": 0.000,
            },
        ),
    },
)


HECTOR_E2E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_oct8.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_flat_foot_bounding_cube.usd", # flat foot
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_flat_foot_small_bounding_cube.usd", # flat foot without cover
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_vel={".*": 0.0},
        joint_pos={
            'L_hip_joint': 0.0,
            'L_hip2_joint': 0.0,
            'L_thigh_joint': 0.0,
            'L_calf_joint': 0.0,
            'L_toe_joint': 0.0,
            'R_hip_joint': 0.0,
            'R_hip2_joint': 0.0,
            'R_thigh_joint': 0.0,
            'R_calf_joint': 0.0,
            'R_toe_joint': 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_hip_joint",
                "L_hip2_joint",
                "L_thigh_joint",
                "L_calf_joint",
                "R_hip_joint",
                "R_hip2_joint",
                "R_thigh_joint",
                "R_calf_joint",
            ],
            effort_limit=67.0,
            velocity_limit=30.0,
            stiffness={
                "L_hip_joint": 10,
                "L_hip2_joint": 20,
                "L_thigh_joint": 20,
                "L_calf_joint": 20,
                "R_hip_joint": 10,
                "R_hip2_joint": 20,
                "R_thigh_joint": 20,
                "R_calf_joint": 20,
            },
            damping={
                "L_hip_joint": 0.45,
                "L_hip2_joint": 0.6,
                "L_thigh_joint": 0.45,
                "L_calf_joint": 1.0,
                "R_hip_joint": 0.45,
                "R_hip2_joint": 0.6,
                "R_thigh_joint": 0.45,
                "R_calf_joint": 1.0,
            },
            armature={
                "L_hip_joint": 0,
                "L_hip2_joint": 0,
                "L_thigh_joint": 0,
                "L_calf_joint": 0,
                "R_hip_joint": 0,
                "R_hip2_joint": 0,
                "R_thigh_joint": 0,
                "R_calf_joint": 0,
            },
            # min_delay=0,  # physics time steps (min: 1.0*0=0.0ms)
            # max_delay=8,  # physics time steps (max: 1.0*8=8.0ms)
        ),
        "toes": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_toe_joint",
                "R_toe_joint",
            ],
            effort_limit=33.5,
            velocity_limit=30.0,
            stiffness={
                "L_toe_joint": 10.0,
                "R_toe_joint": 10.0,
            },
            damping={
                "L_toe_joint": 0.6,
                "R_toe_joint": 0.6,
            },
            armature={
                "L_toe_joint": 0.000,
                "R_toe_joint": 0.000,
            },
        ),
    },
)