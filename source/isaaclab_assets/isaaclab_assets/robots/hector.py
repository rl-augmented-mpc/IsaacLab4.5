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
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robot/hector/hector_oct8.usd", #joint friction=0
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robot/hector/hector_flat_foot.usd", #joint friction=0
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
        # joint_pos={
        #     'L_hip_joint': 0.0,
        #     'L_hip2_joint': 0.0,
        #     'L_thigh_joint': 3.14*0.25,
        #     'L_calf_joint': -3.14*0.5,
        #     'L_toe_joint': 3.14*0.25,
        #     'R_hip_joint': 0.0,
        #     'R_hip2_joint': 0.0,
        #     'R_thigh_joint': 3.14*0.25,
        #     'R_calf_joint': -3.14*0.5,
        #     'R_toe_joint': 3.14*0.25,
        # },
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
        "feet": IdealPDActuatorCfg(
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
                "L_hip_joint": 0.1,
                "L_hip2_joint": 0.1,
                "L_thigh_joint": 0.1,
                "L_calf_joint": 0.1,
                "R_hip_joint": 0.1,
                "R_hip2_joint": 0.1,
                "R_thigh_joint": 0.1,
                "R_calf_joint": 0.1,
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
        "toes": IdealPDActuatorCfg(
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
                "L_toe_joint": 0.1,
                "R_toe_joint": 0.1,
            },
            armature={
                "L_toe_joint": 0.000,
                "R_toe_joint": 0.000,
            },
        ),
    },
)


# HECTOR_ACTIVE_FOOT_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/hector_extra_foot.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.60),
#         rot=(1.0, 0.0, 0.0, 0.0),
#         joint_vel={".*": 0.0},
#         joint_pos={
#             'L_hip_joint': 0.0,
#             'L_hip2_joint': 0.0,
#             'L_thigh_joint': 0.0,
#             'L_calf_joint': 0.0,
#             'L_toe_joint': 0.0,
#             'R_hip_joint': 0.0,
#             'R_hip2_joint': 0.0,
#             'R_thigh_joint': 0.0,
#             'R_calf_joint': 0.0,
#             'R_toe_joint': 0.0,
#         },
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "feet": IdealPDActuatorCfg(
#             joint_names_expr=[
#                 "L_hip_joint",
#                 "L_hip2_joint",
#                 "L_thigh_joint",
#                 "L_calf_joint",
#                 "R_hip_joint",
#                 "R_hip2_joint",
#                 "R_thigh_joint",
#                 "R_calf_joint",
#             ],
#             effort_limit=200.0,
#             velocity_limit=30.0,
#             stiffness={
#                 "L_hip_joint": 0,
#                 "L_hip2_joint": 0,
#                 "L_thigh_joint": 0,
#                 "L_calf_joint": 0,
#                 "R_hip_joint": 0,
#                 "R_hip2_joint": 0,
#                 "R_thigh_joint": 0,
#                 "R_calf_joint": 0,
#             },
#             damping={
#                 "L_hip_joint": 0.1,
#                 "L_hip2_joint": 0.1,
#                 "L_thigh_joint": 0.1,
#                 "L_calf_joint": 0.1,
#                 "R_hip_joint": 0.1,
#                 "R_hip2_joint": 0.1,
#                 "R_thigh_joint": 0.1,
#                 "R_calf_joint": 0.1,
#             },
#             armature={
#                 "L_hip_joint": 0,
#                 "L_hip2_joint": 0,
#                 "L_thigh_joint": 0,
#                 "L_calf_joint": 0,
#                 "R_hip_joint": 0,
#                 "R_hip2_joint": 0,
#                 "R_thigh_joint": 0,
#                 "R_calf_joint": 0,
#             },
#             # min_delay=0,  # physics time steps (min: 1.0*0=0.0ms)
#             # max_delay=8,  # physics time steps (max: 1.0*8=8.0ms)
#         ),
#         "toes": IdealPDActuatorCfg(
#             joint_names_expr=[
#                 "L_toe_joint",
#                 "R_toe_joint",
#             ],
#             effort_limit=200.0,
#             velocity_limit=30.0,
#             stiffness={
#                 "L_toe_joint": 0,
#                 "R_toe_joint": 0,
#             },
#             damping={
#                 "L_toe_joint": 0.1,
#                 "R_toe_joint": 0.1,
#             },
#             armature={
#                 "L_toe_joint": 0.000,
#                 "R_toe_joint": 0.000,
#             },
#         ),
#     },
# )