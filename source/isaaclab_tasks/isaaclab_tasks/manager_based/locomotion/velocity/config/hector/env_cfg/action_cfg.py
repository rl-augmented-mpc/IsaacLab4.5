# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp


@configclass
class HECTORBlindLocomotionActionsCfg:
    """Action specifications for the MDP."""

    # # 3C1
    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgDyn(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.2/13.856, -0.2/13.856, -0.2/13.856,    -0.2/0.5413, -0.2/0.52, -0.2/0.0691), 
    #         (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.2/13.856,  0.2/13.856,  0.2/13.856,     0.2/0.5413,  0.2/0.52,  0.2/0.0691)
    #     ), 
    #     negative_action_clip_idx=None,
    # )

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgSwing(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-0.15, -0.66), 
    #         (0.15, 0.66)
    #     ), 
    #     negative_action_clip_idx=[0],
    #     debug_vis=True,
    # )

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgGait(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-0.25), 
    #         (0.25)
    #     ), 
    #     negative_action_clip_idx=None,
    # )

    # 3C2

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgSwingGait(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-0.25, -0.15, -0.66), 
    #         (0.25, 0.15, 0.66)
    #     ), 
    #     negative_action_clip_idx=[1],
    # )

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgDynSwing(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.2/13.856, -0.2/13.856, -0.2/13.856,    -0.2/0.5413, -0.2/0.52, -0.2/0.0691,   -0.15, -0.66), 
    #         (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.2/13.856,  0.2/13.856,  0.2/13.856,     0.2/0.5413,  0.2/0.52,  0.2/0.0691,    0.15, 0.66)
    #     ), 
    #     negative_action_clip_idx=[12],
    # )

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgDynGait(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.2/13.856, -0.2/13.856, -0.2/13.856,    -0.2/0.5413, -0.2/0.52, -0.2/0.0691,   -0.25), 
    #         (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.2/13.856,  0.2/13.856,  0.2/13.856,     0.2/0.5413,  0.2/0.52,  0.2/0.0691,    0.25)
    #     ), 
    #     negative_action_clip_idx=None,
    # )

    # # 3C3

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgSimpleDynSwingGait(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.25, -0.15, -0.66), 
    #         (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.25, 0.15, 0.66)
    #     ), 
    #     negative_action_clip_idx=[7],
    # )

    
    mpc_action = hector_mdp.BlindLocomotionMPCActionCfgResAll(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.2/13.856, -0.2/13.856, -0.2/13.856,    -0.2/0.5413, -0.2/0.52, -0.2/0.0691,   -0.25, -0.15, -0.66), 
            (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.2/13.856,  0.2/13.856,  0.2/13.856,     0.2/0.5413,  0.2/0.52,  0.2/0.0691,    0.25, 0.15, 0.66)
        ), 
        negative_action_clip_idx=[13],
        debug_vis=True,
    )

@configclass
class HECTORSlipActionsCfg:
    """Action specifications for the MDP."""

   # 3C3

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfgSimpleDynSwingGait(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.3, -0.05, -0.33), 
    #         (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.3, 0.05, 0.33)
    #     ), 
    #     negative_action_clip_idx=None,
    # )

    
    mpc_action = hector_mdp.BlindLocomotionMPCActionCfgResAll(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.2/13.856, -0.2/13.856, -0.2/13.856,    -0.2/0.5413, -0.2/0.52, -0.2/0.0691,   -0.3, -0.05, -0.33), 
            (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.2/13.856,  0.2/13.856,  0.2/13.856,     0.2/0.5413,  0.2/0.52,  0.2/0.0691,    0.3, 0.05, 0.33)
        ), 
        # negative_action_clip_idx=None,
        debug_vis=True,
    )

@configclass
class HECTORPerceptiveLocomotionActionsCfg:
    """Action specifications for the MDP."""

    mpc_action = hector_mdp.PerceptiveLocomotionMPCActionCfg(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-0.25, -0.1, -0.33), 
            (0.25, 0.1, 0.33)
        )
    )

    # mpc_action = hector_mdp.PerceptiveLocomotionMPCActionCfg2(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0, -1.0, -1.0, -1.0, -0.25, -0.15, -0.66), # symmetric action space
    #         (2.0, 2.0, 4.0, 1.0, 1.0, 1.0, 0.25, 0.15, 0.66)
    #     ), 
    #     negative_action_clip_idx=[7],
    # )
    
    # mpc_action = hector_mdp.PerceptiveLocomotionMPCActionCfg3(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-0.3, 0.0, -0.0, -0.0, -0.3), 
    #         (0.3, 0.1, 0.0, 0.0, 0.3)
    #     )
    # )
    
    # mpc_action = hector_mdp.PerceptiveLocomotionMPCActionCfg4(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0, -0.1, -1.0, -1.0, -0.6, 0.0, -0.4, -1.5, -1.5), 
    #         (2.0, 2.0, 4.0, 0.1, 1.0, 1.0, 0.6, 0.1, 0.4, 0.5, 0.5)
    #     )
    # )

@configclass
class HECTORGPUBlindLocomotionActionsCfg:
    # mpc_action = hector_mdp.BlindLocomotionGPUMPCActionCfg(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-0.25, -0.15, -0.66), 
    #         (0.25, 0.15, 0.66)
    #     )
    # )

    mpc_action = hector_mdp.BlindLocomotionGPUMPCActionCfg2(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.25, -0.15, -0.66), 
            (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.25, 0.15, 0.66)
        ), 
        negative_action_clip_idx=[7],
    )

@configclass
class HECTORGPUSlipActionsCfg:
    """Action specifications for the MDP."""

    mpc_action = hector_mdp.BlindLocomotionGPUMPCActionCfg2(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,   -1.0, -1.0, -1.0,    -0.3, -0.05, -0.33), # symmetric action space
            (2.0, 2.0, 4.0,       1.0, 1.0, 1.0,       0.3, 0.05, 0.33)
        ), 
        negative_action_clip_idx=None,
    )
    
# @configclass
# class HECTORL2TActionsCfg:
#     """Action specifications for the MDP."""

#     mpc_action = hector_mdp.PerceptiveLocomotionMPCActionCfg(
#         asset_name="robot", 
#         joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
#         action_range = (
#             (-0.6, 0.0, -0.4), 
#             (0.6, 0.1, 0.4)
#         )
#     )
    
#     # mpc_action = hector_mdp.MPCActionCfgV4(
#     #     asset_name="robot", 
#     #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
#     #     action_range = (
#     #         (-2.0, -2.0, -4.0, -0.1, -1.0, -1.0, -0.6, 0.0, -0.4, -1.5, -1.5), 
#     #         (2.0, 2.0, 4.0, 0.1, 1.0, 1.0, 0.6, 0.1, 0.4, 0.5, 0.5)
#     #     )
#     # )


"""
E2E
"""

@configclass
class HECTORRLBlindLocomotionActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg( # type: ignore
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        scale={
            'L_hip_joint': 0.9 * math.pi/6, 
            'L_hip2_joint': 0.9 * (55.0/180.0) * math.pi,
            'L_thigh_joint': 0.9 * (45.0/180.0) * math.pi,
            'L_calf_joint': 0.9 * (90.0/180.0) * math.pi,
            'L_toe_joint': 0.9 * (55.0/180.0) * math.pi,
            'R_hip_joint': 0.9 * math.pi/6, 
            'R_hip2_joint': 0.9 * (55.0/180.0) * math.pi,
            'R_thigh_joint': 0.9 * (45.0/180.0) * math.pi,
            'R_calf_joint': 0.9 * (90.0/180.0) * math.pi,
            'R_toe_joint': 0.9 * (55.0/180.0) * math.pi,
            }, 
        use_default_offset=True, 
        preserve_order=True, 
        )
