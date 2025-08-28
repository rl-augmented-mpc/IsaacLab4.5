# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp


@configclass
class HECTORBlindLocomotionActionsCfg:
    """Action specifications for the MDP."""

    # mpc_action = hector_mdp.BlindLocomotionMPCActionCfg(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-0.6, 0.0, -0.6), 
    #         (0.6, 0.1, 0.6)
    #     )
    # )

    mpc_action = hector_mdp.BlindLocomotionMPCActionCfg2(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.25, -0.15, -0.66), 
            (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.25, 0.15, 0.66)
        ), 
        negative_action_clip_idx=[7],
    )

@configclass
class HECTORSlipActionsCfg:
    """Action specifications for the MDP."""

    mpc_action = hector_mdp.BlindLocomotionMPCActionCfg2(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,   -1.0, -1.0, -1.0,    -0.3, -0.05, -0.33), # symmetric action space
            (2.0, 2.0, 4.0,       1.0, 1.0, 1.0,       0.3, 0.05, 0.33)
        ), 
        # negative_action_clip_idx=[7],
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
class HECTORTorchBlindLocomotionActionsCfg:
    mpc_action = hector_mdp.BlindLocomotionTorchMPCActionCfg(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-0.25, -0.15, -0.66), 
            (0.25, 0.15, 0.66)
        )
    )
    
@configclass
class HECTORL2TActionsCfg:
    """Action specifications for the MDP."""

    mpc_action = hector_mdp.PerceptiveLocomotionMPCActionCfg(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-0.6, 0.0, -0.4), 
            (0.6, 0.1, 0.4)
        )
    )
    
    # mpc_action = hector_mdp.MPCActionCfgV4(
    #     asset_name="robot", 
    #     joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
    #     action_range = (
    #         (-2.0, -2.0, -4.0, -0.1, -1.0, -1.0, -0.6, 0.0, -0.4, -1.5, -1.5), 
    #         (2.0, 2.0, 4.0, 0.1, 1.0, 1.0, 0.6, 0.1, 0.4, 0.5, 0.5)
    #     )
    # )