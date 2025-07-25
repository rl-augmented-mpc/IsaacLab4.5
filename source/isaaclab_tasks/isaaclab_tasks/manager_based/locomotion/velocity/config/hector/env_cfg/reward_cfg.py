# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

@configclass
class HECTORBlindLocomotionRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""
    # -- rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        # weight=0.1,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        # weight=0.1, 
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5}, 
    )
    track_height_exp = RewTerm(
        func=hector_mdp.track_torso_height_exp, 
        weight=0.1,
        params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_sole"),
                "std": 0.1,
                "reference_height": 0.55,
                },
    )

    # -- penalties
    # termination = RewTerm(func=mdp.is_terminated, weight=-200.0) # type: ignore
    termination = None
    # lin_vel_y_l2 = RewTerm(func=hector_mdp.lin_vel_y_l2, weight=-0.5) # type: ignore
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) # type: ignore
    # lin_accel_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}) # type: ignore
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, # type: ignore
        weight=-0.015, 
        )
    
    # -- joint penalties
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,  # type: ignore
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
    )
    dof_acc_l2 = None
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,  # type: ignore
        weight=-1.0e-5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
        )
    
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_toe_joint"])},
    )
    # joint_deviation = None

    # energy_penalty_l2 = RewTerm(
    #     func=hector_mdp.energy_penalty_l2, # type: ignore
    #     weight=-0.02,
    #     params={
    #         "assymetric_indices": [7], 
    #         "action_name": "mpc_action",
    #     }
    # )
    
    energy_penalty_l2 = RewTerm(
        func=hector_mdp.terrain_dependent_energy_penalty_l2, # type: ignore
        weight=-0.005,
        params={
            "assymetric_indices": [7], 
            "action_name": "mpc_action",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "lookahead_distance": 0.3,
            "lookback_distance": 0.0, 
            "patch_width": 0.15,
        }
    )
    
    # body-leg angle penalties
    leg_body_angle_l2 = RewTerm(
        func=hector_mdp.leg_body_angle_l2,
        weight=-1.0,
        params={"action_name": "mpc_action"}
    )
    leg_body_distance_l2 = RewTerm(
        func=hector_mdp.leg_distance_l2,
        # weight=-0.2,
        weight=-0.5,
        params={"action_name": "mpc_action"}
    )

    # contact penalties
    undesired_contacts_knee = RewTerm(
        func=mdp.undesired_contacts, # type: ignore
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )
    
    # -- MPC cost
    mpc_cost_l2 = RewTerm(
        func=hector_mdp.mpc_cost_l1, # type: ignore
        weight=-1e-4,
        params={
            "action_name": "mpc_action",
        },
        )
    
    dof_pos_limits = None
    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = None


@configclass
class HECTORPerceptiveLocomotionRewardsCfg(HECTORBlindLocomotionRewardsCfg):

    termination = None
    leg_body_angle_l2 = None

    # -- energy penalty
    energy_penalty_l2 = RewTerm(
        func=hector_mdp.terrain_dependent_energy_penalty_l2, # type: ignore
        weight=-0.01,
        params={
            "assymetric_indices": [1],
            "action_name": "mpc_action",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "lookahead_distance": 0.3,
            "lookback_distance": 0.0, 
            "patch_width": 0.15,
        }
    )

    # -- contact penalty
    undesired_contacts_toe = RewTerm(
        func=mdp.undesired_contacts, # type: ignore
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("toe_contact", body_names=".*_toe_tip"), "threshold": 1.0},
    )

    # foot_landing_penalty = RewTerm(
    #     func=hector_mdp.log_barrier_swing_foot_landing_penalty,
    #     weight=-2.0,
    #     params={
    #         "action_name": "mpc_action",
    #         "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
    #         "l_toe": 0.07,  # ankle to toe
    #         "l_heel": 0.04,  # ankle to heel
    #         "l_width": 0.03,  # width of foot
    #     },
    # )
    
    # -- use height scanner atatached to foot
    foot_landing_penalty_left = RewTerm(
        func=hector_mdp.swing_foot_landing_penalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner_L_foot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names="L_toe"),
            "std": 0.03, 
        },
    )
    
    foot_landing_penalty_right = RewTerm(
        func=hector_mdp.swing_foot_landing_penalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner_R_foot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names="R_toe"),
            "std": 0.03, 
        },
    )
    
    # # -- penalize foot placement
    # foot_placement = RewTerm(
    #     func=hector_mdp.foot_placement_penalty,
    #     weight=-0.01,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner_fine"),
    #         "action_name": "mpc_action",
    #         "l_toe": 0.091+0.02,
    #         "l_heel": 0.054+0.02,
    #         "l_width": 0.073+0.04,
    #         "std": 0.03, 
    #     },
    # )
    
@configclass
class HECTORSlipRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""
    # -- rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    track_height_exp = RewTerm(
        func=hector_mdp.track_torso_height_exp, 
        weight=0.1,
        params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_sole"),
                "std": 0.1,
                "reference_height": 0.55,
                },
    )

    # -- penalties
    # termination = RewTerm(func=mdp.is_terminated, weight=-200.0) # type: ignore
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) # type: ignore
    # lin_accel_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}) # type: ignore
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, # type: ignore
        weight=-0.015, 
        )
    
    # -- joint penalties
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,  # type: ignore
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
    )
    dof_acc_l2 = None
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,  # type: ignore
        weight=-1.0e-5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
        )
    
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_toe_joint"])},
    )
    
    # -- energy penalty
    energy_penalty_l2 = RewTerm(
        func=hector_mdp.energy_penalty_l2, # type: ignore
        weight=-0.005,
        params={
            "assymetric_indices": [7], 
            "action_name": "mpc_action",
        }
    )
    
    # -- foot penalties
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
        },
    )
    
    # body-leg angle penalties
    leg_body_angle_l2 = RewTerm(
        func=hector_mdp.leg_body_angle_l2,
        weight=-1.0,
        params={"action_name": "mpc_action"}
    )
    leg_body_distance_l2 = RewTerm(
        func=hector_mdp.leg_distance_l2,
        weight=-1.0,
        params={"action_name": "mpc_action"}
    )
    
    # -- MPC cost
    mpc_cost_l2 = RewTerm(
        func=hector_mdp.mpc_cost_l1, # type: ignore
        weight=-1e-4,
        params={
            "action_name": "mpc_action",
        },
        )
    
    # disable rewards from parent config
    dof_pos_limits = None
    joint_deviation = None
    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = None