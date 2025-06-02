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
class HECTORRewards(RewardsCfg):
    """Reward terms for the MDP."""
    # -- rewards
    # alive = RewTerm(
    #     func=mdp.is_alive,  # type: ignore
    #     weight=0.1,
    # )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.1,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.1, 
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
        
    # # -- foot landing rewards
    # foot_landing = RewTerm(
    #     func=hector_mdp.stance_foot_position_reward,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
    #         "action_name": "mpc_action",
    #         "std": 0.03, 
    #     },
    # )
    
    # slacked_foot_landing = RewTerm(
    #     func=hector_mdp.stance_foot_position_reward,
    #     weight=0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
    #         "action_name": "mpc_action",
    #         "l_toe": 0.091+0.02,
    #         "l_heel": 0.054+0.02,
    #         "l_width": 0.073+0.04,
    #         "std": 0.03, 
    #     },
    # )
    
    # foot_placement = RewTerm(
    #     func=hector_mdp.foot_placement_reward,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "action_name": "mpc_action",
    #         "std": 0.03, 
    #     },
    # )
    
    # slacked_foot_placement = RewTerm(
    #     func=hector_mdp.foot_placement_reward,
    #     weight=0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "action_name": "mpc_action",
    #         "l_toe": 0.091+0.02,
    #         "l_heel": 0.054+0.02,
    #         "l_width": 0.073+0.04,
    #         "std": 0.03, 
    #     },
    # )

    # -- penalties
    termnation = RewTerm(func=mdp.is_terminated, weight=-200.0) # type: ignore
    # negative_lin_vel_l2 = RewTerm(func=hector_mdp.negative_lin_vel_l2, weight=-0.1)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) # type: ignore
    lin_accel_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-5e-4, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}) # type: ignore
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05) # type: ignore
    
    # -- joint penalties
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,  # type: ignore
        weight=-1.0e-5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
        )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,  # type: ignore
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
    )
    dof_acc_l2 = None
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, # type: ignore
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_toe_joint"])},
    )
    
    # -- energy penalty
    processed_action_l2 = RewTerm(
        func=hector_mdp.individual_action_l2, # type: ignore
        weight=-0.5,
        params={
            "action_idx": [-3, -2, -1],
            "action_name": "mpc_action",
        }
    )
    
    # -- foot penalties
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
        },
    )
    
    # body-leg angle penalties
    leg_body_angle_l2 = RewTerm(
        func=hector_mdp.leg_body_angle_l2, 
        weight=-0.2,
        params={"action_name": "mpc_action"}
    )
    leg_body_distance_l2 = RewTerm(
        func=hector_mdp.leg_distance_l2,
        weight=-0.2,
        params={"action_name": "mpc_action"}
    )
    
    # contact penalties
    undesired_contacts_knee = RewTerm(
        func=mdp.undesired_contacts, # type: ignore
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )

    undesired_contacts_toe = RewTerm(
        func=mdp.undesired_contacts, # type: ignore
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("toe_contact", body_names=".*_toe_tip"), "threshold": 1.0},
    )
    
    # -- MPC cost
    mpc_cost_l2 = RewTerm(
        func=hector_mdp.mpc_cost_l1, # type: ignore
        weight=-1e-4,
        params={
            "action_name": "mpc_action",
        },
        )
    
    # -- foot placement penalties
    foot_landing = RewTerm(
        func=hector_mdp.stance_foot_position_penalty,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "action_name": "mpc_action",
            "std": 0.03, 
        },
    )
    
    slacked_foot_landing = RewTerm(
        func=hector_mdp.stance_foot_position_penalty,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "action_name": "mpc_action",
            "l_toe": 0.091+0.02,
            "l_heel": 0.054+0.02,
            "l_width": 0.073+0.04,
            "std": 0.03, 
        },
    )
    
    foot_placement = RewTerm(
        func=hector_mdp.foot_placement_penalty,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "action_name": "mpc_action",
            "std": 0.03, 
        },
    )
    
    slacked_foot_placement = RewTerm(
        func=hector_mdp.foot_placement_penalty,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "action_name": "mpc_action",
            "l_toe": 0.091+0.02,
            "l_heel": 0.054+0.02,
            "l_width": 0.073+0.04,
            "std": 0.03, 
        },
    )
    
    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = None