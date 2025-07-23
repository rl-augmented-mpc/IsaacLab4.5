# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp


@configclass
class PPOHECTORObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos_z = ObsTerm(
            func=hector_mdp.base_pos_z, # type: ignore
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_sole"),
                },
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, # type: ignore
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, # type: ignore
            # noise=Unoise(n_min=-0.2, n_max=0.2)
            )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            # noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
        
        joint_pos = ObsTerm(
            func=hector_mdp.joint_pos, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            )
        joint_vel = ObsTerm(
            func=hector_mdp.joint_vel, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        joint_torque = ObsTerm(
            func=hector_mdp.joint_torque, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        
        swing_phase = ObsTerm(
            func=hector_mdp.swing_phase, 
            params={"action_name": "mpc_action"}
        )
        foot_placement_b = ObsTerm(
            func=hector_mdp.foot_placement_b,
            params={"action_name": "mpc_action"}
        )
        foot_position_b = ObsTerm(
            func=hector_mdp.foot_position_b,
            params={"action_name": "mpc_action"}
        )
        reference_foot_position_b = ObsTerm(
            func=hector_mdp.reference_foot_position_b,
            params={"action_name": "mpc_action"}
        )
        
        
        actions = ObsTerm(func=mdp.last_action) # type: ignore

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class ImageCfg(ObsGroup):
        """Observations for image group."""
        height_scan = ObsTerm(
            func=hector_mdp.height_scan, # type: ignore
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"), 
                "reshape_as_image": True
                },
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    image: ImageCfg = ImageCfg()



@configclass
class HECTORBlindLocomotionObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved !!!!)
        base_pos_z = ObsTerm(
            func=hector_mdp.base_pos_z, # type: ignore
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
                },
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, # type: ignore
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, # type: ignore
            # noise=Unoise(n_min=-0.2, n_max=0.2)
            )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            # noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # user_velocity_commands = ObsTerm(func=hector_mdp.reference_command, params={"action_name": "mpc_action"}) # type: ignore
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
        
        joint_pos = ObsTerm(
            func=hector_mdp.joint_pos, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            )
        joint_vel = ObsTerm(
            func=hector_mdp.joint_vel, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        joint_torque = ObsTerm(
            func=hector_mdp.joint_torque, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        
        swing_phase = ObsTerm(
            func=hector_mdp.swing_phase, 
            params={"action_name": "mpc_action"}
        )
        foot_placement_b = ObsTerm(
            func=hector_mdp.foot_placement_b,
            params={"action_name": "mpc_action"}
        )
        foot_position_b = ObsTerm(
            func=hector_mdp.foot_position_b,
            params={"action_name": "mpc_action"}
        )
        reference_foot_position_b = ObsTerm(
            func=hector_mdp.reference_foot_position_b,
            params={"action_name": "mpc_action"}
        )
        
        actions = ObsTerm(func=mdp.last_action) # type: ignore

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ExtraObsCfg(ObsGroup):
        """Observations for extra like debug."""

        # observation terms (order preserved !!!!)
        
        contact_force = ObsTerm(
            func=hector_mdp.contact_forces, 
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                },
            ) # type: ignore

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    extra: ExtraObsCfg = ExtraObsCfg() # <- only for data analysis

@configclass
class HECTORPerceptiveLocomotionObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved !!!!)

        # -- proprioception 
        # base_pos_z = ObsTerm(
        #     func=hector_mdp.base_pos_z, # type: ignore
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
        #         },
        #     # noise=Unoise(n_min=-0.1, n_max=0.1)
        #     )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            # noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, # type: ignore
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, # type: ignore
            # noise=Unoise(n_min=-0.2, n_max=0.2)
            )
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity, # type: ignore
        #     # noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        
        # user_velocity_commands = ObsTerm(func=hector_mdp.reference_command, params={"action_name": "mpc_action"}) # type: ignore
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
        
        joint_pos = ObsTerm(
            func=hector_mdp.joint_pos, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            )
        joint_vel = ObsTerm(
            func=hector_mdp.joint_vel, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        joint_torque = ObsTerm(
            func=hector_mdp.joint_torque, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        
        # -- MPC state
        swing_phase = ObsTerm(
            func=hector_mdp.swing_phase, 
            params={"action_name": "mpc_action"}
        )
        foot_placement_b = ObsTerm(
            func=hector_mdp.foot_placement_b,
            params={"action_name": "mpc_action"}
        )
        foot_position_b = ObsTerm(
            func=hector_mdp.foot_position_b,
            params={"action_name": "mpc_action"}
        )
        reference_foot_position_b = ObsTerm(
            func=hector_mdp.reference_foot_position_b,
            params={"action_name": "mpc_action"}
        )
        
        actions = ObsTerm(func=mdp.last_action) # type: ignore
        
        height_scan = ObsTerm(
            func=mdp.height_scan, # type: ignore
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.56,
                },
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # height_scan = ObsTerm(
        #     func=hector_mdp.foot_centric_height_scan, # type: ignore
        #     params={"action_name": "mpc_action"}, 
        #     # noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # ) # -- does not have effect on performance

        
        # depth_image = ObsTerm(
        #     func=hector_mdp.depth_image, # type: ignore
        #     params={
        #         "sensor_cfg": SceneEntityCfg("tiled_d455"),
        #         },
        #     # noise=Unoise(n_min=-0.1, n_max=0.1),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class ExtraObsCfg(ObsGroup):
        """Observations for extra like debug."""

        # observation terms (order preserved !!!!)
        
        contact_force = ObsTerm(
            func=hector_mdp.contact_forces, 
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                },
            ) # type: ignore

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    extra: ExtraObsCfg = ExtraObsCfg() # <- only for data analysis
    
    
@configclass
class SACHECTORSlipObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos_z = ObsTerm(
            func=hector_mdp.base_pos_z, # type: ignore
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
                },
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, # type: ignore
            # noise=Unoise(n_min=-0.1, n_max=0.1)
            )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, # type: ignore
            # noise=Unoise(n_min=-0.2, n_max=0.2)
            )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            # noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
        
        joint_pos = ObsTerm(
            func=hector_mdp.joint_pos, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            )
        joint_vel = ObsTerm(
            func=hector_mdp.joint_vel, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        joint_torque = ObsTerm(
            func=hector_mdp.joint_torque, 
            params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            )
        
        swing_phase = ObsTerm(
            func=hector_mdp.swing_phase, 
            params={"action_name": "mpc_action"}
        )
        foot_placement_b = ObsTerm(
            func=hector_mdp.foot_placement_b,
            params={"action_name": "mpc_action"}
        )
        foot_position_b = ObsTerm(
            func=hector_mdp.foot_position_b,
            params={"action_name": "mpc_action"}
        )
        reference_foot_position_b = ObsTerm(
            func=hector_mdp.reference_foot_position_b,
            params={"action_name": "mpc_action"}
        )
        
        
        actions = ObsTerm(func=mdp.last_action) # type: ignore

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class ExtraObsCfg(ObsGroup):
        """Observations for extra like debug."""

        # observation terms (order preserved !!!!)
        
        contact_force = ObsTerm(
            func=hector_mdp.contact_forces, 
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
                },
            ) # type: ignore

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    extra: ExtraObsCfg = ExtraObsCfg() # <- only for data analysis
    

"""
L2T policy observations
"""

@configclass
class TeacherObsCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_pos_z = ObsTerm(
        func=hector_mdp.base_pos_z, # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
            },
        )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity, # type: ignore
    )
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel, # type: ignore
        )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, # type: ignore
        )
    # user_velocity_commands = ObsTerm(func=hector_mdp.reference_command, params={"action_name": "mpc_action"}) # type: ignore
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
    
    joint_pos = ObsTerm(
        func=hector_mdp.joint_pos, 
        params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
        )
    joint_vel = ObsTerm(
        func=hector_mdp.joint_vel, 
        params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
        )
    joint_torque = ObsTerm(
        func=hector_mdp.joint_torque, 
        params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
        )
    
    swing_phase = ObsTerm(
        func=hector_mdp.swing_phase, 
        params={"action_name": "mpc_action"}
    )
    foot_placement_b = ObsTerm(
        func=hector_mdp.foot_placement_b,
        params={"action_name": "mpc_action"}
    )
    foot_position_b = ObsTerm(
        func=hector_mdp.foot_position_b,
        params={"action_name": "mpc_action"}
    )
    reference_foot_position_b = ObsTerm(
        func=hector_mdp.reference_foot_position_b,
        params={"action_name": "mpc_action"}
    )
    
    actions = ObsTerm(func=mdp.last_action) # type: ignore
    
    height_scan = ObsTerm(
        func=mdp.height_scan, # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": 0.56,
            },
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        
@configclass
class StudentObsCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    # base_pos_z = ObsTerm(
    #     func=hector_mdp.base_pos_z, # type: ignore
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
    #         },
    #     noise=Gnoise(mean=0.0, std=0.1),
    #     )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity, # type: ignore
        noise=Gnoise(mean=0.0, std=0.075),
    )
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel, # type: ignore
        noise=Gnoise(mean=0.0, std=0.15),
        )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, # type: ignore
        noise=Gnoise(mean=0.0, std=0.15),
        )
    
    # user_velocity_commands = ObsTerm(func=hector_mdp.reference_command, params={"action_name": "mpc_action"}) # type: ignore
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
    
    joint_pos = ObsTerm(
        func=hector_mdp.joint_pos, 
        params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
        noise=Gnoise(mean=0.0, std=0.175),
        )
    joint_vel = ObsTerm(
        func=hector_mdp.joint_vel, 
        params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
        noise=Gnoise(mean=0.0, std=0.175),
        )
    joint_torque = ObsTerm(
        func=hector_mdp.joint_torque, 
        params={"joint_names": ['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint']}, 
        noise=Gnoise(mean=0.0, std=0.175),
        )
    
    swing_phase = ObsTerm(
        func=hector_mdp.swing_phase, 
        params={"action_name": "mpc_action"}
    )
    foot_placement_b = ObsTerm(
        func=hector_mdp.foot_placement_b,
        params={"action_name": "mpc_action"}
    )
    foot_position_b = ObsTerm(
        func=hector_mdp.foot_position_b,
        params={"action_name": "mpc_action"}
    )
    reference_foot_position_b = ObsTerm(
        func=hector_mdp.reference_foot_position_b,
        params={"action_name": "mpc_action"}
    )
    
    actions = ObsTerm(func=mdp.last_action) # type: ignore
    
    height_scan = ObsTerm(
        func=mdp.height_scan, # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "offset": 0.56,
            },
        clip=(-1.0, 1.0),
        noise=Gnoise(mean=0.0, std=0.03),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True