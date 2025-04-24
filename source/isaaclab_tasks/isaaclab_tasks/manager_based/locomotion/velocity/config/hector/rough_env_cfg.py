# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.hector import HECTOR_CFG
ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class HECTORRewards(RewardsCfg):
    """Reward terms for the MDP."""
    # -- rewards
    alive = RewTerm(
        func=mdp.is_alive,  # type: ignore
        weight=0.01,
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.2,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=0.2, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) # type: ignore
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05) # type: ignore
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
        },
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,  # type: ignore
        weight=-1.0e-5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
        )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,  # type: ignore
        weight=-2.5e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
        )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,  # type: ignore
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_toe_joint"])},
    )


@configclass
class HECTORObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # type: ignore
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # type: ignore
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            noise=Unoise(n_min=-0.05, n_max=0.05),
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
        actions = ObsTerm(func=mdp.last_action) # type: ignore
        height_scan = ObsTerm(
            func=hector_mdp.height_scan, # type: ignore
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
@configclass
class HECTORActionsCfg:
    """Action specifications for the MDP."""

    mpc_action = hector_mdp.MPCActionCfg(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-0.4, -0.001, -0.5), 
            (0.4, 0.15, 0.5)
        )
    )
    
@configclass
class HECTORTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # type: ignore
    body_contact = DoneTerm(
        func=mdp.illegal_contact, # type: ignore
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,  # type: ignore
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi/3},
        time_out=True,
    )
    # terrain_out_of_bounds = DoneTerm(
    #     func=mdp.terrain_out_of_bounds,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    #     time_out=True,
    # )


@configclass
class HECTORRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HECTORRewards = HECTORRewards()
    actions: HECTORActionsCfg = HECTORActionsCfg()
    observations: HECTORObservationsCfg = HECTORObservationsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    seed = 10

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 10
        
        # viewer 
        self.viewer = ViewerCfg(
            # eye=(10.0, -10.0, 2.0), 
            # lookat=(5.0, -5.0, 0.0),
            eye=(2.0, -5.0, 2.0), 
            lookat=(0.0, 0.0, 0.0),
            resolution=(1920, 1080)
        )
        
        # Scene
        self.scene.robot = HECTOR_CFG.replace(prim_path=f"{ENV_REGEX_NS}/Robot")
        # self.scene.contact_forces.prim_path = f"{ENV_REGEX_NS}/Robot/[L|R]_toe"
        self.scene.height_scanner.prim_path = f"{ENV_REGEX_NS}/Robot/base"
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0])
        self.scene.terrain = hector_mdp.SteppingStoneTerrain

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.base_external_force_torque = None
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["trunk"]
        
        # Reset
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-5.0, 5.0), "yaw": (-0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Rewards
        self.rewards.undesired_contacts = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)


@configclass
class HECTORRoughEnvCfg_PLAY(HECTORRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
