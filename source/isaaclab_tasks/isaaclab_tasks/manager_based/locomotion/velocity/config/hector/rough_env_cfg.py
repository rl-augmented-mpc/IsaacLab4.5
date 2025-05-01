# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, EventCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.hector import HECTOR_CFG

@configclass
class HECTORSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = hector_mdp.CurriculumSteppingStoneTerrain
    # robots
    robot: ArticulationCfg = HECTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/10,
    )
    contact_forces = ContactSensorCfg(
        # prim_path="{ENV_REGEX_NS}/Robot/.*", 
        prim_path="{ENV_REGEX_NS}/Robot/.*_toe", 
        history_length=3, 
        track_air_time=True,
        update_period=1/100,
        )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # light = AssetBaseCfg(
    #         prim_path="/World/distantLight",
    #         spawn=sim_utils.DistantLightCfg(
    #             intensity=1500.0,
    #         )
    #     )

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
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.2, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    foot_placement = RewTerm(
        func=hector_mdp.stance_foot_position_reward,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "action_name": "mpc_action", 
            "std": 0.01,
        },
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01) # type: ignore
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05) # type: ignore
    
    # processed action regularization 
    stepping_frequency_l2 = RewTerm(
        func=hector_mdp.individual_action_l2, # type: ignore
        # weight=-0.1, 
        weight=-0.3, 
        params={
            "action_idx": 0,
        },
        )
    foot_height_l2 = RewTerm(
        func=hector_mdp.individual_action_l2, # type: ignore
        # weight=-0.1, 
        weight=-0.5, 
        params={
            "action_idx": 1,
        },
        )
    control_point_l2 = RewTerm(
        func=hector_mdp.individual_action_l2, # type: ignore
        # weight=-0.1,
        weight=-0.3,
        params={
            "action_idx": 2,
        },
        )
    mpc_cost_l2 = RewTerm(
        func=hector_mdp.mpc_cost_l1, # type: ignore
        weight=-1e-4,
        params={
            "action_name": "mpc_action",
        },
        )
    # energy_l2 = RewTerm(func=mdp.action_l2, weight=-0.01) # type: ignore
    
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
    dof_acc_l2 = None
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,  # type: ignore
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"])}
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_toe_joint"])},
    )
    
    # # Penalize body leg angle
    leg_body_angle_l2 = RewTerm(
        func=hector_mdp.leg_body_angle_l2, 
        weight=-0.1,
        params={"action_name": "mpc_action"}
    )
    
    undesired_contacts = None


@configclass
class HECTORObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos_z = ObsTerm(
            func=mdp.base_pos_z, # type: ignore
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
        height_scan = ObsTerm(
            func=hector_mdp.height_scan, # type: ignore
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
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
            (-0.4, 0.0, -0.4), 
            (0.4, 0.15, 0.4)
        )
    )

@configclass
class HECTORCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg( # type: ignore
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( # type: ignore
            lin_vel_x=(0.4, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi)
        ),
    )
    
@configclass
class HECTORTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # type: ignore
    # body_contact = DoneTerm(
    #     func=mdp.illegal_contact, # type: ignore
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
    # ) # this triggers wrong body names register to contact sensor
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,  # type: ignore
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi/3},
        time_out=True,
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.5},
        time_out=True,
    )
    
@configclass 
class HECTOREventCfg(EventCfg):
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = None

    # reset
    base_external_force_torque = None
    
    # random initial noise added to default state defined in articulation cfg
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, # type: ignore
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-3.0, 3.0), 
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.0, 0.0),
                # "yaw": (-math.pi/6, math.pi/6),
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
    reset_camera = EventTerm(
        func=hector_mdp.reset_camera, # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=hector_mdp.terrain_levels_vel)


@configclass
class HECTORRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORSceneCfg = HECTORSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HECTORRewards = HECTORRewards()
    actions: HECTORActionsCfg = HECTORActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    observations: HECTORObservationsCfg = HECTORObservationsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTOREventCfg = HECTOREventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 10

@configclass
class HECTORRoughEnvCfgPLAY(HECTORRoughEnvCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 77
        self.scene.terrain.max_init_terrain_level = 3
        self.scene.height_scanner.debug_vis = True
        # self.events.reset_camera = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.7)
        self.curriculum.terrain_levels = None