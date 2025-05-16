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
from .rough_env_cfg import HECTORSceneCfg, HECTORRewards, HECTORActionsCfg, HECTORCommandsCfg, HECTORTerminationsCfg, HECTOREventCfg, CurriculumCfg


@configclass
class HECTORObservationsCfg:
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
        
        height_scan = ObsTerm(
            func=hector_mdp.height_scan, # type: ignore
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                },
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class HECTORRoughEnvSACCfg(LocomotionVelocityRoughEnvCfg):
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
        self.sim.dt = 1/500
        self.decimation = 5
        self.sim.render_interval = 10
        self.events.reset_camera = None

@configclass
class HECTORRoughEnvSACCfgPLAY(HECTORRoughEnvSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 42
        self.scene.terrain = hector_mdp.SteppingStoneTerrain
        self.scene.height_scanner.debug_vis = True
        # self.events.reset_camera = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
        self.curriculum.terrain_levels = None