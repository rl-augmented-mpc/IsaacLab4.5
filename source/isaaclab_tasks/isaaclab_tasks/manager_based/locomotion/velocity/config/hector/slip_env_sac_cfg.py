# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

from .env_cfg import (
    HECTORCommandsCfg,
    HECTORSlipCurriculumCfg,
    HECTORSlipEventCfg,
    HECTORTerminationsCfg,
    HECTORSlipActionsCfg, 
    HECTORBlindLocomotionObservationsCfg,
    HECTORSlipRewardsCfg,
    HECTORSlipSceneCfg,
)

@configclass
class HECTORSlipEnvSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORSlipSceneCfg = HECTORSlipSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HECTORSlipRewardsCfg = HECTORSlipRewardsCfg()
    actions: HECTORSlipActionsCfg = HECTORSlipActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    observations: HECTORBlindLocomotionObservationsCfg = HECTORBlindLocomotionObservationsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTORSlipEventCfg = HECTORSlipEventCfg()
    curriculum: HECTORSlipCurriculumCfg = HECTORSlipCurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 2*self.decimation

        self.scene.terrain = hector_mdp.CurriculumFrictionPatchTerrain
        self.scene.height_scanner = None
        
        self.viewer = ViewerCfg(
            eye=(0.0, -2.0, 0.0), 
            lookat=(0.0, -1.0, 0.0),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )
        
        # command
        # self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
        # self.commands.base_velocity.ranges.ang_vel_z = (-(20.0/180)*math.pi, (20.0/180)*math.pi)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        # scene
        self.observations.exteroception = None
        self.scene.height_scanner_L_foot = None
        self.scene.height_scanner_R_foot = None
        
        # event 
        self.events.reset_base.params["pose_range"] = {
            "x": (-2.0, 2.0), 
            "y": (-2.0, 2.0), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
            # "yaw": (-0, 0),
        }

        # friction pyramid
        # self.actions.mpc_action.friction_cone_coef = 0.5
        self.events.reset_terrain_type = None

@configclass
class HECTORSlipEnvSACCfgPLAY(HECTORSlipEnvSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.seed = 42
        self.seed = 100
        self.scene.terrain = hector_mdp.InferenceAlternatingFrictionPatchTerrain
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_base.params["pose_range"] = {
            # "x": (-1.0, 1.0), 
            # "y": (-1.0, 1.0), 
            "x": (-0.5, -0.5), 
            "y": (-2.0, 2.0), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            # "yaw": (-math.pi/6, math.pi/6),
            "yaw": (-0, 0),
        }
        self.events.reset_terrain_type = None


        # better visualization 
        # self.sim.render_interval = self.decimation
        self.scene.sky_light.init_state.rot = (0.9063078, 0.0, 0.0, 0.4226183)
        self.viewer = ViewerCfg(
            eye=(-0.5, -2.0, 0.3), 
            lookat=(0.0, -0.5, 0.0),
            # resolution=(1920, 1080), # full HD
            resolution=(3840, 2160), # 4K
            origin_type="asset_root", 
            asset_name="robot"
        )