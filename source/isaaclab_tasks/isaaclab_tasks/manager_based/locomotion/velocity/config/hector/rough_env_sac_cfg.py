# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

from .env_cfg import (
    HECTORActionsCfg, 
    HECTORCommandsCfg,
    HECTORCurriculumCfg,
    HECTOREventCfg,
    SACHECTORObservationsCfg,
    HECTORRewardsCfg,
    HECTORTerminationsCfg,
    HECTORSceneCfg,
)

@configclass
class HECTORRoughEnvSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORSceneCfg = HECTORSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HECTORRewardsCfg = HECTORRewardsCfg()
    actions: HECTORActionsCfg = HECTORActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    observations: SACHECTORObservationsCfg = SACHECTORObservationsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTOREventCfg = HECTOREventCfg()
    curriculum: HECTORCurriculumCfg = HECTORCurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/500
        self.decimation = 5
        self.sim.render_interval = 10

@configclass
class HECTORRoughEnvSACCfgPLAY(HECTORRoughEnvSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 42
        # self.scene.terrain = hector_mdp.InferenceSteppingStoneTerrain
        self.scene.terrain = hector_mdp.InferenceRandomBlockTerrain
        # self.scene.height_scanner.debug_vis = True
        # self.events.reset_camera = None
        # self.curriculum.terrain_levels = None
        self.viewer = ViewerCfg(
            eye=(0.0, -8.0, 0.5), 
            lookat=(0.0, 0.0, 0.0),
            resolution=(1920, 1080), 
        )
        
        # lower resolution of heightmap since we do not use these during inference
        # self.scene.height_scanner_fine.pattern_cfg.resolution = 0.5
        self.scene.height_scanner_L_foot.pattern_cfg.resolution = 0.5
        self.scene.height_scanner_R_foot.pattern_cfg.resolution = 0.5
        
        # self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)