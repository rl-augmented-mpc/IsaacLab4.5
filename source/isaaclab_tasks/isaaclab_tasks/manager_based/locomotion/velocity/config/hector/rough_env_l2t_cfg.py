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
    HECTORL2TActionsCfg, 
    HECTORCommandsCfg,
    HECTORCurriculumCfg,
    HECTOREventCfg,
    TeacherObsCfg,
    StudentObsCfg,
    HECTORRewardsCfg,
    HECTORTerminationsCfg,
    HECTORSceneCfg,
)

@configclass
class L2TObservationCfg:
    """Base class for L2T observations."""
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()

@configclass
class HECTORRoughEnvL2TCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORSceneCfg = HECTORSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HECTORRewardsCfg = HECTORRewardsCfg()
    actions: HECTORL2TActionsCfg = HECTORL2TActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    observations: L2TObservationCfg = L2TObservationCfg()
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
        
        self.viewer = ViewerCfg(
            # eye=(-2.5, 0.0, 0.2), 
            # lookat=(-1.0, 0.0, 0.0),
            eye=(0.0, -2.2, 0.4), 
            lookat=(0.0, -1.0, 0.2),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )
        
        self.scene.terrain = hector_mdp.CurriculumRandomBlockTerrain
        
        # command
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-(20.0/180)*math.pi, (20.0/180)*math.pi)
        
        # event 
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }

@configclass
class HECTORRoughEnvL2TCfgPLAY(HECTORRoughEnvL2TCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 42
        self.scene.terrain = hector_mdp.InferenceRandomBlockTerrain
        
        self.scene.height_scanner.debug_vis = True
        # self.curriculum.terrain_levels = None # disable terrain curriculum
        
        # lower resolution of heightmap since we do not use these during inference
        self.scene.height_scanner_fine.pattern_cfg.resolution = 0.5
        self.scene.height_scanner_L_foot.pattern_cfg.resolution = 0.5
        self.scene.height_scanner_R_foot.pattern_cfg.resolution = 0.5
        
        # self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)