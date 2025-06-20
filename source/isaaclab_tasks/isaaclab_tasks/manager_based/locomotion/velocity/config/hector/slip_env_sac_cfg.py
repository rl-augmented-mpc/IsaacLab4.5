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
    HECTORCommandsCfg,
    HECTORSlipCurriculumCfg,
    HECTORSlipEventCfg,
    HECTORTerminationsCfg,
    HECTORSlipActionsCfg, 
    SACHECTORSlipObservationsCfg,
    HECTORSlipRewardsCfg,
    HECTORSlipSceneCfg,
)

@configclass
class HECTORSlipEnvSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORSlipSceneCfg = HECTORSlipSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HECTORSlipRewardsCfg = HECTORSlipRewardsCfg()
    actions: HECTORSlipActionsCfg = HECTORSlipActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    observations: SACHECTORSlipObservationsCfg = SACHECTORSlipObservationsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTORSlipEventCfg = HECTORSlipEventCfg()
    curriculum: HECTORSlipCurriculumCfg = HECTORSlipCurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/500
        self.decimation = 5
        self.sim.render_interval = 10
        
        self.viewer = ViewerCfg(
            eye=(0.0, -2.5, 0.0), 
            lookat=(0.0, -1.0, 0.0),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )

@configclass
class HECTORSlipEnvSACCfgPLAY(HECTORSlipEnvSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 42
        self.scene.terrain = hector_mdp.FrictionPatchTerrain
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)