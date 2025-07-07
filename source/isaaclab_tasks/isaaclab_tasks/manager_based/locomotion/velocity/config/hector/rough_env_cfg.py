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
    PPOHECTORObservationsCfg,
    HECTORBlindLocomotionRewardsCfg,
    HECTORTerminationsCfg,
    HECTORSceneCfg,
)


@configclass
class HECTORRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORSceneCfg = HECTORSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: HECTORBlindLocomotionRewardsCfg = HECTORBlindLocomotionRewardsCfg()
    actions: HECTORActionsCfg = HECTORActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    observations: PPOHECTORObservationsCfg = PPOHECTORObservationsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTOREventCfg = HECTOREventCfg()
    curriculum: HECTORCurriculumCfg = HECTORCurriculumCfg()
    seed = 42
    # viewer: ViewerCfg = ViewerCfg(
    #     resolution=(1920, 1080)
    #     )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/500
        self.decimation = 5
        self.sim.render_interval = 10

@configclass
class HECTORRoughEnvCfgPLAY(HECTORRoughEnvCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        super().__post_init__()
        self.seed = 42
        # self.scene.height_scanner.debug_vis = True
        # self.events.reset_camera = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.curriculum.terrain_levels = None