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
    HECTORBlindLocomotionActionsCfg, 
    HECTORPerceptiveLocomotionActionsCfg, 
    HECTORBlindLocomotionObservationsCfg,
    HECTORPerceptiveLocomotionObservationsCfg, 
    HECTORBlindLocomotionRewardsCfg,
    HECTORPerceptiveLocomotionRewardsCfg, 
    HECTORCommandsCfg,
    HECTORCurriculumCfg,
    HECTOREventCfg,
    HECTORTerminationsCfg,
    HECTORBlindLocomotionSceneCfg,
    HECTORPerceptiveLocomotionSceneCfg,
)

@configclass
class HECTORRoughEnvBlindLocomotionSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORBlindLocomotionSceneCfg = HECTORBlindLocomotionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HECTORBlindLocomotionObservationsCfg = HECTORBlindLocomotionObservationsCfg()
    rewards: HECTORBlindLocomotionRewardsCfg = HECTORBlindLocomotionRewardsCfg()
    actions: HECTORBlindLocomotionActionsCfg = HECTORBlindLocomotionActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTOREventCfg = HECTOREventCfg()
    curriculum: HECTORCurriculumCfg = HECTORCurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 2*self.decimation

        # terain
        self.scene.terrain = hector_mdp.SteppingStoneTerrain 
        # self.scene.terrain = hector_mdp.BaseTerrain # policy sanity check

        # sensor
        self.scene.height_scanner = None
        
        self.viewer = ViewerCfg(
            eye=(0.0, -2.0, 0.4), 
            lookat=(0.0, -0.5, 0.1),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )

        # event 
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5), 
            "y": (-0.5, 0.5), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            # "yaw": (-math.pi, math.pi),
            "yaw": (-math.pi, math.pi),
        }
        
        # command 
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

@configclass
class HECTORRoughEnvBlindLocomotionSACCfgPLAY(HECTORRoughEnvBlindLocomotionSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 42
        # self.seed = 100

        # terrain
        self.scene.terrain = hector_mdp.InferenceSteppingStoneTerrain
        # self.scene.terrain = hector_mdp.BaseTerrain # policy sanity check
        
        # event 
        # self.events.reset_base.func=hector_mdp.reset_root_state_orthogonal
        # self.events.reset_base.params["multiplier"] = 2
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3), 
            # "x": (0.635, 0.635), # intentional trip over
            # "y": (-0.0, 0.0), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
            # "yaw": (0.0, 0.0),
        }
        self.events.reset_terrain_type = None

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)

@configclass
class HECTORRoughEnvPerceptiveLocomotionSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORPerceptiveLocomotionSceneCfg = HECTORPerceptiveLocomotionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HECTORPerceptiveLocomotionObservationsCfg = HECTORPerceptiveLocomotionObservationsCfg()
    rewards: HECTORPerceptiveLocomotionRewardsCfg = HECTORPerceptiveLocomotionRewardsCfg()
    actions: HECTORPerceptiveLocomotionActionsCfg = HECTORPerceptiveLocomotionActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTOREventCfg = HECTOREventCfg()
    curriculum: HECTORCurriculumCfg = HECTORCurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 2*self.decimation

        self.scene.terrain = hector_mdp.SteppingStoneTerrain
        
        self.viewer = ViewerCfg(
            eye=(0.0, -2.0, 0.4), 
            lookat=(0.0, -0.5, 0.1),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )

        # event 
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5), 
            "y": (-0.5, 0.5), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
        
        # command 
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)

@configclass
class HECTORRoughEnvPerceptiveLocomotionSACCfgPLAY(HECTORRoughEnvPerceptiveLocomotionSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 42

        # self.scene.terrain = hector_mdp.InferenceRandomBlockTerrain
        # self.scene.terrain = hector_mdp.TripOverChallengeTerrain
        # self.scene.terrain = hector_mdp.BoxRoughTerrain
        self.scene.terrain = hector_mdp.InferenceSteppingStoneTerrain

        # event 
        # self.events.reset_base.func=hector_mdp.reset_root_state_orthogonal
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            # "yaw": (-math.pi, math.pi),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_terrain_type = None

        # debug vis
        self.scene.height_scanner_L_foot.debug_vis = True
        self.scene.height_scanner_R_foot.debug_vis = True
        
        self.curriculum.terrain_levels = None
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)