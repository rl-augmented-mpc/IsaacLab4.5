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
        self.sim.dt = 1/200
        self.decimation = 2
        self.sim.render_interval = 2*self.decimation
        self.episode_length_s = 10.0

        # terain
        self.scene.terrain = hector_mdp.SteppingStoneTerrain 
        # self.scene.terrain = hector_mdp.BaseTerrain # policy sanity check

        # sensor
        self.scene.height_scanner = None
        self.scene.height_scanner_L_foot = None
        self.scene.height_scanner_R_foot = None
        self.observations.exteroception = None
        
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
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

@configclass
class HECTORRoughEnvBlindLocomotionSACCfgPLAY(HECTORRoughEnvBlindLocomotionSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.seed = 42
        self.seed = 100

        # terrain
        self.scene.terrain = hector_mdp.InferenceSteppingStoneTerrain
        
        # event 
        # self.events.reset_terrain_type = None
        # self.curriculum.terrain_levels = None

        self.events.reset_base.func=hector_mdp.reset_root_state_orthogonal
        self.events.reset_base.params["multiplier"] = 2
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
            # "yaw": (-math.pi/4, math.pi/4),
            # "yaw": (0.0, 0.0),
        }

        # # intentional trip over
        # self.events.reset_base.params["pose_range"] = {
        #     # "x": (0.635, 0.635), # intentional trip over
        #     "x": (0.635, 0.635), # intentional trip over
        #     "y": (-0.4, 0.4),  
        #     "z": (0.0, 0.0),
        #     "roll": (0.0, 0.0),
        #     "pitch": (0.0, 0.0),
        #     "yaw": (0.0, 0.0),
        # }

        # command
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)

        # better visualization 
        self.scene.toe_contact.debug_vis = False
        # self.sim.render_interval = self.decimation
        self.scene.sky_light.init_state.rot = (0.9238795, 0.0, 0.0, -0.3826834)
        self.viewer = ViewerCfg(
            # eye=(-0.0, -2.0, 0.5), 
            # lookat=(0.0, -0.5, 0.0),
            eye=(-0.0, -1.5, 0.2), 
            lookat=(0.0, -0.8, 0.0),
            # resolution=(1920, 1080), # full HD
            resolution=(3840, 2160), # 4K
            origin_type="asset_root", 
            asset_name="robot"
        )

        # rendering optimization 
        self.sim.render.dlss_mode = 1
        self.sim.render.antialiasing_mode = None
        # self.sim.render.enable_global_illumination = True

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
        self.sim.dt = 1/200
        self.decimation = 2
        self.sim.render_interval = 2*self.decimation
        self.episode_length_s = 10.0

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