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
class HECTORFlatEnvBlindLocomotionSACCfg(LocomotionVelocityRoughEnvCfg):
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
        self.scene.terrain = hector_mdp.BaseTerrain

        # event (disable on plane)
        self.events.reset_terrain_type = None
        self.curriculum.terrain_levels = None

        # sensor (disable for blind locomotion)
        self.scene.height_scanner = None
        self.scene.height_scanner_L_foot = None
        self.scene.height_scanner_R_foot = None
        self.observations.exteroception = None
        
        self.viewer = ViewerCfg(
            eye=(-0.0, -2.0, -0.2), 
            lookat=(0.0, -0.8, -0.2),
            resolution=(3840, 2160), # 4K
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
            "yaw": (0, 0),
        }

        # light setting
        self.scene.sky_light.init_state.rot = (0.8660254, 0.0, 0.0, 0.5)  # yaw=60deg
        
        # command 
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5) # fixed vel
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

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
        # self.scene.terrain = hector_mdp.CurriculumSteppingStoneTerrain

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

        # light setting
        # self.scene.sky_light.init_state.rot = (0.8660254, 0.0, 0.0, 0.5)  # yaw=60deg
        self.scene.sky_light.init_state.rot = (0.8660254, 0.5, 0.0, 0.0)  # roll=60deg
        
        # command 
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5) # fixed vel
        # self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6) # uniform samplilng
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

@configclass
class HECTORRoughEnvBlindLocomotionSACCfgPLAY(HECTORRoughEnvBlindLocomotionSACCfg):
    """Playground environment configuration for HECTOR."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 100

        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 2*self.decimation

        # terrain
        # self.scene.terrain = hector_mdp.InferenceSteppingStoneTerrain
        self.scene.terrain = hector_mdp.RegularStair
        # self.scene.terrain = hector_mdp.RandomStair
        # self.scene.terrain = hector_mdp.RandomBlock
        
        # event 
        # self.events.reset_terrain_type = None
        self.curriculum.terrain_levels = None

        self.events.reset_base.func=hector_mdp.reset_root_state_orthogonal
        self.events.reset_base.params["multiplier"] = 2
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
            # "yaw": (math.pi/2, math.pi/2),
        }

        # # spawn robot close to stair
        # self.events.reset_base.params["pose_range"] = {
        #     # "x": (0.675, 0.675), 
        #     "x": (0.65, 0.65), 
        #     "y": (0.4, 0.4),  
        #     "z": (0.0, 0.0),
        #     "roll": (0.0, 0.0),
        #     "pitch": (0.0, 0.0),
        #     "yaw": (0.0, 0.0),
        # }

        # command
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6)
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.debug_vis = False

        # termination 
        self.terminations.terrain_out_of_bounds.params["distance_buffer"] = 0.125

        # rendering optimization 
        RECORDING = True

        if RECORDING:
            # quality rendering
            self.viewer = ViewerCfg(
                # eye=(0.0, -2.0, 0.4), 
                # lookat=(0.0, -0.5, 0.1),
                eye=(-0.0, -1.4, -0.2), 
                lookat=(0.0, -0.8, -0.2),
                # resolution=(3840, 2160), # 4K
                resolution=(1920, 1080), 
                origin_type="asset_root", 
                asset_name="robot"
            )
            # self.sim.render_interval = self.decimation
            # self.sim.render.dlss_mode = 2 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            # self.sim.render.antialiasing_mode = "DLSS" # "Off", "FXAA", "DLSS", "TAA", "DLAA"
            self.sim.render.dlss_mode = 0 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            self.sim.render.antialiasing_mode = None # "Off", "FXAA", "DLSS", "TAA", "DLAA"

        else:
            # performance rendering
            self.sim.render.dlss_mode = 0 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            self.sim.render.antialiasing_mode = None # "Off", "FXAA", "DLSS", "TAA", "DLAA"

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

        self.scene.terrain = hector_mdp.RegularStair

        self.events.reset_base.func=hector_mdp.reset_root_state_orthogonal
        self.events.reset_base.params["multiplier"] = 2
        self.events.reset_base.params["pose_range"] = {
            # "x": (-0.3, 0.3), 
            # "y": (-0.3, 0.3), 
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
        self.events.reset_terrain_type = None

        # debug vis
        # self.scene.height_scanner_L_foot.debug_vis = True
        # self.scene.height_scanner_R_foot.debug_vis = True
        
        self.curriculum.terrain_levels = None
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)


        # light setting
        self.scene.sky_light.init_state.rot = (0.9238795, 0.0, 0.0, -0.3826834)

        # rendering optimization 
        RECORDING = False

        if RECORDING:
            # quality rendering
            self.viewer = ViewerCfg(
                eye=(0.0, -2.0, 0.4), 
                lookat=(0.0, -0.5, 0.1),
                # eye=(-0.0, -1.5, -0.15), 
                # lookat=(0.0, -0.8, -0.15),
                resolution=(3840, 2160), # 4K
                origin_type="asset_root", 
                asset_name="robot"
            )
            # self.sim.render_interval = self.decimation
            self.sim.render.dlss_mode = 2 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            self.sim.render.antialiasing_mode = "DLSS" # "Off", "FXAA", "DLSS", "TAA", "DLAA"

        else:
            # performance rendering
            self.sim.render.dlss_mode = 0 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            self.sim.render.antialiasing_mode = None # "Off", "FXAA", "DLSS", "TAA", "DLAA"