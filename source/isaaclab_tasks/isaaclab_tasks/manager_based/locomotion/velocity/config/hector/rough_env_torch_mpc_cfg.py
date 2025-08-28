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
    HECTORTorchBlindLocomotionActionsCfg, 
    HECTORTorchBlindLocomotionObservationsCfg,
    HECTORTorchBlindLocomotionRewardsCfg,
    HECTORCommandsCfg,
    HECTORCurriculumCfg,
    HECTOREventCfg,
    HECTORTerminationsCfg,
    HECTORBlindLocomotionSceneCfg,
)

@configclass
class HECTORTorchRoughEnvBlindLocomotionSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORBlindLocomotionSceneCfg = HECTORBlindLocomotionSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: HECTORTorchBlindLocomotionObservationsCfg = HECTORTorchBlindLocomotionObservationsCfg()
    rewards: HECTORTorchBlindLocomotionRewardsCfg = HECTORTorchBlindLocomotionRewardsCfg()
    actions: HECTORTorchBlindLocomotionActionsCfg = HECTORTorchBlindLocomotionActionsCfg()
    commands: HECTORCommandsCfg = HECTORCommandsCfg()
    terminations: HECTORTerminationsCfg = HECTORTerminationsCfg()
    events: HECTOREventCfg = HECTOREventCfg()
    curriculum: HECTORCurriculumCfg = HECTORCurriculumCfg()
    seed = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # sim time
        # self.sim.dt = 1/400
        # self.decimation = 4
        self.sim.dt = 1/200
        self.decimation = 2
        self.sim.render_interval = 2*self.decimation

        # terain
        self.scene.terrain = hector_mdp.BaseTerrain 
        # self.scene.terrain = hector_mdp.InferenceSteppingStoneTerrain

        # event (disable on plane)
        self.events.reset_terrain_type = None
        self.curriculum.terrain_levels = None

        # sensor
        self.scene.height_scanner = None
        
        # self.viewer = ViewerCfg(
        #     # eye=(-0.0, -1.5, -0.2), 
        #     # lookat=(0.0, -0.8, -0.2),
        #     # resolution=(3840, 2160), # 4K
        #     # eye=(0.0, -4.0, 0.4), 
        #     # lookat=(0.0, -0.5, 0.1),
        #     eye=(0.0, -3.0, 0.4), 
        #     lookat=(-1.0, -0.5, 0.1),
        #     resolution=(1920, 1080), 
        #     origin_type="asset_root", 
        #     asset_name="robot"
        # )

        # event 
        self.events.reset_base.params["pose_range"] = {
            # "x": (-0.5, 0.5), 
            # "y": (-0.5, 0.5), 
            "x": (-0.3, 0.3), 
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
            # "yaw": (-0.0, 0.0),
        }

        # light setting
        self.scene.sky_light.init_state.rot = (0.8660254, 0.0, 0.0, 0.5)  # yaw=60deg
        
        # command 
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.debug_vis = False