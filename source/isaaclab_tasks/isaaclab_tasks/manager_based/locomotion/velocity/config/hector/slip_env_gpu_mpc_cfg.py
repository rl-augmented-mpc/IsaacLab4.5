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
    HECTORGPUSlipActionsCfg,
    HECTORGPUBlindLocomotionObservationsCfg,
    HECTORGPUBlindLocomotionRewardsCfg,
    HECTORCommandsCfg,
    HECTORCurriculumCfg,
    HECTOREventCfg,
    HECTORTerminationsCfg,
    HECTORBlindLocomotionSceneCfg,
)

@configclass
class HECTORGPUSlipEnvSACCfg(LocomotionVelocityRoughEnvCfg):
    scene: HECTORBlindLocomotionSceneCfg = HECTORBlindLocomotionSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: HECTORGPUBlindLocomotionObservationsCfg = HECTORGPUBlindLocomotionObservationsCfg()
    rewards: HECTORGPUBlindLocomotionRewardsCfg = HECTORGPUBlindLocomotionRewardsCfg()
    actions: HECTORGPUSlipActionsCfg = HECTORGPUSlipActionsCfg()
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
        self.episode_length_s = 10.0

        # terain
        self.scene.terrain = hector_mdp.CurriculumFrictionPatchTerrain

        # event (disable on plane)
        self.events.reset_terrain_type = None

        # sensor
        self.scene.height_scanner = None
        
        self.viewer = ViewerCfg(
            eye=(0.0, -4.0, 0.5), 
            lookat=(0.0, -0.5, 0.1),
            resolution=(1920, 1080), 
            origin_type="asset_root", 
            asset_name="robot"
        )

        # event 
        self.events.reset_base.params["pose_range"] = {
            "x": (-2.0, 2.0), 
            "y": (-2.0, 2.0), 
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }

        # light setting
        self.scene.sky_light.init_state.rot = (0.8660254, 0.0, 0.0, 0.5)  # yaw=60deg
        
        # command 
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.commands.base_velocity.debug_vis = True


        # termination 
        self.terminations.terrain_out_of_bounds.params["distance_buffer"] = 0.125

@configclass
class HECTORGPUSlipEnvSACCfgPLAY(HECTORGPUSlipEnvSACCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.seed = 100

        # sim time
        self.sim.dt = 1/400
        self.decimation = 4
        self.sim.render_interval = 2*self.decimation

        # solver
        # self.actions.mpc_action.solver_name = "osqp"
        # self.actions.mpc_action.solver_name = "qpth"
        # self.actions.mpc_action.solver_name = "casadi"
        self.actions.mpc_action.solver_name = "cusadi"

        # terrain
        self.seed = 42
        self.scene.terrain = hector_mdp.InferenceAlternatingFrictionPatchTerrain
        self.curriculum.terrain_levels = None
        
        # command 
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        # self.commands.base_velocity.debug_vis = False

        # termination 
        self.terminations.terrain_out_of_bounds.params["distance_buffer"] = 0.125


        # events
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.25, 0.25), 
            # "x": (-0.3-0.25, 0.3-0.25), 
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
              
            # "yaw": (-0, 0),
            "yaw": (-math.pi/6, math.pi/6),
        }
        self.events.reset_terrain_type = None

        # rendering optimization 
        RECORDING = True
        if RECORDING:
            # quality rendering
            self.viewer = ViewerCfg(
                eye=(-0.0, -1.4, -0.2), 
                lookat=(0.0, -0.8, -0.2),
                resolution=(3840, 2160), # 4K
                origin_type="asset_root", 
                asset_name="robot"
            )
            self.sim.render_interval = self.decimation
            self.sim.render.dlss_mode = 2 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            self.sim.render.antialiasing_mode = "DLSS" # "Off", "FXAA", "DLSS", "TAA", "DLAA"

        else:
            # performance rendering
            self.sim.render.dlss_mode = 0 # 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto)
            self.sim.render.antialiasing_mode = None # "Off", "FXAA", "DLSS", "TAA", "DLAA"