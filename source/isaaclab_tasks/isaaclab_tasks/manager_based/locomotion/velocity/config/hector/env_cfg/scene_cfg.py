# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

# from isaaclab.envs.common import ViewerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.hector import HECTOR_CFG

@configclass
class HECTORSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = hector_mdp.CurriculumSteppingStoneTerrain
    
    # gravel 
    # gravel = hector_mdp.GravelTerrain
    
    # --robots
    robot: ArticulationCfg = HECTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # --sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/10,
    )
    height_scanner.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RayCaster",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    
    height_scanner_fine = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/10,
    )
    height_scanner_fine.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RayCasterFine",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    
    
    # height_scanner_L_foot = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/L_toe",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.14+0.04, 0.07]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    #     update_period=1/100,
    # )
    # height_scanner_L_foot.visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/RayCasterLFoot",
    #     markers={
    #         "hit": sim_utils.SphereCfg(
    #             radius=0.01,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    #         ),
    #     },
    # )
    
    # height_scanner_R_foot = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/R_toe",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.14+0.04, 0.07]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    #     update_period=1/100,
    # )
    # height_scanner_R_foot.visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/RayCasterRFoot",
    #     markers={
    #         "hit": sim_utils.SphereCfg(
    #             radius=0.01,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    #         ),
    #     },
    # )
    
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        # track_air_time=True,
        track_air_time=False,
        update_period=1/100,
        debug_vis=True,
        )
    contact_forces.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ContactSensor",
        markers={
            "contact": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.8)),
            ),
            "no_contact": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.8)),
                visible=False,
            ),
        },
    )
    
    toe_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_toe_tip",
        history_length=3,
        # track_air_time=True,
        track_air_time=False,
        update_period=1/100,
        )
    toe_contact.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ContactSensorToe",
        markers={
            "contact": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.8)),
            ),
            "no_contact": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.8)),
                visible=False,
            ),
        },
    )
    
    # --lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )