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
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.hector import HECTOR_CFG

@configclass
class HECTORBlindLocomotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = hector_mdp.CurriculumSteppingStoneTerrain
    
    # # gravel 
    # terrain = hector_mdp.BaseTerrain
    # gravel = hector_mdp.GravelTerrain
    
    # --robots
    robot: ArticulationCfg = HECTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # -- sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/10,
    )
    height_scanner.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RayCaster",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 1.0, 0.4)),
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        # track_air_time=True,
        track_air_time=False,
        update_period=1/100,
        debug_vis=False,
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
    
    # toe_contact = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*_toe_tip",
    #     history_length=3,
    #     # track_air_time=True,
    #     track_air_time=False,
    #     update_period=1/100,
    #     )
    # toe_contact.visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/ContactSensorToe",
    #     markers={
    #         "contact": sim_utils.SphereCfg(
    #             radius=0.03,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.8)),
    #         ),
    #         "no_contact": sim_utils.SphereCfg(
    #             radius=0.03,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.8)),
    #             visible=False,
    #         ),
    #     },
    # )
    
    # --lights
    # ## gray studio ##
    # distant_light = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(
    #         intensity=3000.0,
    #         angle=34.3
    #     ),
    # )
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=1003.29999,
    #         exposure=0.4, 
            
    #     ),
    # )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class HECTORPerceptiveLocomotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = hector_mdp.CurriculumSteppingStoneTerrain
    
    # # gravel 
    # terrain = hector_mdp.BaseTerrain
    # gravel = hector_mdp.GravelTerrain
    
    # --robots
    robot: ArticulationCfg = HECTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # --sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/10,
    )
    height_scanner.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RayCaster",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 1.0, 0.4)),
            ),
        },
    )
    
    # height_scanner_fine = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.0, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    #     update_period=1/10,
    # )
    # height_scanner_fine.visualizer_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/RayCasterFine",
    #     markers={
    #         "hit": sim_utils.SphereCfg(
    #             radius=0.01,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 1.0, 0.4)),
    #         ),
    #     },
    # )
    
    
    height_scanner_L_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L_toe",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.0)),
        # offset=RayCasterCfg.OffsetCfg(pos=(0.02, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.14+0.04, 0.08]),
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.14, 0.07]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/100,
    )
    height_scanner_L_foot.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RayCasterLFoot",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    
    height_scanner_R_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_toe",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.0)),
        # offset=RayCasterCfg.OffsetCfg(pos=(0.02, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.14+0.04, 0.08]),
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.14, 0.07]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=1/100,
    )
    height_scanner_R_foot.visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RayCasterRFoot",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    
    # tiled_d455 = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=1/30,
    #     height=480,
    #     width=640,
    #     data_types=["distance_to_camera"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=1.93, 
    #         focus_distance=0.6, 
    #         horizontal_aperture=3.896, 
    #         vertical_aperture=2.453, 
    #         clipping_range=(0.01, 20.0)),
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.12, 0.0, -0.1), 
    #         # rot=(0.5609855,0.4304593, -0.4304593, -0.5609855), # pitch=15deg
    #         # rot=(0.5963678, 0.3799282, -0.3799282, -0.5963678), # pitch=25deg
    #         # rot=(0.6123724, 0.3535534, -0.3535534, -0.6123724), # pitch=30deg
    #         # rot=(0.6532815, 0.2705981, -0.2705981, -0.6532815), # pitch=45deg
    #         rot=(0.6830127, 0.1830127, -0.1830127, -0.6830127), # pitch=60deg
    #         # rot=(0.6963642, 0.1227878, -0.1227878, -0.6963642), # pitch=70deg
    #         convention="opengl"),
    # )
    
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        # track_air_time=True,
        track_air_time=False,
        update_period=1/100,
        debug_vis=False,
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
    ## gray studio ##
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            angle=34.3
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1003.29999,
            exposure=0.4, 
            
        ),
    )
    
@configclass
class HECTORSlipSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = hector_mdp.CurriculumFrictionPatchTerrain
    
    # gravel 
    # gravel = hector_mdp.GravelTerrain
    
    # --robots
    robot: ArticulationCfg = HECTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # --sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground_visual"], 
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
    
    # --lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )