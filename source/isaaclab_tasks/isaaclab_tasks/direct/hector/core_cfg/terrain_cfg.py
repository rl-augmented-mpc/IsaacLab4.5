from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

CurriculumFrictionPatchTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="patched",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(1.0, 1.0), # size of sub-terrain
        border_width=0.0,
        num_rows=20*3,
        num_cols=20*3,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
    disable_colllider=False,
    static_friction_range = (0.2, 0.3),
    friction_group_patch_num = 20,
    num_curriculums=9,
)


FrictionPatchTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="patched",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(1.0, 1.0), # size of sub-terrain
        border_width=0.0,
        num_rows=20*1,
        num_cols=20*1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
    disable_colllider=False,
    # static_friction_range = (0.21, 0.21),
    static_friction_range=(0.4, 0.5),
    # static_friction_range=(0.25, 0.25),
    friction_group_patch_num = 20,
    num_curriculums=1,
)


BaseTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(60.0, 60.0), # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
            # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=-0.07), # ground level
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
    disable_colllider=False,
    # center_position=(30.0, 30.0, 0.0)
)

SoftVisualTerrain = TerrainImporterCfg(
    prim_path="/World/soft_ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(100.0, 100.0), # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "box": terrain_gen.MeshThickTerrainCfg(
                proportion=1.0, thickness=0.16),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.2)),
    visual_material=sim_utils.MdlFileCfg(
            mdl_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.2/NVIDIA/Materials/Base/Natural/Dirt.mdl",
            project_uvw=True,
            texture_scale=(1.0, 1.0),
        ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #     project_uvw=True,
    #     texture_scale=(0.25, 0.25),
    # ),
    debug_vis=False,
    disable_colllider=True,
    static_friction_range = (1.0, 1.0),
    center_position=(50.0, 50.0, 0.0)
)

RigidTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(0.1, 100.0), # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0),
            # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=-0.07), # ground level
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
    disable_colllider=True,
    center_position=(50.0, 50.0, 0.0)
)