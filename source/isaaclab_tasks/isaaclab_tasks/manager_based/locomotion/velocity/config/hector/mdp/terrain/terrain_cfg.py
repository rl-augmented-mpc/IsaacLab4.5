import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

"""
base flat terrain.
"""

# quat = Rotation.from_euler('xyz', [0, -5, 0], degrees=True).as_quat().astype(np.float32).tolist()
BaseTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(30.0, 30.0), # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "terrain1": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    debug_vis=False,
    disable_colllider=False,
    # center_orientation=(quat[3], quat[0], quat[1], quat[2]),
)

"""
variable friction terrain.
"""

CurriculumFrictionPatchTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="custom_curriculum",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(1.0, 1.0), # size of sub-terrain
        border_width=0.0,
        num_rows=10,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
        },
        num_sub_patches=26,
        custom_curriculum=True,
    ),
    collision_group=-1,
    max_init_terrain_level=0,
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
    static_friction_range=(0.05, 0.5),
    friction_distribution="grid",
)


FrictionPatchTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="custom_curriculum",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(1.0, 4.0), # size of sub-terrain
        border_width=0.0,
        num_rows=10,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
        },
        num_sub_patches=5,
        custom_curriculum=True,
    ),
    collision_group=-1,
    max_init_terrain_level=0,
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
    static_friction_range=(0.1, 0.1),
    friction_distribution="grid",
)

InferenceAlternatingFrictionPatchTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="custom_curriculum",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(1.0, 2.0), # size of sub-terrain
        border_width=0.0,
        num_rows=4,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0),
        },
        num_sub_patches=6,
        custom_curriculum=True,
    ),
    collision_group=-1,
    max_init_terrain_level=0,
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
    static_friction_range=(0.07, 0.5),
    friction_distribution="grid_deterministic",
)


"""
rough terrain.
"""

SteppingStoneTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(10.0, 10.0), # size of sub-terrain
        border_width=5.0,
        num_rows=1,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            # "stair": terrain_gen.StairTerrainCfg(
            # profile_mode="random",
            # proportion=0.5, 
            # num_box=120,
            # box_height_range=(0.1, 0.1), 
            # platform_width=30.0, 
            # platform_length_range_start=(0.2, 0.3), platform_length_range_end=(0.2, 0.3),
            # platform_gap_range_start=(0.0, 0.0), platform_gap_range_end=(0.0, 0.0),
            # border_size=0.0, 
            # height_noise_range=(-0.01, 0.01), 
            # center_area_size=1.5,
            # ),

            # "repeated_object": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            #     object_type="box", 
            #     max_height_noise=0.00, 
            #     platform_width=1.8,
            #     proportion=0.05,
            #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            #         num_objects=2000, 
            #         height=0.1*2, 
            #         size=(0.3, 0.3),
            #         max_yx_angle=0.0,
            #     ), 
            #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            #         num_objects=2000, 
            #         height=0.1*2, 
            #         size=(0.3, 0.3),
            #         max_yx_angle=0.0, 
            #     ),
            # ), 

            "random_grid_narrow": terrain_gen.MeshRandomBlockTerrainCfg(
            proportion=0.25, grid_width=0.2, 
            # grid_height_range=(0.06, 0.06), 
            grid_height_range=(0.07, 0.07), 
            platform_width=2.0
            ),

            "random_grid": terrain_gen.MeshRandomBlockTerrainCfg(
            proportion=0.25, grid_width=0.3, 
            # grid_height_range=(0.06, 0.06), 
            grid_height_range=(0.07, 0.07), 
            platform_width=2.1
            ),

            "random_grid_wide": terrain_gen.MeshRandomBlockTerrainCfg(
            proportion=0.25, grid_width=0.4, 
            # grid_height_range=(0.06, 0.06), 
            grid_height_range=(0.07, 0.07), 
            platform_width=2.0
            ),

            # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            # proportion=0.05,
            # # step_height_range=(0.1, 0.1),
            # step_height_range=(0.11, 0.11),
            # step_width=0.2,
            # platform_width=2.5,
            # border_width=1.0,
            # holes=False,
            # ),

            "pyramid_stairs_inv_1": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.11, 0.11),
            # step_height_range=(0.1, 0.1),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            ),

            "pyramid_stairs_inv_2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.11, 0.11),
            # step_height_range=(0.1, 0.1),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            ),

            "random_stair": terrain_gen.StairTerrainCfg(
            profile_mode="random",
            proportion=0.3, 
            num_box=50, # one side
            # box_height_range=(0.1, 0.1), 
            box_height_range=(0.11, 0.11), 
            platform_width=10.0, 
            platform_length_range_start=(0.2, 0.25), platform_length_range_end=(0.2, 0.25),
            platform_gap_range_start=(0.0, 0.0), platform_gap_range_end=(0.0, 0.0),
            border_size=0.0, 
            # height_noise_range=(-0.01, 0.01), 
            height_noise_range=(-0.0, 0.0), 
            center_area_size=1.5,
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    max_init_terrain_level=0,
    keep_max_terrain_level=True,
    debug_vis=False,
    disable_colllider=False,
)


InferenceSteppingStoneTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(6.0, 6.0), # size of sub-terrain
        # size=(8.0, 8.0), # size of sub-terrain
        # size=(10.0, 10.0), # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75, 
        use_cache=False,
        curriculum=True,
        sub_terrains={
            # "repeated_object": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            #     object_type="box", 
            #     max_height_noise=0.00, 
            #     platform_width=1.0,
            #     proportion=0.05,
            #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            #         num_objects=130, 
            #         height=0.1*2, 
            #         size=(0.3, 0.3),
            #         max_yx_angle=0.0,
            #     ), 
            #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            #         num_objects=130, 
            #         height=0.1*2, 
            #         size=(0.3, 0.3),
            #         max_yx_angle=0.0, 
            #     ),
            # ), 

            # "random_grid_narrow": terrain_gen.MeshRandomBlockTerrainCfg(
            # proportion=0.33, grid_width=0.25, 
            # # grid_height_range=(0.03, 0.03), 
            # # grid_height_range=(0.045, 0.045), 
            # # grid_height_range=(0.07, 0.07), 
            # grid_height_range=(0.08, 0.08), 
            # platform_width=1.25, 
            # # uniform_noise_range=(0.8, 1.0),
            # ),

            # "random_grid": terrain_gen.MeshRandomBlockTerrainCfg(
            # proportion=0.33, grid_width=0.3, 
            # # grid_height_range=(0.03, 0.03), 
            # # grid_height_range=(0.04, 0.04), 
            # grid_height_range=(0.07, 0.07), 
            # platform_width=1.5, 
            # uniform_noise_range=(0.0, 1.0),
            # ),

            # "random_wide": terrain_gen.MeshRandomBlockTerrainCfg(
            # proportion=0.33, grid_width=0.45, 
            # # grid_height_range=(0.03, 0.03), 
            # # grid_height_range=(0.04, 0.04), 
            # grid_height_range=(0.07, 0.07), 
            # platform_width=1.8
            # ),

            # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            # proportion=0.05,
            # step_height_range=(0.1, 0.1),
            # step_width=0.2,
            # platform_width=1.5,
            # border_width=0.0,
            # holes=False,
            # ),

            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            # step_height_range=(0.06, 0.06),
            # step_height_range=(0.07, 0.07),
            step_height_range=(0.08, 0.08),
            # step_height_range=(0.09, 0.09),
            # step_height_range=(0.1, 0.1),
            # step_height_range=(0.11, 0.11),
            # step_height_range=(0.115, 0.115),
            # step_height_range=(0.12, 0.12),
            step_width=0.25,
            platform_width=2.0,
            border_width=0.0,
            holes=False,
            ),

            # "random_stair": terrain_gen.StairTerrainCfg(
            # profile_mode="random",
            # proportion=0.5, 
            # num_box=50, # one side
            # # box_height_range=(0.09, 0.09), 
            # # box_height_range=(0.1, 0.1), 
            # box_height_range=(0.11, 0.11), 
            # platform_width=6.0, 
            # platform_length_range_start=(0.25, 0.25), platform_length_range_end=(0.25, 0.25),
            # platform_gap_range_start=(0.0, 0.0), platform_gap_range_end=(0.0, 0.0),
            # border_size=0.0, 
            # # height_noise_range=(-0.01, 0.01), 
            # height_noise_range=(-0.0, 0.0), 
            # center_area_size=1.0,
            # ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #     project_uvw=True,
    #     texture_scale=(0.25, 0.25),
    # ),
    debug_vis=False,
    disable_colllider=False,
)


"""
continuous rough terrain.
"""

sub_terrains = {}
slop_range = (0.2, 0.8)
curriculum_x = 3
curriculum_y = 3
nx = 60
ny = 60
for i in range(nx*ny):
    i_x = i // ny
    i_y = i % ny
    
    local_slope_range = (slop_range[0] + i * (slop_range[1] - slop_range[0]) / (nx*ny), slop_range[0] + (i+1) * (slop_range[1] - slop_range[0]) / (nx*ny))
    inv = int(np.random.randint(0, 2))
    inv = True if inv == 1 else False
    
    # place flat terrain at each local center
    if (i_x % (nx//curriculum_x) >= (nx//curriculum_x)//2 -1 and i_x% (nx//curriculum_x) <= (nx//curriculum_x)//2) \
        and (i_y % (ny//curriculum_y) >= (ny//curriculum_y)//2 -1 and i_y% (ny//curriculum_y) <= (ny//curriculum_y)//2):
        sub_terrains[f"flat{i}"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, height=0.0, size=(0.5, 0.5))
    else:
        sub_terrains[f"box{i}"] = terrain_gen.HfPyramidSlopedTerrainCfg(
            size=(0.5, 0.5),
            slope_range=local_slope_range, 
            platform_width=0.05, 
            border_width=0.03,
            inverted=inv
            )

PyramidHfTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(0.5, 0.5), # size of sub-terrain
        border_width=0.0,
        num_rows=ny,
        num_cols=nx,
        horizontal_scale=0.03,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    debug_vis=False,
    disable_colllider=False,
)


FractalTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(10.0, 10.0), # size of sub-terrain
        border_width=0.0,
        num_rows=4,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "terrain1": terrain_gen.HfFractalTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.2),
            ),
            "terrain2": terrain_gen.HfFractalTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.3),
            ),
            "terrain3": terrain_gen.HfFractalTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.4),
            ),
            "terrain4": terrain_gen.HfFractalTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.5),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    debug_vis=False,
    disable_colllider=False,
)


"""
soft contact terrain.
"""

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
    # center_position=(50.0, 50.0, 0.0)
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
)