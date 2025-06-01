from typing import Tuple, List
import math
from dataclasses import dataclass, MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import StaticColliderObjectCfg
from .util import (
    GridCubicSampler, 
    UniformLineSampler, 
    UniformCubicSampler, 
    EulertoQuaternionSampler,
)

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR # type: ignore

class SphereTerrainGenerator:
    def __init__(self, 
                 prim_path:str,
                 friction_coeff:float,  
                 position_x_range:Tuple[float, float],
                 position_y_range:Tuple[float, float],
                 position_z_range:Tuple[float, float],
                 position_x_resolution:float,
                 position_y_resolution:float,
                 position_z_resolution:float,
                 angle_x_range: Tuple[float, float],
                 angle_y_range: Tuple[float, float],
                 angle_z_range: Tuple[float, float],
                 randius_range: Tuple[float, float],
                 mass_range: Tuple[float, float]
                 ):
        
        self.prim_path = prim_path
        self.friction_coeff = friction_coeff
        
        self.position_sampler = GridCubicSampler(
            x_range=position_x_range,
            y_range=position_y_range,
            z_range=position_z_range,
            x_resolution=position_x_resolution,
            y_resolution=position_y_resolution,
            z_resolution=position_z_resolution
        )
        self.yaw_angle_sampler = EulertoQuaternionSampler(
            x_range=angle_x_range,
            y_range=angle_y_range,
            z_range=angle_z_range
        )
        self.radius_sampler = UniformLineSampler(
            x_range=randius_range,
        )
        self.mass_sampler = UniformLineSampler(
            x_range=mass_range,
        )
        
    def sample_geometry_data(self)->tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]], list[float], list[float]]:
        positions = self.position_sampler.sample()
        quat = self.yaw_angle_sampler.sample(len(positions))
        radius = self.radius_sampler.sample(len(positions))
        masses = self.mass_sampler.sample(len(positions))
        return positions, quat, radius, masses
    
    def get_cfg(self)->StaticColliderObjectCfg:
        p, q, radius, mass = self.sample_geometry_data()
        print(f"[INFO] Generated {len(p)} instances.")
        is_ = StaticColliderObjectCfg.InitialStateCfg(pos=[p], rot=[q])
        collider_cfg = StaticColliderObjectCfg(
            prim_path=self.prim_path,
            spawn=sim_utils.ManyAssetSpawnerCfg(
                assets_cfg=[sim_utils.SphereCfg(
                    radius=radius[0],
                    # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.0)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                                        friction_combine_mode="average",
                                        restitution_combine_mode="average",
                                        static_friction=self.friction_coeff,
                                        dynamic_friction=self.friction_coeff,
                                    ),
                )],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0, 
                                                             max_linear_velocity=8.0, max_angular_velocity=1.0, max_depenetration_velocity=8.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass[0]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                num_assets=[len(p)]
                ),
            init_state=is_
            )
        return collider_cfg



class AssetTerrainGenerator:
    def __init__(self, 
                 prim_path:str,
                 usd_path:str,
                 friction_coeff:float,  
                 position_x_range:Tuple[float, float],
                 position_y_range:Tuple[float, float],
                 position_z_range:Tuple[float, float],
                 position_x_resolution:float,
                 position_y_resolution:float,
                 position_z_resolution:float,
                 angle_x_range: Tuple[float, float],
                 angle_y_range: Tuple[float, float],
                 angle_z_range: Tuple[float, float],
                 scale_x_range: Tuple[float, float],
                 scale_y_range: Tuple[float, float],
                 scale_z_range: Tuple[float, float],
                 mass_range: Tuple[float, float]
                 ):
        
        self.prim_path = prim_path
        self.usd_path = usd_path
        self.friction_coeff = friction_coeff
        
        self.position_sampler = GridCubicSampler(
            x_range=position_x_range,
            y_range=position_y_range,
            z_range=position_z_range,
            x_resolution=position_x_resolution,
            y_resolution=position_y_resolution,
            z_resolution=position_z_resolution
        )
        self.yaw_angle_sampler = EulertoQuaternionSampler(
            x_range=angle_x_range,
            y_range=angle_y_range,
            z_range=angle_z_range
        )
        self.scale_sampler = UniformCubicSampler(
            x_range=scale_x_range,
            y_range=scale_y_range,
            z_range=scale_z_range
        )
        self.mass_sampler = UniformLineSampler(
            x_range=mass_range
        )
        
    def sample_geometry_data(self)->tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]], list[tuple[float, float, float]], list[float]]:
        positions = self.position_sampler.sample()
        quat = self.yaw_angle_sampler.sample(len(positions))
        scales = self.scale_sampler.sample(len(positions))
        masses = self.mass_sampler.sample(len(positions))
        return positions, quat, scales, masses
    
    def get_cfg(self)->StaticColliderObjectCfg:
        p, q, scale, mass = self.sample_geometry_data()
        print(f"[INFO] Generated {len(p)} instances.")
        is_ = StaticColliderObjectCfg.InitialStateCfg(pos=[p], rot=[q], scale=[scale])
        collider_cfg = StaticColliderObjectCfg(
            prim_path=self.prim_path,
            spawn=sim_utils.ManyAssetSpawnerCfg(
                assets_cfg=[sim_utils.UsdFileCfg(
                    usd_path=self.usd_path,
                    scale=scale[0],
                )],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass[0]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                num_assets=[len(p)]
                ),
            init_state=is_
            )
        return collider_cfg
    
    
    
sphere_terrain_generator = SphereTerrainGenerator(
    prim_path="{ENV_REGEX_NS}/StaticColliders",
    friction_coeff=1.0,
    position_x_range=(-10.0, 10.0),
    position_y_range=(-10.0, 10.0),
    position_z_range=(0.0, 5.0),
    position_x_resolution=0.1,
    position_y_resolution=0.1,
    position_z_resolution=0.1,
    angle_x_range=(-math.pi, math.pi),
    angle_y_range=(-math.pi, math.pi),
    angle_z_range=(-math.pi, math.pi),
    randius_range=(0.1, 1.0),
    mass_range=(1.0, 10.0)
)

asset_terrain_generator = AssetTerrainGenerator(
    prim_path="{ENV_REGEX_NS}/StaticColliders",
    usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/rock.usd",
    friction_coeff=1.0,
    position_x_range=(-2.0, 2.0),
    position_y_range=(-2.0, 2.0),
    position_z_range=(0.02, 0.03),
    position_x_resolution=0.06,
    position_y_resolution=0.06,
    position_z_resolution=0.06,
    angle_x_range=(-math.pi, math.pi),
    angle_y_range=(-math.pi, math.pi),
    angle_z_range=(-math.pi, math.pi),
    scale_x_range=(0.4, 0.6),
    scale_y_range=(0.4, 0.6),
    scale_z_range=(0.4, 0.6),
    mass_range=(1.0, 1.5)
)

GravelTerrain = asset_terrain_generator.get_cfg()
ParticleTerrain = sphere_terrain_generator.get_cfg()