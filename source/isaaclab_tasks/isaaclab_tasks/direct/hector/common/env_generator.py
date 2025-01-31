# from typing import Tuple, List
# from dataclasses import dataclass
# import numpy as np
# import torch

# import isaaclab.sim as sim_utils
# from isaaclab_assets import StaticColliderObjectCfg

# from .sampler import GridCubicSampler, UniformLineSampler, UniformCubicSampler, QuaternionSampler
# from .curriculum import CurriculumRateSampler, CurriculumUniformLineSampler

# """
# Implementation of gravel object creation. 
# It's ugly, but it works.
# """

# class SphereClusterGenerator:
#     def __init__(self, prim_path:str,
#                  name:str,
#                  friction_coeff:float,  
#                  position_sampler_conf:GridCubicSampler, 
#                  yaw_angle_sampler_conf:QuaternionSampler,
#                  radius_sampler_conf:UniformLineSampler,
#                  mass_sampler_conf:CurriculumUniformLineSampler):
#         self.prim_path = prim_path
#         self.name = name
#         self.position_sampler = position_sampler_conf
#         self.yaw_angle_sampler = yaw_angle_sampler_conf
#         self.radius_sampler = radius_sampler_conf
#         self.mass_sampler = mass_sampler_conf
#         self.friction_coeff = friction_coeff
        
#     def sample_geometry_data(self)->tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]], list[float], list[float]]:
#         positions = self.position_sampler.sample()
#         quat = self.yaw_angle_sampler.sample(len(positions))
#         radius = self.radius_sampler.sample(len(positions))
#         masses = self.mass_sampler.sample(0, len(positions))
#         return positions, quat, radius, masses
    
#     def get_cfg(self)->StaticColliderObjectCfg:
#         p, q, radius, mass = self.sample_geometry_data()
#         print(f"Generated {len(p)} particles")
#         is_ = StaticColliderObjectCfg.InitialStateCfg(pos=[p], rot=[q])
#         collider_cfg = StaticColliderObjectCfg(
#             prim_path=self.prim_path,
#             spawn=sim_utils.ManyAssetSpawnerCfg(
#                 assets_cfg=[sim_utils.SphereCfg(
#                     radius=radius[0],
#                     # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.0)),
#                     physics_material=sim_utils.RigidBodyMaterialCfg(
#                                         friction_combine_mode="average",
#                                         restitution_combine_mode="average",
#                                         static_friction=self.friction_coeff,
#                                         dynamic_friction=self.friction_coeff,
#                                     ),
#                 )],
#                 rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0, 
#                                                              max_linear_velocity=8.0, max_angular_velocity=1.0, max_depenetration_velocity=8.0),
#                 mass_props=sim_utils.MassPropertiesCfg(mass=mass[0]),
#                 collision_props=sim_utils.CollisionPropertiesCfg(),
#                 num_assets=[len(p)]
#                 ),
#             init_state=is_
#             )
#         return collider_cfg



# class GravelClusterGenerator:
#     def __init__(self, prim_path:str, 
#                  usd_path:str, 
#                  name:str,
#                  position_sampler_conf:GridCubicSampler, 
#                  yaw_angle_sampler_conf:QuaternionSampler,
#                  scale_sampler_conf:UniformCubicSampler,
#                  mass_sampler_conf:CurriculumUniformLineSampler):
#         self.prim_path = prim_path
#         self.usd_path = usd_path
#         self.name = name
#         self.position_sampler = position_sampler_conf
#         self.yaw_angle_sampler = yaw_angle_sampler_conf
#         self.scale_sampler = scale_sampler_conf
#         self.mass_sampler = mass_sampler_conf
        
#     def sample_geometry_data(self)->tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]], list[tuple[float, float, float]], list[float]]:
#         positions = self.position_sampler.sample()
#         quat = self.yaw_angle_sampler.sample(len(positions))
#         scales = self.scale_sampler.sample(len(positions))
#         masses = self.mass_sampler.sample(0, len(positions))
#         return positions, quat, scales, masses
    
#     def get_cfg(self)->StaticColliderObjectCfg:
#         p, q, scale, mass = self.sample_geometry_data()
#         is_ = StaticColliderObjectCfg.InitialStateCfg(pos=[p], rot=[q], scale=[scale])
#         collider_cfg = StaticColliderObjectCfg(
#             prim_path=self.prim_path,
#             spawn=sim_utils.ManyAssetSpawnerCfg(
#                 assets_cfg=[sim_utils.UsdFileCfg(
#                     usd_path=self.usd_path,
#                     scale=scale[0],
#                 )],
#                 rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
#                 mass_props=sim_utils.MassPropertiesCfg(mass=mass[0]),
#                 collision_props=sim_utils.CollisionPropertiesCfg(),
#                 num_assets=[len(p)]
#                 ),
#             init_state=is_
#             )
#         return collider_cfg