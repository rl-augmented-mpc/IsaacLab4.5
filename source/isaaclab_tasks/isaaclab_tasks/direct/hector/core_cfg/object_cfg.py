# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
# from isaaclab.assets import StaticColliderObjectCfg
# from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


# ## Base Plane ##

# GROUND_SPAWN_CFG = sim_utils.GroundPlaneCfg(
#     size=(30, 30), 
#     physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="multiply",
#             restitution_combine_mode="multiply",
#             static_friction=1.0,
#             dynamic_friction=1.0,
#         )
# )

# GROUND_CFG = AssetBaseCfg(
#     prim_path="/World/ground/GroundPlane",
#     spawn=sim_utils.GroundPlaneCfg(
#         size=(30, 30), 
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#                 friction_combine_mode="multiply",
#                 restitution_combine_mode="multiply",
#                 static_friction=1.0,
#                 dynamic_friction=1.0,
#             )
#     )
# )

# STAIR_CFG = AssetBaseCfg(
#     prim_path="/World/stair",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/stair_2.usd",
#         scale=(0.50, 1.50, 1.00),
#         collision_props=sim_utils.CollisionPropertiesCfg(),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.025),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     )
# )

# DISK_STAIR_CFG = StaticColliderObjectCfg(
#     prim_path="/World/stair",
#     spawn=sim_utils.ManyAssetSpawnerCfg(
#          assets_cfg=[
#              sim_utils.CylinderCfg(
#                 radius=0.18, 
#                 height=0.05,
#                 # height=0.075,
#                 collision_props=sim_utils.CollisionPropertiesCfg(),
#                 mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
#                 rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
#         )], 
#          num_assets=[1]),
#     init_state=StaticColliderObjectCfg.InitialStateCfg(
#         pos=[[(0.0, 0.0, 0.0)]],
#         rot=[[(1.0, 0.0, 0.0, 0.0)]],
#         scale=[[(1.0, 1.0, 1.0)]],
#     )
# )

# # Deformable Object Group
# FEM_GROUND = DeformableObjectCfg(
#         prim_path="/World/Ground",
#         spawn=sim_utils.MeshCuboidCfg(
#             size=(3.0, 1.5, 0.05),
#             deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.0001),
#             mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
#             physics_material=sim_utils.DeformableBodyMaterialCfg(
#                 dynamic_friction=1.0,
#                 poissons_ratio=0.4, 
#                 youngs_modulus=1e2,
#                 elasticity_damping=0.05,
#                 damping_scale=1.0,
#                 ),
#         ),
#         init_state=DeformableObjectCfg.InitialStateCfg(pos=(1.8, 0.0, 0.025)),
#         debug_vis=True,
#     )