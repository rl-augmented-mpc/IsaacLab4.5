# import math

# from omni.isaac.lab_tasks.direct.hector.common.sampler import UniformLineSampler, UniformCubicSampler, GridCubicSampler, QuaternionSampler
# from omni.isaac.lab_tasks.direct.hector.common.curriculum import CurriculumRateSampler, CurriculumUniformLineSampler, CurriculumUniformCubicSampler, CurriculumQuaternionSampler
# from omni.isaac.lab_tasks.direct.hector.common.env_generator import SphereClusterGenerator, GravelClusterGenerator
# # macros 
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
# from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
# ENV_REGEX_NS = "/World/envs/env_.*"

# GRAVEL_GENERATOR = GravelClusterGenerator(
#             prim_path=f"{ENV_REGEX_NS}/StaticColliders",
#             usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Hector/props/rock_1.usd",
#             name="gravel",
#             position_sampler_conf=GridCubicSampler(x_range=[-1.0, 1.0],
#                                                 y_range=[-1.0, 1.0],
#                                                 z_range=[0.02, 0.03],
#                                                 x_resolution=0.05, y_resolution=0.05, z_resolution=0.05),
#             yaw_angle_sampler_conf=QuaternionSampler(x_range=[0.0, 2*math.pi]),
#             scale_sampler_conf=UniformCubicSampler(x_range=[0.2, 0.4], y_range=[0.2, 0.4], z_range=[0.2, 0.3]),
#             mass_sampler_conf=CurriculumUniformLineSampler(x_range_start=(3.0, 5.0), x_range_end=(0.5, 1.0), 
#                                                            rate_sampler=CurriculumRateSampler(function="linear", start=0, end=24*10000))
#             )

# SPHERE_CLUSTER_GENERATOR = SphereClusterGenerator(
#             prim_path=f"{ENV_REGEX_NS}/StaticColliders",
#             name="gravel",
#             friction_coeff=0.5,
#             position_sampler_conf=GridCubicSampler(x_range=[-1.3, 1.3],
#                                                 y_range=[-0.55, 0.55],
#                                                 z_range=[0.01, 0.09],
#                                                 x_resolution=0.02, y_resolution=0.02, z_resolution=0.02),
#             yaw_angle_sampler_conf=QuaternionSampler(x_range=[0.0, 2*math.pi]),
#             radius_sampler_conf=UniformLineSampler(x_range=[0.01, 0.01]),
#             mass_sampler_conf=CurriculumUniformLineSampler(x_range_start=(0.05, 0.1), x_range_end=(0.05, 0.1),
#                                                            rate_sampler=CurriculumRateSampler(function="linear", start=0, end=24*10000))
#             )