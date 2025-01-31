# from omni.isaac.lab.managers import EventTermCfg
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.utils import configclass
# import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


# @configclass
# class EventCfg:
#     """Configuration for events."""
    
#     # startup
#     robot_foot_friction = EventTermCfg(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg(
#                 "robot",
#                 body_names=[
#                     "L_toe",
#                     "R_toe",
#                 ],
#             ),
#             "static_friction_range": (1.0, 1.0),
#             "dynamic_friction_range": (1.0, 1.0),
#             "restitution_range": (0.0, 0.0),
#             "num_buckets": 64,
#         },
#     )

#     # add_base_mass = EventTermCfg(
#     #     func=mdp.randomize_rigid_body_mass,
#     #     mode="reset",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
#     #         "mass_distribution_params": (-1.0, 1.0),
#     #         "operation": "add",
#     #     },
#     # )
    
#     # add_rock_mass = EventTermCfg(
#     #     func=mdp.randomize_rigid_body_mass,
#     #     mode="reset",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
#     #         "mass_distribution_params": (-1.0, 1.0),
#     #         "operation": "add",
#     #     },
#     # )