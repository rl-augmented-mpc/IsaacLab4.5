from .actions.mpc_actions_cfg import (
    BlindLocomotionMPCActionCfg, 
    BlindLocomotionMPCActionCfg2, 
    BlindLocomotionMPCActionCfg3, 
    BlindLocomotionMPCActionCfg4, 
    PerceptiveLocomotionMPCActionCfg,
    PerceptiveLocomotionMPCActionCfg2, 
    PerceptiveLocomotionMPCActionCfg3, 
    PerceptiveLocomotionMPCActionCfg4, 
    TorchMPCActionCfg, 
)
from .terrain.terrain_cfg import (
    SteppingStoneTerrain, 
    InferenceSteppingStoneTerrain, CurriculumSteppingStoneTerrain, 
    RandomOrientationCubeTerrain, PyramidHfTerrain, FractalTerrain, BaseTerrain, 
    CurriculumFrictionPatchTerrain, FrictionPatchTerrain, 
    InferenceRandomBlockTerrain, CurriculumRandomBlockTerrain, 
    TripOverChallengeTerrain, BoxRoughTerrain
)

# # only enable when using dynamic terrain 
# # otherwise, importing this takes long time
# from .terrain.dynamic_terrain_cfg import (
#     GravelTerrain, 
#     # ParticleTerrain
# )

from .observation.observations import (
    base_pos_z,
    joint_pos, joint_vel, joint_torque, 
    height_scan, depth_image,
    swing_phase, foot_placement_b, foot_position_b, reference_foot_position_b, 
    reference_command
)
from .reward.rewards import (
    track_command_lin_vel_xy_exp, track_command_ang_vel_z_exp, 
    leg_body_angle_l2, leg_distance_l2, lin_vel_y_l2, 
    negative_lin_vel_l2, track_torso_height_exp,
    individual_action_l2, mpc_cost_l1, processed_action_l2, 
    stance_foot_position_reward, foot_placement_reward, 
    log_barrier_swing_foot_landing_penalty, 
    feet_accel_l2, active_action_reward,
    stance_foot_position_penalty, foot_placement_penalty, 
    swing_foot_landing_penalty, rough_terrain_processed_action_l2, 
    depth_image_r
)
from .events.events import (
    reset_camera, reset_root_state_uniform, reset_particle_mass, reset_terrain_type, 
    reset_root_state_orthogonal, apply_tangential_external_force_torque
)
from .curriculums.curriculums import terrain_levels_episode, custom_terrain_levels_episode
from .termination.termination import (
    root_height_below_minimum_adaptive, 
    root_height_above_maximum_adaptive, 
    bad_foot_contact
)
from .commands.commands import TerrainAwareUniformVelocityCommand