from .actions.mpc_actions_cfg import MPCActionCfg, MPCActionCfgV2, MPCActionCfgV3, TorchMPCActionCfg
from .terrain.terrain_cfg import (
    SteppingStoneTerrain, CurriculumSteppingStoneTerrain, RandomOrientationCubeTerrain,
    PyramidHfTerrain, FractalTerrain, BaseTerrain
)
# from .terrain.dynamic_terrain_cfg import (
#     GravelTerrain, 
#     ParticleTerrain
# )
from .observation.observations import (
    base_pos_z,
    joint_pos, joint_vel, joint_torque, height_scan,
    swing_phase, foot_placement_b, foot_position_b, reference_foot_position_b
)
from .reward.rewards import (
    leg_body_angle_l2, leg_distance_l2, 
    negative_lin_vel_l2, track_torso_height_exp,
    individual_action_l2, mpc_cost_l1, processed_action_l2, 
    stance_foot_position_reward, foot_placement_reward, feet_accel_l2, active_action_reward,
    stance_foot_position_penalty, foot_placement_penalty
)
from .events.events import reset_camera, reset_root_state_uniform, reset_particle_mass, reset_terrain_type
from .curriculums.curriculums import terrain_levels_episode
from .termination.termination import root_height_below_minimum_adaptive