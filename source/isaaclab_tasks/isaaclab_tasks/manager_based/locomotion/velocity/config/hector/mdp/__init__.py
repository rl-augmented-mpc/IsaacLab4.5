from .actions.actions_cfg import MPCActionCfg
from .terrain.terrain_cfg import SteppingStoneTerrain, CurriculumSteppingStoneTerrain
from .observation.observations import joint_pos, joint_vel, joint_torque, height_scan, \
    swing_phase, foot_placement_b, foot_position_b, reference_foot_position_b
from .reward.rewards import foot_placement_reward, leg_body_angle_l2, individual_action_l2