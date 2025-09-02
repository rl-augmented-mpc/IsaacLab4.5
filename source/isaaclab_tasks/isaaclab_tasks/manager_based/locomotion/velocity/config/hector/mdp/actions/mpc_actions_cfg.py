# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

# from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from . import mpc_actions
from . import mpc_actions_torch

"""
Pybind controller
"""
@configclass
class BlindLocomotionMPCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCAction

    joint_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    action_range: tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] = (-1.0, 1.0)
    """action range to deal with assymetric action space. """
    negative_action_clip_idx: list[int] = None  # type: ignore
    """List of indices of the action that negative action value should be clipped."""
    command_name: str = "base_velocity"
    """Name of the command to be used for the action term."""
    nominal_height: float = 0.55
    """Reference height of the robot."""
    nominal_swing_height : float = 0.1
    """Nominal swing height of the robot."""
    nominal_stepping_frequency: float = 1.0
    """Nominal stepping frequency of the robot."""
    horizon_length: int = 10
    """Horizon length of the robot."""
    friction_cone_coef: float = 1.0
    """Friction cone coefficient of the robot."""

    control_iteration_between_mpc: int = 10
    """Number of control iterations between MPC updates."""

    # # -- horizon is entire walking step
    # nominal_mpc_dt: float = 0.04
    # """Nominal MPC dt of the robot."""
    # double_support_duration: int = 1 # 0.05s double support
    # """Double support duration of the robot."""
    # single_support_duration: int = 5 # 0.2s single support
    # """Single support duration of the robot."""

    # -- horizon is half of walking step (one foot swing)
    nominal_mpc_dt: float = 0.025
    """Nominal MPC dt of the robot."""
    double_support_duration: int = 2 # 0.05s double support
    """Double support duration of the robot."""
    single_support_duration: int = 8 # 0.2s single support
    """Single support duration of the robot."""

    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "Raibert"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    swing_foot_reference_frame: Literal["world", "base"] = "base"
    """Swing foot reference frame to be used. Can be either "world" or "base"."""

@configclass
class BlindLocomotionMPCActionCfg2(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCAction2
    
@configclass
class BlindLocomotionMPCActionCfg3(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCAction3
    
@configclass
class BlindLocomotionMPCActionCfg4(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCAction4


@configclass
class PerceptiveLocomotionMPCActionCfg(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.PerceptiveLocomotionMPCAction
    swing_foot_reference_frame: Literal["world", "base"] = "world"

@configclass
class PerceptiveLocomotionMPCActionCfg2(PerceptiveLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.PerceptiveLocomotionMPCAction2
    
@configclass
class PerceptiveLocomotionMPCActionCfg3(PerceptiveLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.PerceptiveLocomotionMPCAction3
    
@configclass
class PerceptiveLocomotionMPCActionCfg4(PerceptiveLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.PerceptiveLocomotionMPCAction4
    

"""
PyTorch version of MPC controller
"""

@configclass
class BlindLocomotionGPUMPCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = mpc_actions_torch.BlindLocomotionGPUMPCAction

    # generic controller params
    joint_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    action_range: tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] = (-1.0, 1.0)
    """action range to deal with assymetric action space. """
    negative_action_clip_idx: list[int] = None  # type: ignore
    """List of indices of the action that negative action value should be clipped."""
    command_name: str = "base_velocity"
    """Name of the command to be used for the action term."""
    nominal_height: float = 0.55
    """Reference height of the robot."""
    nominal_swing_height : float = 0.1
    """Nominal swing height of the robot."""
    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "Raibert"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    friction_cone_coef: float = 1.0
    """Friction cone coefficient of the robot."""
    gait_id: int = 2
    """Gait ID of the robot."""


    # MPC specific params
    horizon_length: int = 10
    """Horizon length of the robot."""

    nominal_mpc_dt: float = 0.04
    """Nominal MPC dt of the robot."""
    double_support_duration: int = 0
    """Double support duration of the robot."""
    single_support_duration: int = 5
    """Single support duration of the robot."""

    # nominal_mpc_dt: float = 0.025
    # """Nominal MPC dt of the robot."""
    # double_support_duration: int = 2
    # """Double support duration of the robot."""
    # single_support_duration: int = 8
    # # """Single support duration of the robot."""

    Q: list[float] = [150, 150, 250,   100, 100, 800,   1, 1, 1,   10, 10, 1]
    # Q: list[float] = [150, 150, 250,   100, 100, 50,   1, 1, 10,   10, 10, 1]
    """State cost weights."""
    R: list[float] = [1e-5, 1e-5, 1e-5,   1e-5, 1e-5, 1e-5,   1e-4 , 1e-4, 1e-4,   1e-4, 1e-4, 1e-4]
    # R: list[float] = [1e-5, 1e-5, 1e-4,   1e-5, 1e-5, 1e-4,   1e-4, 1e-4, 1e-4,   1e-4, 1e-4, 1e-4]

    # solver
    solver_name: Literal["osqp", "qpth", "casadi", "cusadi"] = "cusadi"
    print_solve_time: bool = False
    robot: Literal["HECTOR", "T1"] = "HECTOR"
    swing_reference_frame: Literal["world", "base"] = "base"


@configclass
class BlindLocomotionGPUMPCActionCfg2(BlindLocomotionGPUMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions_torch.BLindLocomotionGPUMPCAction2