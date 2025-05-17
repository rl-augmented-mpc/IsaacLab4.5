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


@configclass
class MPCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = mpc_actions.MPCAction

    joint_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    action_range: tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] = (-1.0, 1.0)
    """action range to deal with assymetric action space. """
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
    control_iteration_between_mpc: int = 8
    """Control iteration between MPC iterations."""
    double_support_duration: int = 0
    """Double support duration of the robot."""
    single_support_duration: int = 5
    """Single support duration of the robot."""
    nominal_swing_duration: float = 0.2
    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "Raibert"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    friction_cone_coef: float = 1.0
    """Friction cone coefficient of the robot."""

@configclass
class MPCActionCfgV2(ActionTermCfg):
    class_type: type[ActionTerm] = mpc_actions.MPCAction2

    joint_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    action_range: tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] = (-1.0, 1.0)
    """action range to deal with assymetric action space. """
    command_name: str = "base_velocity"
    """Name of the command to be used for the action term."""
    nominal_height: float = 0.55
    """Reference height of the robot."""
    nominal_swing_height : float = 0.12
    """Nominal swing height of the robot."""
    nominal_stepping_frequency: float = 1.0
    """Nominal stepping frequency of the robot."""
    horizon_length: int = 10
    """Horizon length of the robot."""
    
    ### == construct gait in mpc dt == ###
    control_iteration_between_mpc: int = 20
    """Control iteration between MPC iterations."""
    nominal_mpc_dt: float = 0.04
    """Nominal MPC dt of the robot."""
    double_support_duration: int = 0
    """Double support duration of the robot."""
    single_support_duration: int = 5
    """Single support duration of the robot."""
    
    # ### == construct gait in control dt == ###
    # control_iteration_between_mpc: int = 20
    # """Control iteration between MPC iterations."""
    # nominal_mpc_dt: float = 0.002
    # """Nominal MPC dt of the robot."""
    # double_support_duration: int = 0
    # """Double support duration of the robot."""
    # single_support_duration: int = 100
    # """Single support duration of the robot."""
    
    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "Raibert"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    friction_cone_coef: float = 1.0
    """Friction cone coefficient of the robot."""
    

@configclass
class MPCActionCfgV3(ActionTermCfg):
    class_type: type[ActionTerm] = mpc_actions.MPCAction3

    joint_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    action_range: tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] = (-1.0, 1.0)
    """action range to deal with assymetric action space. """
    command_name: str = "base_velocity"
    """Name of the command to be used for the action term."""
    nominal_height: float = 0.55
    """Reference height of the robot."""
    nominal_swing_height : float = 0.12
    """Nominal swing height of the robot."""
    nominal_stepping_frequency: float = 1.0
    """Nominal stepping frequency of the robot."""
    horizon_length: int = 10
    """Horizon length of the robot."""
    
    ### == construct gait in control dt == ###
    control_iteration_between_mpc: int = 20
    """Control iteration between MPC iterations."""
    nominal_mpc_dt: float = 0.002
    """Nominal MPC dt of the robot."""
    double_support_duration: int = 0
    """Double support duration of the robot."""
    single_support_duration: int = 100
    """Single support duration of the robot."""
    
    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "Raibert"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    friction_cone_coef: float = 1.0
    """Friction cone coefficient of the robot."""