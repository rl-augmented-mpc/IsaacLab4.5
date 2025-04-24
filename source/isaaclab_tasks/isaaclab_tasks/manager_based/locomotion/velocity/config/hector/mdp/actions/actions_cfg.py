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
    reference_height: float = 0.55
    """Reference height of the robot."""
    nominal_swing_height : float = 0.12
    """Nominal swing height of the robot."""
    nominal_stepping_frequency: float = 1.0
    """Nominal stepping frequency of the robot."""
    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "LIP"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    friction_cone_coef: float = 0.5
    """Friction cone coefficient of the robot."""