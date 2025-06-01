# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp


@configclass
class HECTORTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # type: ignore
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,  # type: ignore
    #     params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi/10},
    #     time_out=True,
    # )
    base_too_low = DoneTerm(
        func=hector_mdp.root_height_below_minimum_adaptive,  # type: ignore
        params={
            "minimum_height": 0.35,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_sole"),
        },
    )
    
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.0},
        time_out=True,
    )