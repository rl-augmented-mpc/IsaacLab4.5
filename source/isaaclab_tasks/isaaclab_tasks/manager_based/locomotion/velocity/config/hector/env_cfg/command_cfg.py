# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class HECTORCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg( # type: ignore
        # class_type=hector_mdp.TerrainAwareUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( # type: ignore
            lin_vel_x=(0.4, 0.7), 
            lin_vel_y=(0.0, 0.0), 
            # ang_vel_z=(-0.0, 0.0), 
            ang_vel_z=(-0.5, 0.5), 
            heading=(-math.pi, math.pi)
        ),
    )