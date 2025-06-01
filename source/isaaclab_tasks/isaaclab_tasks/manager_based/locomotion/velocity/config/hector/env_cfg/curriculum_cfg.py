# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp


@configclass
class HECTORCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=hector_mdp.terrain_levels_episode)