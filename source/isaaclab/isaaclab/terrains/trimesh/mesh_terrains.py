# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
from scipy.stats import qmc
import torch
import trimesh
from typing import TYPE_CHECKING

from .utils import *  # noqa: F401, F403
from .utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def flat_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPlaneTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain as a plane.

    .. image:: ../../_static/terrains/trimesh/flat_terrain.jpg
       :width: 45%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # compute the position of the terrain
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.height)
    # compute the vertices of the terrain
    plane_mesh = make_plane(cfg.size, cfg.height, center_zero=False)
    # return the tri-mesh and the position
    return [plane_mesh], np.array(origin)


def pyramid_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin


def inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps + 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders
    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin


def random_grid_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRandomGridTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with cells of random heights and fixed width.

    The terrain is generated in the x-y plane and has a height of 1.0. It is then divided into a grid of the
    specified size :obj:`cfg.grid_width`. Each grid cell is then randomly shifted in the z-direction by a value uniformly
    sampled between :obj:`cfg.grid_height_range`. At the center of the terrain, a platform of the specified width
    :obj:`cfg.platform_width` is generated.

    If :obj:`cfg.holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.

    .. image:: ../../_static/terrains/trimesh/random_grid_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/random_grid_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the terrain is not square. This method only supports square terrains.
        RuntimeError: If the grid width is large such that the border width is negative.
    """
    # check to ensure square terrain
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
    # resolve the terrain configuration
    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # compute the number of boxes in each direction
    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)
    # constant parameters
    terrain_height = 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # generate the border
    border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
    if border_width > 0:
        # compute parameters for the border
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders
    else:
        raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")

    # create a template grid of terrain height
    grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
    grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces of the box to create a template
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to span the terrain (num_boxes_x * num_boxes_y, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_boxes_x, device=device)
    y = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = cfg.grid_width * xx_yy + border_width / 2
    vertices[:, :, :2] += offsets.unsqueeze(1)
    # mask the vertices to create holes, s.t. only grids along the x and y axis are present
    if cfg.holes:
        # -- x-axis
        mask_x = torch.logical_and(
            (vertices[:, :, 0] > (cfg.size[0] - border_width - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 0] < (cfg.size[0] + border_width + cfg.platform_width) / 2).all(dim=1),
        )
        vertices_x = vertices[mask_x]
        # -- y-axis
        mask_y = torch.logical_and(
            (vertices[:, :, 1] > (cfg.size[1] - border_width - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 1] < (cfg.size[1] + border_width + cfg.platform_width) / 2).all(dim=1),
        )
        vertices_y = vertices[mask_y]
        # -- combine these vertices
        vertices = torch.cat((vertices_x, vertices_y))
    # add noise to the vertices to have a random height over each grid cell
    num_boxes = len(vertices)
    # create noise for the z-axis
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].uniform_(-grid_height, grid_height)
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only the top vertices of the box are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes (num_boxes, 12, 3). Each box has 6 faces, each face has 2 triangles.
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    # move to numpy
    faces = faces.view(-1, 3).cpu().numpy()
    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes_list.append(grid_mesh)

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.platform_width, terrain_height + grid_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2 + grid_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], grid_height])

    return meshes_list, origin


def rails_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRailsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with box rails as extrusions.

    The terrain contains two sets of box rails created as extrusions. The first set  (inner rails) is extruded from
    the platform at the center of the terrain, and the second set is extruded between the first set of rails
    and the terrain border. Each set of rails is extruded to the same height.

    .. image:: ../../_static/terrains/trimesh/rails_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. this is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    rail_height = cfg.rail_height_range[1] - difficulty * (cfg.rail_height_range[1] - cfg.rail_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    rail_1_thickness, rail_2_thickness = cfg.rail_thickness_range
    rail_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], rail_height * 0.5)
    # constants for terrain generation
    terrain_height = 1.0
    rail_2_ratio = 0.6

    # generate first set of rails
    rail_1_inner_size = (cfg.platform_width, cfg.platform_width)
    rail_1_outer_size = (cfg.platform_width + 2.0 * rail_1_thickness, cfg.platform_width + 2.0 * rail_1_thickness)
    meshes_list += make_border(rail_1_outer_size, rail_1_inner_size, rail_height, rail_center)
    # generate second set of rails
    rail_2_inner_x = cfg.platform_width + (cfg.size[0] - cfg.platform_width) * rail_2_ratio
    rail_2_inner_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * rail_2_ratio
    rail_2_inner_size = (rail_2_inner_x, rail_2_inner_y)
    rail_2_outer_size = (rail_2_inner_x + 2.0 * rail_2_thickness, rail_2_inner_y + 2.0 * rail_2_thickness)
    meshes_list += make_border(rail_2_outer_size, rail_2_inner_size, rail_height, rail_center)
    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes_list, origin


def pit_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPitTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pit with levels (stairs) leading out of the pit.

    The terrain contains a platform at the center and a staircase leading out of the pit.
    The staircase is a series of steps that are aligned along the x- and y- axis. The steps are
    created by extruding a ring along the x- and y- axis. If :obj:`is_double_pit` is True, the pit
    contains two levels.

    .. image:: ../../_static/terrains/trimesh/pit_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/trimesh/pit_terrain_with_two_levels.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    pit_depth = cfg.pit_depth_range[0] + difficulty * (cfg.pit_depth_range[1] - cfg.pit_depth_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    inner_pit_size = (cfg.platform_width, cfg.platform_width)
    total_depth = pit_depth
    # constants for terrain generation
    terrain_height = 1.0
    ring_2_ratio = 0.6

    # if the pit is double, the inner ring is smaller to fit the second level
    if cfg.double_pit:
        # increase the total height of the pit
        total_depth *= 2.0
        # reduce the size of the inner ring
        inner_pit_x = cfg.platform_width + (cfg.size[0] - cfg.platform_width) * ring_2_ratio
        inner_pit_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * ring_2_ratio
        inner_pit_size = (inner_pit_x, inner_pit_y)

    # generate the pit (outer ring)
    pit_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -total_depth * 0.5]
    meshes_list += make_border(cfg.size, inner_pit_size, total_depth, pit_center)
    # generate the second level of the pit (inner ring)
    if cfg.double_pit:
        pit_center[2] = -total_depth
        meshes_list += make_border(inner_pit_size, (cfg.platform_width, cfg.platform_width), total_depth, pit_center)
    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -total_depth - terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], -total_depth])

    return meshes_list, origin


def box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with boxes (similar to a pyramid).

    The terrain has a ground with boxes on top of it that are stacked on top of each other.
    The boxes are created by extruding a rectangle along the z-axis. If :obj:`double_box` is True,
    then two boxes of height :obj:`box_height` are stacked on top of each other.

    .. image:: ../../_static/terrains/trimesh/box_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/trimesh/box_terrain_with_two_boxes.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    total_height = box_height
    if cfg.double_box:
        total_height *= 2.0
    # constants for terrain generation
    terrain_height = 1.0
    box_2_ratio = 0.6

    # Generate the top box
    dim = (cfg.platform_length, cfg.platform_width, terrain_height + total_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], (total_height - terrain_height) / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)
    # Generate the lower box
    if cfg.double_box:
        # calculate the size of the lower box
        outer_box_x = cfg.platform_length + (cfg.size[0] - cfg.platform_length) * box_2_ratio
        outer_box_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * box_2_ratio
        # create the lower box
        dim = (outer_box_x, outer_box_y, terrain_height + total_height / 2)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], (total_height - terrain_height) / 2 - total_height / 4)
        box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box_mesh)
    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin

def gap_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by a gap
    of width :obj:`gap_width` on all sides.

    .. image:: ../../_static/terrains/trimesh/gap_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)

    # Generate the outer ring
    inner_size = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width)
    meshes_list += make_border(cfg.size, inner_size, terrain_height, terrain_center)
    # Generate the inner box
    box_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin


def floating_ring_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshFloatingRingTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a floating square ring.

    The terrain has a ground with a floating ring in the middle. The ring extends from the center from
    :obj:`platform_width` to :obj:`platform_width` + :obj:`ring_width` in the x and y directions.
    The thickness of the ring is :obj:`ring_thickness` and the height of the ring from the terrain
    is :obj:`ring_height`.

    .. image:: ../../_static/terrains/trimesh/floating_ring_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    ring_height = cfg.ring_height_range[1] - difficulty * (cfg.ring_height_range[1] - cfg.ring_height_range[0])
    ring_width = cfg.ring_width_range[0] + difficulty * (cfg.ring_width_range[1] - cfg.ring_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0

    # Generate the floating ring
    ring_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], ring_height + 0.5 * cfg.ring_thickness)
    ring_outer_size = (cfg.platform_width + 2 * ring_width, cfg.platform_width + 2 * ring_width)
    ring_inner_size = (cfg.platform_width, cfg.platform_width)
    meshes_list += make_border(ring_outer_size, ring_inner_size, cfg.ring_thickness, ring_center)
    # Generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.asarray([pos[0], pos[1], 0.0])

    return meshes_list, origin


def star_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStarTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a star.

    The terrain has a ground with a cylinder in the middle. The star is made of :obj:`num_bars` bars
    with a width of :obj:`bar_width` and a height of :obj:`bar_height`. The bars are evenly
    spaced around the cylinder and connect to the peripheral of the terrain.

    .. image:: ../../_static/terrains/trimesh/star_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If :obj:`num_bars` is less than 2.
    """
    # check the number of bars
    if cfg.num_bars < 2:
        raise ValueError(f"The number of bars in the star must be greater than 2. Received: {cfg.num_bars}")

    # resolve the terrain configuration
    bar_height = cfg.bar_height_range[0] + difficulty * (cfg.bar_height_range[1] - cfg.bar_height_range[0])
    bar_width = cfg.bar_width_range[1] - difficulty * (cfg.bar_width_range[1] - cfg.bar_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # Generate a platform in the middle
    platform_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -bar_height / 2)
    platform_transform = trimesh.transformations.translation_matrix(platform_center)
    platform = trimesh.creation.cylinder(
        cfg.platform_width * 0.5, bar_height, sections=2 * cfg.num_bars, transform=platform_transform
    )
    meshes_list.append(platform)
    # Generate bars to connect the platform to the terrain
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(platform_center)
    yaw = 0.0
    for _ in range(cfg.num_bars):
        # compute the length of the bar based on the yaw
        # length changes since the bar is connected to a square border
        bar_length = cfg.size[0]
        if yaw < 0.25 * np.pi:
            bar_length /= np.math.cos(yaw)
        elif yaw < 0.75 * np.pi:
            bar_length /= np.math.sin(yaw)
        else:
            bar_length /= np.math.cos(np.pi - yaw)
        # compute the transform of the bar
        transform[0:3, 0:3] = tf.Rotation.from_euler("z", yaw).as_matrix()
        # add the bar to the mesh
        dim = [bar_length - bar_width, bar_width, bar_height]
        bar = trimesh.creation.box(dim, transform)
        meshes_list.append(bar)
        # increment the yaw
        yaw += np.pi / cfg.num_bars
    # Generate the exterior border
    inner_size = (cfg.size[0] - 2 * bar_width, cfg.size[1] - 2 * bar_width)
    meshes_list += make_border(cfg.size, inner_size, bar_height, platform_center)
    # Generate the ground
    ground = make_plane(cfg.size, -bar_height, center_zero=False)
    meshes_list.append(ground)
    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin


def repeated_objects_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects.

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Depending on the object type, the objects are generated with different parameters. The objects
    The types of objects that can be generated are: ``"cylinder"``, ``"box"``, ``"cone"``.

    The object parameters are specified in the configuration as curriculum parameters. The difficulty
    is used to linearly interpolate between the minimum and maximum values of the parameters.

    .. image:: ../../_static/terrains/trimesh/repeated_objects_cylinder_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_box_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_pyramid_terrain.jpg
       :width: 30%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    # import the object functions -- this is done here to avoid circular imports
    from .mesh_terrains_cfg import (
        MeshRepeatedBoxesTerrainCfg,
        MeshRepeatedCylindersTerrainCfg,
        MeshRepeatedPyramidsTerrainCfg,
    )

    # if object type is a string, get the function: make_{object_type}
    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    # -- common parameters
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    if isinstance(cfg, MeshRepeatedBoxesTerrainCfg):
        cp_0: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, MeshRepeatedPyramidsTerrainCfg):  # noqa: SIM114
        cp_0: MeshRepeatedPyramidsTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedPyramidsTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, MeshRepeatedCylindersTerrainCfg):  # noqa: SIM114
        cp_0: MeshRepeatedCylindersTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedCylindersTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")
    # constants for the terrain
    platform_clearance = 0.0

    # initialize list of meshes
    meshes_list = list()
    # compute quantities
    # origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * height))
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # sample valid center for objects
    object_centers = np.zeros((num_objects, 3))
    # use a mask to track invalid objects that still require sampling
    mask_objects_left = np.ones((num_objects,), dtype=bool)
    # loop until no objects are left to sample
    while np.any(mask_objects_left):
        # only sample the centers of the remaining invalid objects
        num_objects_left = mask_objects_left.sum()
        object_centers[mask_objects_left, 0] = np.random.uniform(0, cfg.size[0], num_objects_left)
        object_centers[mask_objects_left, 1] = np.random.uniform(0, cfg.size[1], num_objects_left)
        # filter out the centers that are on the platform
        is_within_platform_x = np.logical_and(
            object_centers[mask_objects_left, 0] >= platform_corners[0, 0],
            object_centers[mask_objects_left, 0] <= platform_corners[1, 0],
        )
        is_within_platform_y = np.logical_and(
            object_centers[mask_objects_left, 1] >= platform_corners[0, 1],
            object_centers[mask_objects_left, 1] <= platform_corners[1, 1],
        )
        # update the mask to track the validity of the objects sampled in this iteration
        mask_objects_left[mask_objects_left] = np.logical_and(is_within_platform_x, is_within_platform_y)

    # generate obstacles (but keep platform clean)
    for index in range(len(object_centers)):
        # randomize the height of the object
        ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
        if ob_height > 0.0:
            object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
            meshes_list.append(object_mesh)

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate a platform in the middle
    # dim = (cfg.platform_width, cfg.platform_width, 0.5 * height)
    # pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * height)
    # platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    # meshes_list.append(platform)

    return meshes_list, origin



"""
Custom made terrain not in the original codebase.
"""

def thick_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshThickTerrainCfg
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a planar terrain with some depth

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    meshes_list = list()
    terrain_height = cfg.thickness
    
    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], pos[2]])

    return meshes_list, origin

def tiled_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.TiledMeshBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with boxes (similar to a pyramid).

    The terrain has a ground with boxes on top of it that are stacked on top of each other.
    The boxes are created by extruding a rectangle along the z-axis. If :obj:`double_box` is True,
    then two boxes of height :obj:`box_height` are stacked on top of each other.

    .. image:: ../../_static/terrains/trimesh/box_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/trimesh/box_terrain_with_two_boxes.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])
    platform_gap_0 = cfg.platform_gap_range_start[0] + difficulty * (cfg.platform_gap_range_end[0] - cfg.platform_gap_range_start[0])
    platform_gap_1 = cfg.platform_gap_range_start[1] + difficulty * (cfg.platform_gap_range_end[1] - cfg.platform_gap_range_start[1])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    total_height = box_height
    # constants for terrain generation
    terrain_height = 0.2
    
    step_sizes = np.random.uniform(platform_gap_0, platform_gap_1, size=cfg.num_box).cumsum()
    box_origins_x = cfg.border_size + step_sizes
    
    # filter box origins
    is_within_platform_1 = (cfg.border_size < box_origins_x)
    is_within_platform_2 = (box_origins_x < cfg.size[0] / 2 - cfg.center_area_size / 2)
    is_within_platform_3 = (box_origins_x > cfg.size[0] / 2 + cfg.center_area_size / 2)
    is_within_platform_4 = (box_origins_x < cfg.size[0] - cfg.border_size)
    valid_indices = np.logical_or(np.logical_and(is_within_platform_1, is_within_platform_2),
                                  np.logical_and(is_within_platform_3, is_within_platform_4))
    box_origins_x = box_origins_x[valid_indices]
    valid_num_boxes = len(box_origins_x)
    
    height_noise = np.random.uniform(cfg.height_noise_range[0], cfg.height_noise_range[1], size=valid_num_boxes)
    
    for i in range(valid_num_boxes):
        # Generate the top box
        dim = (cfg.platform_length, cfg.platform_width, total_height+height_noise[i])
        pos = (box_origins_x[i], 0.5 * cfg.size[1], (total_height+height_noise[i]) / 2)
        box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box_mesh)
    
    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain (z to be 0)
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes_list, origin

def stair_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.StairTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with multiple stairs with random up and downs.

    .. image:: ../../_static/terrains/trimesh/box_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/trimesh/box_terrain_with_two_boxes.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])
    platform_gap_0 = cfg.platform_gap_range_start[0] + difficulty * (cfg.platform_gap_range_end[0] - cfg.platform_gap_range_start[0]) # lower bound
    platform_gap_1 = cfg.platform_gap_range_start[1] + difficulty * (cfg.platform_gap_range_end[1] - cfg.platform_gap_range_start[1]) # upper bound
    platform_length_0 = cfg.platform_length_range_start[0] + difficulty * (cfg.platform_length_range_end[0] - cfg.platform_length_range_start[0]) # lower bound
    platform_length_1 = cfg.platform_length_range_start[1] + difficulty * (cfg.platform_length_range_end[1] - cfg.platform_length_range_start[1]) # upper bound

    # initialize list of meshes
    meshes_list = list()
    
    # construct center block
    dim = (cfg.center_area_size, cfg.platform_width, box_height)
    pos = (0.5*cfg.size[0], 0.5 * cfg.size[1], -box_height / 2)
    meshes_list.append(
        trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    )
    
    # construct the left and right blocks
    platform_gap_left_half = np.random.uniform(platform_gap_0, platform_gap_1, size=cfg.num_box//2)
    platform_gap_right_half = np.random.uniform(platform_gap_0, platform_gap_1, size=cfg.num_box//2)
    platform_length_left_half = np.random.uniform(platform_length_0, platform_length_1, size=cfg.num_box//2)
    platform_length_right_half = np.random.uniform(platform_length_0, platform_length_1, size=cfg.num_box//2)
    
    # up
    assert cfg.profile_mode in ["up", "down", "up_down", "random", "pyramid", "inv_pyramid"], "Invalid profile mode. Choose from 'up', 'down', 'up_down', 'random', 'pyramid', or 'inv_pyramid'."
    if cfg.profile_mode == "up":
        block_height_profile_left = np.ones(len(platform_gap_left_half))
        block_height_profile_right = np.ones(len(platform_gap_right_half))
    elif cfg.profile_mode == "down":
        block_height_profile_left = -np.ones(len(platform_gap_left_half))
        block_height_profile_right = -np.ones(len(platform_gap_right_half))
    elif cfg.profile_mode == "up_down":
        block_height_profile_left = np.ones(len(platform_gap_left_half))
        block_height_profile_left[np.arange(1, len(platform_gap_left_half), 2)] = -1
        block_height_profile_right = np.ones(len(platform_gap_right_half))
        block_height_profile_right[np.arange(1, len(platform_gap_right_half), 2)] = -1
    elif cfg.profile_mode == "random":
        block_height_profile_left = np.random.choice([-1, 1], size=len(platform_gap_left_half))
        block_height_profile_right = np.random.choice([-1, 1], size=len(platform_gap_right_half))
    elif cfg.profile_mode == "pyramid":
        left_up_number = len(platform_gap_left_half) // 2
        left_down_number = len(platform_gap_left_half) - left_up_number
        right_up_number = len(platform_gap_right_half) // 2
        right_down_number = len(platform_gap_right_half) - right_up_number
        block_height_profile_left = np.concatenate([np.ones(left_up_number), -np.ones(left_down_number)])
        block_height_profile_right = np.concatenate([np.ones(right_up_number), -np.ones(right_down_number)])
    elif cfg.profile_mode == "inv_pyramid":
        left_up_number = len(platform_gap_left_half) // 2
        left_down_number = len(platform_gap_left_half) - left_up_number
        right_up_number = len(platform_gap_right_half) // 2
        right_down_number = len(platform_gap_right_half) - right_up_number
        block_height_profile_left = np.concatenate([-np.ones(left_down_number), np.ones(left_up_number)])
        block_height_profile_right = np.concatenate([-np.ones(right_down_number), np.ones(right_up_number)])
    
    # left blocks
    block_length_prev = cfg.center_area_size
    h_prev = box_height
    center_z = -box_height/2
    height_noise = np.random.uniform(cfg.height_noise_range[0], cfg.height_noise_range[1], size=len(platform_gap_left_half))
    block_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
    for i in range(len(platform_gap_left_half)):
        block_length = platform_length_left_half[i]
        h = box_height + height_noise[i]
        center_z = center_z + (h_prev/2 + h/2)*block_height_profile_left[i]
        dim = (block_length, cfg.platform_width, h)
        block_center = (block_center[0]-block_length/2-block_length_prev/2 - platform_gap_left_half[i], block_center[1], center_z)
        if block_center[0] < cfg.border_size:
            break
        
        meshes_list.append(
            trimesh.creation.box(dim, trimesh.transformations.translation_matrix(block_center))
        )
        
        # update memory
        h_prev = h
        block_length_prev = block_length
    
    # right blocks
    block_length_prev = cfg.center_area_size
    h_prev = box_height
    center_z = -box_height/2
    height_noise = np.random.uniform(cfg.height_noise_range[0], cfg.height_noise_range[1], size=len(platform_gap_right_half))
    block_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
    for i in range(len(platform_gap_right_half)):
        block_length = platform_length_right_half[i]
        h = box_height + height_noise[i]
        center_z = center_z + (h_prev/2 + h/2)*block_height_profile_right[i]
        dim = (block_length, cfg.platform_width, h)
        block_center = (block_center[0]+block_length/2+block_length_prev/2 + platform_gap_right_half[i], block_center[1], center_z)
        if block_center[0] > cfg.size[0] - cfg.border_size:
            break
        
        meshes_list.append(
            trimesh.creation.box(dim, trimesh.transformations.translation_matrix(block_center))
        )
        
        # update memory
        h_prev = h
        block_length_prev = block_length
    
    pos = (0.5*cfg.size[0], 0.5 * cfg.size[1], 0.0)
    origin = np.array([pos[0], pos[1], pos[2]])

    return meshes_list, origin

def random_block_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRandomBlockTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with cells of random heights and fixed width.

    The terrain is generated in the x-y plane and has a height of 1.0. It is then divided into a grid of the
    specified size :obj:`cfg.grid_width`. Each grid cell is then randomly shifted in the z-direction by a value uniformly
    sampled between :obj:`cfg.grid_height_range`. At the center of the terrain, a platform of the specified width
    :obj:`cfg.platform_width` is generated.

    If :obj:`cfg.holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.

    .. image:: ../../_static/terrains/trimesh/random_grid_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/random_grid_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the terrain is not square. This method only supports square terrains.
        RuntimeError: If the grid width is large such that the border width is negative.
    """
    # check to ensure square terrain
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
    # resolve the terrain configuration
    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # compute the number of boxes in each direction
    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)
    # constant parameters
    terrain_height = 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # generate the border
    border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
    if border_width > 0:
        # compute parameters for the border
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders
    # else:
    #     raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")

    # create a template grid of terrain height
    grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
    grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces of the box to create a template
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to span the terrain (num_boxes_x * num_boxes_y, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_boxes_x, device=device)
    y = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = cfg.grid_width * xx_yy + border_width / 2
    vertices[:, :, :2] += offsets.unsqueeze(1)
    # mask the vertices to create holes, s.t. only grids along the x and y axis are present
    if cfg.holes:
        # -- x-axis
        mask_x = torch.logical_and(
            (vertices[:, :, 0] > (cfg.size[0] - border_width - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 0] < (cfg.size[0] + border_width + cfg.platform_width) / 2).all(dim=1),
        )
        vertices_x = vertices[mask_x]
        # -- y-axis
        mask_y = torch.logical_and(
            (vertices[:, :, 1] > (cfg.size[1] - border_width - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 1] < (cfg.size[1] + border_width + cfg.platform_width) / 2).all(dim=1),
        )
        vertices_y = vertices[mask_y]
        # -- combine these vertices
        vertices = torch.cat((vertices_x, vertices_y))
    # add noise to the vertices to have a random height over each grid cell
    num_boxes = len(vertices)
    # create noise for the z-axis
    h_noise = torch.zeros((num_boxes, 3), device=device)
    
    # h_noise[:, 2].uniform_(-grid_height, grid_height) # uniform noise
    h_noise[:, 2].uniform_(cfg.uniform_noise_range[0], cfg.uniform_noise_range[1]) # uniform noise
    h_noise[:, 2] *= torch.from_numpy(np.random.choice([-1, 1], size=num_boxes)).to(device) * grid_height # random sign
    
    # h_noise[:, 2] = (2* torch.randint(0, 2, (num_boxes,), device=device) - 1) * grid_height # deterministic noise (-1 or 1)
    
    # # checkerboard like noise
    # x_idx = torch.arange(num_boxes_x, device=device)
    # y_idx = torch.arange(num_boxes_y, device=device)
    # XX, YY = torch.meshgrid(x_idx, y_idx, indexing="ij")
    # X = XX.flatten().view(-1, 1)
    # Y = YY.flatten().view(-1, 1)
    # box_xy = torch.cat((X, Y), dim=1)
    # parity = (box_xy[:, 0] + box_xy[:, 1]) % 2  # 0 or 1
    # height_sign = 2 * parity - 1  # 0 -> -1, 1 -> +1
    # h_noise[:, 2] = height_sign * grid_height * h_noise[:, 2].uniform_(0.7, 1.0)  # random noise in [-1, 1] scaled by grid_height
    
    # zero at the center
    num_platform = int(cfg.platform_width / cfg.grid_width)//2
    h_noise = h_noise.reshape(num_boxes_y, num_boxes_x, 3)
    h_noise[num_boxes_y//2 - num_platform : num_boxes_y//2 + num_platform, num_boxes_x//2 - num_platform : num_boxes_x//2 + num_platform, 2] = 0.0
    h_noise = h_noise.view(-1, 3)
    
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only the top vertices of the box are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes (num_boxes, 12, 3). Each box has 6 faces, each face has 2 triangles.
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    # move to numpy
    faces = faces.view(-1, 3).cpu().numpy()
    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes_list.append(grid_mesh)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin



def poisson_disk_sampling_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects with Poisson disk sampling.

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Depending on the object type, the objects are generated with different parameters. The objects
    The types of objects that can be generated are: ``"cylinder"``, ``"box"``, ``"cone"``.

    The object parameters are specified in the configuration as curriculum parameters. The difficulty
    is used to linearly interpolate between the minimum and maximum values of the parameters.

    .. image:: ../../_static/terrains/trimesh/repeated_objects_cylinder_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_box_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_pyramid_terrain.jpg
       :width: 30%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    # import the object functions -- this is done here to avoid circular imports
    from .mesh_terrains_cfg import (
        MeshRepeatedBoxesTerrainCfg,
        MeshRepeatedCylindersTerrainCfg,
        MeshRepeatedPyramidsTerrainCfg,
    )

    # if object type is a string, get the function: make_{object_type}
    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    # -- common parameters
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    if isinstance(cfg, MeshRepeatedBoxesTerrainCfg):
        cp_0: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
        
        poisson_radius = np.sqrt(cp_1.size[0]**2 + cp_1.size[1]**2) / np.sqrt(cfg.size[0]**2 + cfg.size[1]**2) # type: ignore
    elif isinstance(cfg, MeshRepeatedPyramidsTerrainCfg):  # noqa: SIM114
        cp_0: MeshRepeatedPyramidsTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedPyramidsTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
        poisson_radius = cp_1.radius / np.sqrt(cfg.size[0]**2 + cfg.size[1]**2)
    elif isinstance(cfg, MeshRepeatedCylindersTerrainCfg):  # noqa: SIM114
        cp_0: MeshRepeatedCylindersTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedCylindersTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
        poisson_radius = cp_1.radius / np.sqrt(cfg.size[0]**2 + cfg.size[1]**2)
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")
    # constants for the terrain
    platform_clearance = 0.0

    # initialize list of meshes
    meshes_list = list()
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    
    # create Poisson disk sampler
    engine = qmc.PoissonDisk(d=2, radius=2.0*poisson_radius, seed=42)
    
    # sample valid center for objects
    object_centers = np.zeros((num_objects, 3))
    # use a mask to track invalid objects that still require sampling
    mask_objects_left = np.ones((num_objects,), dtype=bool)
    # loop until no objects are left to sample
    while np.any(mask_objects_left):
        # only sample the centers of the remaining invalid objects
        num_objects_left = mask_objects_left.sum()
        poisson_samples = engine.random(num_objects_left) * cfg.size[0]  # engine.random gives (0, 1) range
        
        # sample from poisson disk until we have enough samples
        while poisson_samples.shape[0] < num_objects_left:
            extra_samples = engine.random(num_objects_left - poisson_samples.shape[0]) * cfg.size[0]
            poisson_samples = np.concatenate((poisson_samples, extra_samples), axis=0)
        object_centers[mask_objects_left, :2] = poisson_samples[:num_objects_left, :2]
        
        # filter out the centers that are on the platform
        is_within_platform_x = np.logical_and(
            object_centers[mask_objects_left, 0] >= platform_corners[0, 0],
            object_centers[mask_objects_left, 0] <= platform_corners[1, 0],
        )
        is_within_platform_y = np.logical_and(
            object_centers[mask_objects_left, 1] >= platform_corners[0, 1],
            object_centers[mask_objects_left, 1] <= platform_corners[1, 1],
        )
        # update the mask to track the validity of the objects sampled in this iteration
        mask_objects_left[mask_objects_left] = np.logical_and(is_within_platform_x, is_within_platform_y)
        
        engine.reset()

    # generate obstacles
    for index in range(len(object_centers)):
        # randomize the height of the object
        ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
        # randomize shape parameters (length, width) or radius depending on the object type by changing ratio from 0.5 to 1.5
        if ob_height > 0.0:
            if isinstance(cfg, MeshRepeatedBoxesTerrainCfg):
                object_width = (1 + np.random.uniform(-0.3, 0.3)) * object_kwargs["width"]
                object_length = (1 + np.random.uniform(-0.3, 0.3)) * object_kwargs["length"]
                object_mesh = object_func(
                    center=object_centers[index], 
                    height=ob_height, 
                    width=object_width, 
                    length=object_length, 
                    max_yx_angle=object_kwargs["max_yx_angle"], 
                    degrees=object_kwargs["degrees"])
            elif isinstance(cfg, MeshRepeatedPyramidsTerrainCfg):
                object_raidus = (1 + np.random.uniform(-0.5, 0.5)) * object_kwargs["radius"]
                object_mesh = object_func(
                    center=object_centers[index], 
                    height=ob_height,
                    radius=object_raidus,
                    max_yx_angle=object_kwargs["max_yx_angle"],
                    degrees=object_kwargs["degrees"])
            elif isinstance(cfg, MeshRepeatedCylindersTerrainCfg):
                object_raidus = (1 + np.random.uniform(-0.5, 0.5)) * object_kwargs["radius"]
                object_mesh = object_func(
                    center=object_centers[index], 
                    height=ob_height,
                    radius=object_raidus,
                    max_yx_angle=object_kwargs["max_yx_angle"],
                    degrees=object_kwargs["degrees"])
            meshes_list.append(object_mesh)

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    return meshes_list, origin