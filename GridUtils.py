import Grid
import copy
import math
import itertools
import numpy as np


def refine_grid(grid, new_resolution):
    new_grid = Grid.Grid()
    new_grid.resolution = new_resolution
    new_grid.voxel_size = grid.get_bounds() / new_resolution
    ratio = math.ceil(new_resolution / grid.resolution)
    for voxel in grid.get_voxels():
        bounds = [list(range(ratio * int(x), ratio)) for x in voxel]
        for new_voxel in itertools.product(*bounds):
            if new_grid.is_valid(new_voxel):
                new_grid.set_occupied(new_voxel)

    for point in grid.get_points():
        new_grid.insert_point(point)
    return new_grid


def flood_filling(x, y, z, grid, flooded=[]):
    if not grid.is_occupied(x, y, z) and (x, y, z) not in flooded:
        flooded = flooded + [(x, y, z)]
        neighbors = grid.get_neighbors((x, y, z))
        neighbors = list(filter(lambda v: np.logical_and.reduce([-1 <= x <= grid.resolution for x in v]), neighbors))
        for neighbor in neighbors:
            flooded = flooded + flood_filling(neighbor, grid, flooded)
    return flooded


def diffusion(v_crust, cur_phi):
    new_phi = cur_phi.copy()
    for voxel, value in cur_phi.items():
        if not (v_crust.is_point(voxel) & value == 0):
            neighbors = v_crust.get_valid_neighbors(voxel)
            neighbors = list(filter(neighbors, v_crust.is_occupied))
            new_phi[voxel] = (cur_phi[voxel] + sum([cur_phi[neighbor] for neighbor in neighbors])) / (float(len(neighbors)) + 1.0)
    return new_phi


def dilation(v_crust):
    dilated_voxels = []
    new_v_crust = copy.deepcopy(v_crust)
    for voxel in v_crust.get_voxels():
        dilated_voxels = dilated_voxels + dilate_voxel(voxel, new_v_crust)
    return new_v_crust, dilated_voxels


def dilate_voxel(voxel, grid):
    dilated_voxels = []
    for neighbor in grid.get_valid_neighbors(voxel):
        dilated_voxels.append(tuple(neighbor))
        grid.set_occupied(neighbor)
    return dilated_voxels


def initial_phi(grid):
    phi = {}
    for voxel in grid.get_voxels():
        if grid.is_point(voxel):
            phi[tuple(voxel)] = 0
        else:
            phi[tuple(voxel)] = 1
    return phi
