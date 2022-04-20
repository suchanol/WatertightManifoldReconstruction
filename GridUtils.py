import Grid
import copy
import math
import itertools
import numpy as np
import random as random


def get_lower_bounds_on_components(grid):
    # start flood filling from out of V to extract V_ext
    V_ext = flood_filling(-1, -1, -1, grid)
    V_ext = set(filter(grid.is_valid, V_ext))

    # get the current dilated voxels
    V_crust = set(map(tuple, grid.voxels))

    # get the whole Volume
    V = set(itertools.product(range(grid.resolution), range(grid.resolution), range(grid.resolution)))
    # calculate the set difference to get one voxel that doesn't lie in V_crust nor in V_ext
    voxel = next(iter(set(V).difference(V_ext.union(V_crust))), None)
    V_int1 = set()
    if voxel is not None:
        V_int1 = flood_filling(*voxel, grid)

    lower_bound = 1
    if len(V_ext) + len(V_crust) + len(V_int1) < grid.resolution ** 3:
        lower_bound = 2

    return lower_bound, V_ext, V_int1


def generate_random_points(r, npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= r
    vec_array = np.array(vec)
    return vec_array.transpose()


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


"""def flood_filling(x, y, z, grid, flooded=[]):
    start = [x,y,z]
    print(start)
    if not grid.is_occupied(start) and start not in flooded:
        flooded = flooded + [start]
        neighbors = grid.get_neighbors(start)
        neighbors = list(filter(lambda v: np.logical_and.reduce([-1 <= x <= grid.resolution for x in v]), neighbors))
        for neighbor in neighbors:
            flooded = flooded + [flood_filling(*neighbor, grid, flooded)]
    return flooded"""


def flood_filling(x, y, z, grid):
    start = (x, y, z)
    flooded = []
    stack = []
    stack.append(start)
    while len(stack) > 0:
        cur_voxel = tuple(stack.pop())
        if cur_voxel not in flooded and not grid.is_occupied(cur_voxel):
            flooded.append(cur_voxel)
            neighbors = grid.get_neighbors(cur_voxel)
            neighbors = list(
                filter(lambda v: np.logical_and.reduce([-1 <= x <= grid.resolution for x in v]), neighbors))
            stack += neighbors
    return flooded


# def diffusion(v_crust, cur_phi):
#     new_phi = cur_phi.copy()
#     for voxel, value in cur_phi.items():
#         if not (v_crust.is_point(voxel) & value == 0):
#             neighbors = v_crust.get_valid_neighbors(voxel)
#             neighbors = list(filter(neighbors, v_crust.is_occupied))
#             new_phi[voxel] = (cur_phi[voxel] + sum([cur_phi[neighbor] for neighbor in neighbors])) / (
#                         float(len(neighbors)) + 1.0)
#     return new_phi
def diffusion(v_crust):
    new_phi = v_crust.phi.copy()
    for voxel, value in v_crust.phi.items():
        if value == 1:
            neighbors = v_crust.get_valid_neighbors(voxel)
            neighbors = list(filter(v_crust.is_occupied, neighbors))
            new_phi[tuple(voxel)] = (v_crust.phi[tuple(voxel)] + sum(
                [v_crust.phi[tuple(neighbor)] for neighbor in neighbors])) / (float(len(neighbors)) + 1.0)
    v_crust.phi = new_phi


def dilation(v_crust):
    dilated_voxels = v_crust.voxels
    # new_v_crust = copy.deepcopy(v_crust)
    for voxel in v_crust.voxels:
        dilated_voxels = dilated_voxels + dilate_voxel(voxel, v_crust)
    v_crust.voxels = dilated_voxels
    return


def dilate_voxel(voxel, grid):
    dilated_voxels = []
    for neighbor in grid.get_valid_neighbors(voxel):
        dilated_voxels.append(neighbor)
        grid.phi[tuple(neighbor)] = 1
        # grid.set_occupied(neighbor)
    return dilated_voxels


# can be moved to grid initialization and dilation steps
def initial_phi(grid):
    phi = {}
    for voxel in grid.get_voxels():
        if grid.is_point(voxel):
            phi[tuple(voxel)] = 0
        else:
            phi[tuple(voxel)] = 1
    return phi


# TODO
# remove the voxels that do not contain target surface (not in S_opt) for the next iteration (refinement)
def remove_voxels():
    return
