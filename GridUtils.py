import Grid
import copy


def flood_filling(x, y, z, grid):
    if not grid.is_occupied(x, y, z):
        grid.set_occupied(x, y, z)
        neighbors = grid.get_neighbors((x,y,z))
        for neighbor in neighbors:
            flood_filling(neighbor, grid)


def diffusion(v_crust, cur_phi):
    new_phi = cur_phi.copy()
    for voxel, value in cur_phi:
        if not (v_crust.is_point(voxel) & value == 0):
            neighbors = v_crust.get_neighbors(voxel)
            neighbors = list(filter(neighbors, v_crust.is_occupied))
            new_phi[voxel] = (cur_phi[voxel] + sum([cur_phi[neighbor] for neighbor in neighbors])) / (float(len(neighbors)) + 1.0)
    return new_phi


def dilation(v_crust):
    new_v_crust = copy.deepcopy(v_crust)
    for voxel in v_crust.get_voxels():
        for neighbor in v_crust.get_neighbors(voxel):
            new_v_crust.set_occupied(neighbor)
    return new_v_crust


def initial_phi(grid):
    phi = {}
    for voxel in grid.get_voxels():
        if grid.is_point(voxel):
            phi[voxel] = 0
        else:
            phi[voxel] = 1
    return phi
