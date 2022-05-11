import Grid
import numpy as np
from skimage.morphology import flood_fill

def refine_S_opt_grid(old_grid, S_opt, new_resolution):
    new_grid = Grid.Grid()
    new_grid.change_grid(new_resolution, old_grid.get_bounds() / new_resolution, old_grid.min_coord)
    new_grid.insert_point(old_grid.points)
    for (i,j,k) in S_opt:
        new_grid.set_occupied(np.array((2*i,2*j,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i,2*j+1,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i,2*j,2*k+1), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j+1,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j,2*k+1), dtype=int))
        new_grid.set_occupied(np.array((2*i,2*j+1,2*k+1), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j+1,2*k+1), dtype=int))
    return new_grid
def dilation(v_crust):
    dilated_voxels = np.array(v_crust.voxels.nonzero()).T
    voxels_to_dilate = v_crust.get_valid_neighbors(dilated_voxels)
    unoccupied_voxels = voxels_to_dilate[~v_crust.is_point(voxels_to_dilate)]
    v_crust.set_occupied(voxels_to_dilate)
    v_crust.set_phi(unoccupied_voxels, 1)


def diffusion(v_crust):
    orig_voxels = np.array((v_crust.phi == 1).nonzero()).T
    voxels = orig_voxels.copy()
    neighbors = v_crust.get_neighbors(voxels)

    valid = v_crust.is_valid(neighbors)
    valid_indices = np.where(valid)
    valid[valid_indices] &= v_crust.is_occupied(neighbors[valid_indices])
    neighbors = neighbors[valid]

    voxels = np.repeat(voxels, np.count_nonzero(valid, axis=1), axis=0)

    sizes = np.count_nonzero(valid, axis=1)

    v_crust.phi[tuple(voxels.T)] += v_crust.phi[tuple(neighbors.T)]
    v_crust.phi[tuple(orig_voxels.T)] /= sizes + 1

def first_zero(arr):
    mask = np.where(arr == 0)
    if mask[0].size > 0:
        return (mask[0][0], mask[1][0], mask[2][0])
    return (-1, -1, -1)


def flood_filling(image, seed):
    image = image.astype(int)
    flooded_image = flood_fill(image, seed, 2, connectivity=1)
    return np.where(flooded_image == 2)

def get_lower_bounds_on_components(grid):
    seed = first_zero(grid.voxels)
    V_ext = flood_filling(grid.voxels, seed)

    V = grid.voxels.copy()
    V[V_ext] = True

    seed_v_int = first_zero(V)
    V_int = np.array([])
    if not seed_v_int == (-1, -1, -1):
        V_int = flood_filling(V, seed_v_int)
        V[V_int] = True

    lower_bound = 1
    if V[V == False].size > 0:
        lower_bound = 2
    if len(V_int) <= 0:
        lower_bound = 0

    return lower_bound, np.array(V_ext).T, np.array(V_int).T

# the whole process from dilation, diffusion to flood filling to be used directly
def dilation_process(v_crust):
    while True:
        dilation(v_crust)
        diffusion(v_crust)
        lower_bound, V_ext, V_int = get_lower_bounds_on_components(v_crust)
        if lower_bound > 0:
            break
    return lower_bound, V_ext, V_int