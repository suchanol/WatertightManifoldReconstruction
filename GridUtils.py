import itertools

from matplotlib import pyplot as plt

import Grid
import numpy as np
from skimage.morphology import flood
from mpl_toolkits.mplot3d import Axes3D

def refine_S_opt_grid(old_grid, S_opt, new_resolution):
    new_grid = Grid.Grid()
    new_grid.change_grid(new_resolution, old_grid.get_bounds() / new_resolution, old_grid.center)

    factor = int(new_resolution / old_grid.resolution)
    S_opt_cp = as_array_of_arrays(S_opt.copy(), dtype=int)

    def refiner(x):
        return ((x * factor)[:, np.newaxis] + np.indices([factor] * 3).T.reshape((-1, 3))).reshape((-1, 3))

    indices = refiner(S_opt_cp)
    new_grid.set_occupied(indices)
    dilation(new_grid, check=False)
    #new_grid.set_phi(indices, 1)

    new_grid.set_occupied(refiner(np.array(old_grid.voxels.nonzero()).T))
    dilation(new_grid, steps=3, check=False)
    new_grid.insert_point(old_grid.points)
    #dilation(new_grid)

    return new_grid

    """new_grid.insert_point(old_grid.points)
    for (i,j,k) in S_opt:
        new_grid.set_occupied(np.array((2*i,2*j,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i,2*j+1,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i,2*j,2*k+1), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j+1,2*k), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j,2*k+1), dtype=int))
        new_grid.set_occupied(np.array((2*i,2*j+1,2*k+1), dtype=int))
        new_grid.set_occupied(np.array((2*i+1,2*j+1,2*k+1), dtype=int))
    return new_grid"""

def undo_dilation(v_crust, dilated_voxels):
    unoccupied_voxels = dilated_voxels[~v_crust.is_point(dilated_voxels)]
    v_crust.set_unoccupied(unoccupied_voxels)
    v_crust.set_phi(unoccupied_voxels, 0)

def dilation(v_crust, steps=1, check=True):
    dilated_voxels = np.array(v_crust.voxels.nonzero()).T
    voxels_to_dilate = v_crust.get_valid_neighbors(dilated_voxels)
    for i in range(steps - 1):
        voxels_to_dilate = np.vstack((voxels_to_dilate, v_crust.get_valid_neighbors(voxels_to_dilate)))
    voxels_to_dilate = voxels_to_dilate[~v_crust.is_occupied(voxels_to_dilate)]
    if check:
        unoccupied_voxels = voxels_to_dilate[~v_crust.is_point(voxels_to_dilate)]
        v_crust.set_phi(unoccupied_voxels, 1)
    v_crust.set_occupied(voxels_to_dilate)
    return voxels_to_dilate


def diffusion(v_crust, repeat=1):
    for i in range(repeat):
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


def flooding(image, seed):
    image = image.astype(int)
    return flood(image, seed, connectivity=1)

def get_components(grid, max_components=5):
    voxels = grid.voxels.copy().astype(int)
    seed = first_zero(grid.voxels)
    cur_amount = 1
    components = []
    while seed != (-1, -1, -1):
        cur_amount += 1
        if cur_amount > max_components:
            break
        cur_component = flooding(voxels, seed)
        voxels[cur_component] = cur_amount
        components.append(np.where(cur_component))
        seed = first_zero(voxels)

    return cur_amount, components

    """seed_v_crust = first_zero(~grid.voxels)
    V_crust = np.array(flood_filling(grid.voxels, seed_v_crust)).T

    lower_bound = 1
    if V_crust.shape != np.array(grid.voxels.nonzero()).T.shape:
        lower_bound = 2

    V_ext = np.array(V_ext).T

    v_ext_tup = np.array(list(map(tuple, V_ext)), dtype="i,i,i")
    v_crust_tup = np.array(list(map(tuple, np.array(grid.voxels.nonzero()).T)), dtype="i,i,i")
    v_tup = np.array(list(map(tuple, itertools.product(*[range(grid.resolution)] * 3))), dtype="i,i,i")
    V_int = np.setdiff1d(v_tup, np.hstack((v_ext_tup, v_crust_tup)))
    V_int = np.array(list(map(list, V_int)))

    return lower_bound, V_ext, V_int"""

def as_array_of_tuples(arr):
    return np.array(list(map(tuple, arr)), dtype="i, i, i")

def as_array_of_arrays(arr, **kwargs):
    return np.array(list(map(list, arr)), **kwargs)

# the whole process from dilation, diffusion to flood filling to be used directly
def dilation_process(v_crust, reverse_steps=3):
    dilation_steps = []
    while True:
        dilated_voxels = dilation(v_crust)
        dilation_steps.append(dilated_voxels)
        amount, components = get_components(v_crust)
        print(amount)
        if amount == 2:
            dilations_to_undo = np.array(dilation_steps[-1])
            for i in range(1, min(reverse_steps, len(dilation_steps))):
                dilations_to_undo = np.vstack((dilations_to_undo, dilation_steps[-(i + 1)]))

            undo_dilation(v_crust, dilations_to_undo)
            dilations_tup = as_array_of_tuples(dilations_to_undo)

            seed = first_zero(v_crust.voxels)
            V_ext = np.array(np.where(flooding(v_crust.voxels, seed))).T
            V_ext_tup = as_array_of_tuples(V_ext)
            V_int = as_array_of_arrays(np.setdiff1d(dilations_tup, V_ext_tup))
            break

    return amount, V_ext, V_int


def plot_grid(voxels, resolution):
    voxelarray = np.zeros([resolution]*3, dtype=bool)
    voxelarray[voxels] = True
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxels] = '#7A88CCA0'

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

    plt.show()