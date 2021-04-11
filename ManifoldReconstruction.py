import GridUtils


def calc_v_ext(grid):
    v_ext = GridUtils.flood_filling(-1, -1, -1, grid)
    v_ext = list(filter(grid.is_valid, v_ext))
    return v_ext


def calc_v_int(v_ext, dilated_voxels, grid):
    voxel = next(iter(set(dilated_voxels).difference(set(v_ext))), None)
    if voxel is not None:
        v_int = GridUtils.flood_filling(voxel[0], voxel[1], voxel[2], grid)
        v_int = list(filter(grid.is_valid, v_int))
        return v_int
    return []
