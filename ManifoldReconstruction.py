import Grid
import GridUtils
import SurfaceExtraction


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

def reconstruct_manifold(filename_point_cloud, resolutions=[64, 128, 256]):
    grid = Grid.Grid()
    grid.make_grid_from_file(filename_point_cloud, resolutions[0])

    for l in range(0, len(resolutions)):
        GridUtils.dilation(grid)
        lower_bound, V_ext, V_int = GridUtils.get_lower_bounds_on_components(grid)
        while lower_bound > 1:
            GridUtils.dilation(grid)
            lower_bound, V_ext, V_int = GridUtils.get_lower_bounds_on_components(grid)

        GridUtils.diffusion(grid)

        graph = SurfaceExtraction.generate_graph(grid, grid.phi, V_int, V_ext)