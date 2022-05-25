import Grid
import GridUtils
import MeshExtraction
import SurfaceExtraction
import numpy as np

def reconstruct_manifold(filename_point_cloud, resolutions=[32, 64, 128]):
    grid = Grid.Grid()
    grid.make_grid_from_file(filename_point_cloud, resolutions[0])

    for l in range(0, len(resolutions)):
        print("start process for resolution:", resolutions[l])
        # surface confidence estimation
        print("start dilation process")
        # GridUtils.plot_grid(grid.voxels, grid.resolution)
        print(len(grid.points))
        cur_amount, cur_comp = GridUtils.get_components(grid)
        if l == 0:
            rev_steps = 4
        elif l == 1:
            rev_steps = 8
        elif l == 2:
            rev_steps = 20
        lower_bound, V_ext, V_int = GridUtils.dilation_process(grid,reverse_steps=rev_steps)
        # GridUtils.plot_grid(grid.voxels, grid.resolution)
        print("start diffusion")
        GridUtils.diffusion(grid, repeat=1)

        # graph-based surface extraction
        print("start graph generation")
        graph = SurfaceExtraction.generate_graph(grid, V_int, V_ext)
        print("start calculation of s_opt")
        S_opt, cut_edges = SurfaceExtraction.calc_s_opt(graph, V_int, V_ext)
        # if l == len(resolutions)-1:
        #     voxelarray = np.zeros([resolutions[l]] * 3, dtype=bool)
        #     for i in S_opt:
        #         voxelarray[i] = True
        #     GridUtils.plot_grid(tuple(V_int.T), grid.resolution)
        #     GridUtils.plot_grid(voxelarray, grid.resolution)
        del graph
        # volumetric refinement
        if l < len(resolutions) - 1:
            print("start grid refinement")
            grid = GridUtils.refine_S_opt_grid(grid, S_opt, resolutions[l + 1])


    # mesh extraction
    print("start mesh extraction")
    mesh = MeshExtraction.extract_mesh(S_opt, cut_edges, grid)
    print("extracted mesh")
    mesh.write_to_file("debug.obj")