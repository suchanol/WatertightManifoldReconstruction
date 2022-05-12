import Grid
import GridUtils
import MeshExtraction
import SurfaceExtraction

def reconstruct_manifold(filename_point_cloud, resolutions=[64, 128, 256]):
    grid = Grid.Grid()
    grid.make_grid_from_file(filename_point_cloud, resolutions[0])

    for l in range(0, len(resolutions)):
        print("start process for resolution:", resolutions[l])
        # surface confidence estimation
        print("start dilation process")
        cur_amount, cur_comp = GridUtils.get_components(grid)
        lower_bound, V_ext, V_int = GridUtils.dilation_process(grid)
        print("start diffusion")
        GridUtils.diffusion(grid, repeat=3)

        # graph-based surface extraction
        print("start graph generation")
        graph = SurfaceExtraction.generate_graph(grid, V_int, V_ext)
        print("start calculation of s_opt")
        S_opt, cut_edges = SurfaceExtraction.calc_s_opt(graph)

        # volumetric refinement
        if l < len(resolutions) - 1:
            print("start grid refinement")
            grid = GridUtils.refine_S_opt_grid(grid, S_opt, resolutions[l + 1])

    # mesh extraction
    print("start mesh extraction")
    mesh = MeshExtraction.extract_mesh(S_opt, cut_edges, grid)
    print("extracted mesh")
    mesh.write_to_file("debug.obj")