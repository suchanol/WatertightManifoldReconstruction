from matplotlib import pyplot as plt

import Grid
import numpy as np
import networkx as nx

import GridUtils
import MeshExtraction
import SurfaceExtraction

grid = Grid.Grid()
grid.make_grid_from_file("bun_zipper_res3.ply", 128)
#GridUtils.refine_grid(grid, 128)
GridUtils.dilation(grid)
GridUtils.diffusion(grid)

lower_bound, V_ext, V_int = GridUtils.get_lower_bounds_on_components(grid)
print("lower bound")

graph = SurfaceExtraction.generate_graph(grid, V_int, V_ext)
print("generated graph")
S_opt, cut_edges = SurfaceExtraction.calc_s_opt(graph)
print("calculated s_opt")

print("extract mesh")
mesh = MeshExtraction.extract_mesh(S_opt, cut_edges, grid)
print("extracted mesh")
mesh.write_to_file("debug.wire")
print("cake")