from matplotlib import pyplot as plt

import Grid
import numpy as np
import networkx as nx

import GridUtils
import MeshExtraction
import SurfaceExtraction

grid = Grid.Grid()
grid.make_grid_from_file("bun_zipper_res3.ply", 64)
#GridUtils.refine_grid(grid, 128)

lower_bound, V_ext, V_int = GridUtils.dilation_process(grid)
print("lower bound")
print(lower_bound)
print(V_ext)
print(V_int)
graph = SurfaceExtraction.generate_graph(grid, V_int, V_ext)
print("generated graph")
S_opt, cut_edges = SurfaceExtraction.calc_s_opt(graph)
print("calculated s_opt")
new_grid = GridUtils.refine_S_opt_grid(grid, S_opt, 128)
print('refine the grid')

# print("extract mesh")
# mesh = MeshExtraction.extract_mesh(S_opt, cut_edges, grid)
# print("extracted mesh")
# mesh.write_to_file("debug.wire")
# print("cake")