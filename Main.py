import numpy
import pymesh

#
import ManifoldReconstruction
ManifoldReconstruction.reconstruct_manifold("samples/bun_zipper_res2.ply")
#
faces = numpy.load('samples/faces.npy', allow_pickle=True)
vertices = numpy.load('samples/vertices.npy')
edges = numpy.load('samples/edges.npy')
mesh = pymesh.form_mesh(vertices, faces)
pymesh.save_mesh("samples/debug_faces.obj", mesh)
mesh_wired = pymesh.wires.WireNetwork.create_from_data(vertices, edges)
mesh_wired.write_to_file("samples/debug_wired.obj")
vertices_smooth = numpy.load('samples/vertices_smooth.npy')
mesh_smooth = pymesh.form_mesh(vertices_smooth, faces)
pymesh.save_mesh("samples/debug_smooth.obj", mesh_smooth)

"""grid = Grid.Grid()
grid.make_grid_from_file("bun_zipper_res3.ply", 32)
#GridUtils.refine_grid(grid, 128)

lower_bound, V_ext, V_int = GridUtils.dilation_process(grid)
print("lower bound")
print(lower_bound)
print(V_ext)
print(V_int)

voxelarray = np.zeros((grid.resolution, grid.resolution, grid.resolution), dtype=bool)
voxelarray[grid.voxels.nonzero()] = True
voxelarray[tuple(V_int.T)] = True
colors = np.empty(voxelarray.shape, dtype=object)
colors[grid.voxels.nonzero()] = '#7A88CCA0'
colors[tuple(V_int.T)] = '#FFD65DC0'

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

plt.show()

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

graph = SurfaceExtraction.generate_graph(grid, V_int, V_ext)
print("generated graph")
S_opt, cut_edges = SurfaceExtraction.calc_s_opt(graph)
print("calculated s_opt")
new_grid = GridUtils.refine_S_opt_grid(grid, S_opt, 128)
print('refine the grid')

print("extract mesh")
mesh = MeshExtraction.extract_mesh(S_opt, cut_edges, grid)
print("extracted mesh")
mesh.write_to_file("debug.obj")
print("cake")"""