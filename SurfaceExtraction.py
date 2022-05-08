import networkx as nx
import Grid
import numpy as np


def make_octahedral_subgraph(voxel, weight):
    G = nx.Graph()

    edges = ((voxel + Grid.center)[:, np.newaxis, np.newaxis] + 0.5 * Grid.edges).reshape((-1, 2, 3))
    edges = [(tuple(x[0]), tuple(x[1]), {'capacity': weight[i]}) for i,x in enumerate(edges)]

    G.add_edges_from(edges)

    return G


def generate_graph(v_crust, v_int, v_ext, s=4, a=10 ** (-5)):
    G = nx.Graph()
    G.add_node("source")  # source
    G.add_node("sink")  # sink

    voxels = v_crust.voxels.nonzero()
    weights = np.repeat(v_crust.phi[voxels], 12) ** s + a
    voxels = np.array(voxels).T
    subgraphs = make_octahedral_subgraph(voxels, weights)

    G.update(subgraphs)

    voxel_faces = (voxels[:, np.newaxis, :] + Grid.neighbors).reshape((-1, 3))
    voxel_faces_tup = np.array(list(map(tuple, voxel_faces)), dtype="i,i,i")
    v_ext_tup = np.array(list(map(tuple, v_ext)), dtype="i,i,i")
    v_int_tup = np.array(list(map(tuple, v_int)), dtype="i,i,i")

    voxel_in_ext = np.isin(voxel_faces_tup, v_ext_tup)
    voxel_in_int = np.isin(voxel_faces_tup, v_int_tup)
    edges = (voxel_faces.reshape((-1, 6, 3)) - 0.5 * Grid.neighbors + Grid.center).reshape((-1, 3))
    ext_edges = [("sink", tuple(x), {'capacity': 2.}) for x in edges[voxel_in_ext]]
    int_edges = [("source", tuple(x), {'capacity': 2.}) for x in edges[voxel_in_int]]

    G.add_edges_from(ext_edges)
    G.add_edges_from(int_edges)

    return G


def is_in_component(v_component, voxel):
    return (v_component[:, np.newaxis] == voxel).all(-1).any(0)


def calc_s_opt(graph):
    ut_value, partition = nx.minimum_cut(graph, "source", "sink")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    S_opt = set()
    for cutted_edge in cutset:
        for a, b in Grid.edges:
            # TODO: quite ugly to read; how to improve?
            # first check if edge (a, b) gives the same voxel center
            if np.array_equal(np.array(cutted_edge[0]) - a * 0.5, np.array(cutted_edge[1]) - b * 0.5):
                voxel = cutted_edge[0] - a * 0.5 - Grid.center
                if not np.array_equal(voxel.astype(int), voxel):
                    continue
                S_opt.add(tuple(voxel))
                break

            # if it doesn't hold for edge (a, b), check for edge (b, a)
            if np.array_equal(np.array(cutted_edge[0]) - b * 0.5, np.array(cutted_edge[1]) - a * 0.5):
                voxel = cutted_edge[0] - b * 0.5 - Grid.center
                if not np.array_equal(voxel.astype(int), voxel):
                    continue
                S_opt.add(tuple(voxel))
                break
    return S_opt, cutset