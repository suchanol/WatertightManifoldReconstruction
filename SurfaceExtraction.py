import networkx as nx
import Grid
import numpy as np

from GridUtils import as_array_of_tuples, plot_dual, plot_dual_edges


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
    weights = np.repeat(v_crust.phi[voxels], Grid.edges.shape[0]) ** s + a
    voxels = np.array(voxels).T
    subgraphs = make_octahedral_subgraph(voxels, weights)

    G.update(subgraphs)

    voxel_faces = (voxels[:, np.newaxis, :] + Grid.neighbors).reshape((-1, 3))
    voxel_faces_tup = as_array_of_tuples(voxel_faces)
    v_ext_tup = as_array_of_tuples(v_ext)
    v_int_tup = as_array_of_tuples(v_int)

    voxel_in_ext = np.isin(voxel_faces_tup, v_ext_tup)
    voxel_in_int = np.isin(voxel_faces_tup, v_int_tup)

    edges = (voxels[:, np.newaxis, :] + Grid.center + 0.5 * Grid.neighbors).reshape((-1, 3))
    ext_edges = [("sink", tuple(x)) for x in edges[voxel_in_ext]]
    int_edges = [("source", tuple(x)) for x in edges[voxel_in_int]]
    G.add_edges_from(ext_edges)
    G.add_edges_from(int_edges)

    return G


def is_in_component(v_component, voxel):
    return (v_component[:, np.newaxis] == voxel).all(-1).any(0)


def calc_s_opt(graph, V_int, V_ext):
    """v_int_faces_tup = V_int[:, np.newaxis] + 0.5 * Grid.neighbors + Grid.center
    v_int_faces_tup = as_array_of_tuples(v_int_faces_tup.reshape((-1, 3)), "f, f, f")
    v_ext_faces_tup = V_ext[:, np.newaxis] + 0.5 * Grid.neighbors + Grid.center
    v_ext_faces_tup = as_array_of_tuples(v_ext_faces_tup.reshape((-1, 3)), "f, f, f")
    v_components = {"source": v_int_faces_tup, "sink": v_ext_faces_tup}"""
    cut_value, partition = nx.minimum_cut(graph, "source", "sink")
    print("calculated min cut")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
        if u == 'source' or u == 'sink':
            print([v for v in nbrs if v in non_reachable])
            continue
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    S_opt = set()
    for cutted_edge in cutset:
        """if "source" in cutted_edge or "sink" in cutted_edge:
            v_comp = v_components[cutted_edge[0]]
            face = cutted_edge[1]


            continue"""

        edge_arr = np.array(cutted_edge)
        possible_voxels = np.floor(edge_arr[:, np.newaxis] - (edge_arr - np.floor(edge_arr))).reshape((-1, 3))
        possible_voxels = np.unique(as_array_of_tuples(possible_voxels), return_counts=True)
        final_voxel = possible_voxels[0][np.argmax(possible_voxels[1])]
        S_opt.add(tuple(final_voxel))
    return S_opt, cutset
