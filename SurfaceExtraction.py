import networkx as nx
import Grid
import numpy as np
import click

from Grid import center
from GridUtils import if_voxel_in_list


def make_octahedral_subgraph(voxel, weight):
    G = nx.Graph()

    # for direction in GriGridUtils.dilation(grid)d.neighbors:
    #     # half direction so that voxel faces can be easily joined by matching node names
    #     G.add_node(tuple(voxel + center + direction * 0.5))

    for a, b in Grid.edges:
        G.add_edge(tuple(voxel + center + a * 0.5), tuple(voxel + center + b * 0.5), capacity=weight)

    return G


# def generate_graph(v_crust, phi, v_int, v_ext, s=4, a=10**(-5)):
#     G = nx.Graph()
#     G.add_node("source")  # source
#     G.add_node("sink")  # sink
#     with click.progressbar(v_crust.get_voxels()) as bar:
#         for voxel in bar:
#             weight = phi[tuple(voxel)]**s + a
#             subgraph = make_octahedral_subgraph(voxel, weight)
#             G.update(subgraph)  # join graphs
#             # connect to sink and source respectively
#             for direction in Grid.neighbors:
#                 face = voxel + direction
#                 if face in v_ext or not v_crust.is_valid(face):
#                     G.add_edge("sink", tuple(voxel + center + direction * 0.5), capacity=weight)
#
#                 if face in v_int:
#                     G.add_edge("source", tuple(voxel + center + direction * 0.5), capacity=weight)
#     return G

def generate_graph(v_crust, phi, v_int, v_ext, s=4, a=10 ** (-5)):
    G = nx.Graph()
    G.add_node("source")  # source
    G.add_node("sink")  # sink
    for voxel in v_crust.voxels:
        weight = phi[tuple(voxel)] ** s + a
        subgraph = make_octahedral_subgraph(voxel, weight)
        G.update(subgraph)  # join graphs
        # connect to sink and source respectively
        for direction in Grid.neighbors:
            face = voxel + direction
            if if_voxel_in_list(v_ext, face) or not v_crust.is_valid(face):
                G.add_edge("sink", tuple(voxel + center + direction * 0.5), capacity=2)
            if if_voxel_in_list(v_int, face):
                G.add_edge("source", tuple(voxel + center + direction * 0.5), capacity=2)
    return G


def calc_s_opt(graph):
    cut_value, partition = nx.minimum_cut(graph, "source", "sink")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    S_opt = set()
    for cutted_edge in cutset:
        for a, b in Grid.edges:
            if np.array_equal(np.array(cutted_edge[0]) - a * 0.5, np.array(cutted_edge[1]) - b * 0.5):
                voxel = cutted_edge[0] - a * 0.5 - center
                # S_opt.update(np.array(cutted_edge[0]) - a * 0.5 - center) here update will breack the structure add is needed
                S_opt.add((voxel[0], voxel[1], voxel[2]))
                break
    return S_opt, cutset
