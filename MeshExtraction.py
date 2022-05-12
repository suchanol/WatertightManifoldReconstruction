import Grid
import numpy as np
import pymesh
import GridUtils


"""block = np.array([
    Grid.right,
    Grid.back,
    Grid.back + Grid.right,
    Grid.up,
    Grid.up + Grid.back,
    Grid.up + Grid.right,
    Grid.up + Grid.right + Grid.back
])"""


def sign(x):
    return (1, -1)[x < 0]


def extract_mesh(s_opt, cut_edges, grid):
    vertices = set([])
    edges = set([])

    first_block = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    ranges = np.arange(grid.resolution - 1)
    displacements = np.array(np.meshgrid(ranges, ranges, ranges)).T.reshape(-1, 3)
    all_blocks = first_block[np.newaxis, :] + displacements[:, np.newaxis, :]

    valid_blocks = filter(lambda x: pred_intersect(s_opt, x), all_blocks)
    #print(len(list(valid_blocks)))

    for block in valid_blocks:
        block_center = get_block_center(block)
        # get first voxel of current block; current block is always non-empty
        block_cut = intersect_block(s_opt, block)
        for first_voxel in block_cut:
            break
        adj_edges = get_center_adjacent_edges(first_voxel, block_center)
        adj_cut_edges = intersect_edges(cut_edges, adj_edges)
        if len(adj_cut_edges) <= 0:
            continue
        # get first cut adj edge;
        v = first_voxel
        vertices.add(v)
        it = iter(adj_cut_edges)
        # e = next(it)
        f = next(it)
        block_cp = np.array(list(map(np.array, block_cut.copy())))
        while True:
            # f = next(it)
            # if f == e:
            #    f = next(it)
            equals_v = (block_cp == v).all(-1)
            block_cp = block_cp[~equals_v]
            block_cut = block_cp
            w = get_neighbor_voxel(v, f, block_cut, block_center, first_voxel)
            edges.add((tuple(v), tuple(w)))
            vertices.add(tuple(w))

            v = w
            e = f

            if np.array_equal(v, first_voxel):
                break

            adj_edges = get_center_adjacent_edges(v, block_center)
            adj_cut_edges = intersect_edges(cut_edges, adj_edges)
            it = iter(adj_cut_edges)

    vertices = np.array(list(vertices))
    edges = np.array([[get_index(vertices, a)[0], get_index(vertices, b)[0]] for (a, b) in edges])
    vertices = grid.get_voxel_center(vertices)

    return pymesh.wires.WireNetwork.create_from_data(vertices, edges)


def get_index(array, elem):
    return np.where(np.all(array == elem, axis=-1))[0]


def get_neighbor_voxel(v, edge, block, block_center, first_voxel):
    #block_cp = np.array(list(map(np.array, block.copy())))
    #equals_v = (block_cp == v).all(1)
    #block_wo_v = block_cp[~equals_v]
    block_wo_v = block

    edges_v = get_natural_center_edges(v, block_center)
    block_edges = list(map(lambda x: get_natural_center_edges(x, block_center), block_wo_v))

    for i, cur_block in enumerate(block_edges):
        for cur_edge in cur_block:
            if cur_edge in edges_v:
                return block_wo_v[i]

    return first_voxel


def intersect_edges(cut_edges, adj_edges):
    filtered_edges = []
    for a, b in adj_edges:
        if (a, b) in cut_edges or (b, a) in cut_edges:
            filtered_edges.append((a, b))
    return filtered_edges

def get_natural_center_edges(voxel, center):
    midpoint = tuple(center)
    vec = center - voxel
    vec /= np.linalg.norm(vec)
    x = tuple(center - sign(np.dot(vec, np.array([1, 0, 0]))) * np.array([1, 0, 0]))
    y = tuple(center - sign(np.dot(vec, np.array([0, 1, 0]))) * np.array([0, 1, 0]))
    z = tuple(center - sign(np.dot(vec, np.array([0, 0, 1]))) * np.array([0, 0, 1]))
    return [(midpoint, x), (midpoint, y), (midpoint, z)]

def get_center_adjacent_edges(voxel, center):
    voxel_center = voxel + Grid.center
    vec = center - voxel_center
    vec /= np.linalg.norm(vec)
    x = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([1, 0, 0]))) * np.array([1, 0, 0]))
    y = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([0, 1, 0]))) * np.array([0, 1, 0]))
    z = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([0, 0, 1]))) * np.array([0, 0, 1]))
    return [(x, y), (x, z), (y, z)]


def get_block_center(b):
    return np.mean(b, axis=0) + Grid.center


def intersect_block(s_opt, b):
    return s_opt.intersection(set(map(tuple, b)))

def pred_intersect(s_opt, b):
    return len(intersect_block(s_opt, b)) >= 3

def smoothing(grid, vertices, edges):
    max_vert = vertices.shape[0]

    A = np.zeros((max_vert, max_vert))
    A[tuple(edges.T)] = 1

    D = np.sum(A, axis=1)

    valences = A.nonzero()
    lambda_v = np.ones((max_vert, ))
    lambda_v[valences[0]] += D[valences[1]]
    lambda_v /= D
    lambda_v += 1

    D = np.diag(D / lambda_v)

    L = np.identity(max_vert) - D @ A

    new_vertices = L @ vertices

    stopping_criterion = np.linalg.norm(new_vertices - vertices, axis=1)  \
        < grid.voxel_size * (grid.phi[tuple(vertices.astype(int).T)] + 1)

    new_vertices[stopping_criterion] = vertices[stopping_criterion]
    mesh = pymesh.wires.WireNetwork.create_from_data(new_vertices, edges)
    mesh.write_to_file("new_lap.obj")