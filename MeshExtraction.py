import Grid
import numpy as np
import pymesh

block = np.array([
    Grid.right,
    Grid.back,
    Grid.back + Grid.right,
    Grid.up,
    Grid.up + Grid.back,
    Grid.up + Grid.right,
    Grid.up + Grid.right + Grid.back
])


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

    for block in valid_blocks:
        block_center = get_block_center(block)
        # get first voxel of current block; current block is always non-empty
        for first_voxel in intersect_block(s_opt, block):
            break
        adj_edges = get_center_adjacent_edges(first_voxel, block_center)
        adj_cut_edges = intersect_edges(cut_edges, adj_edges)

        # get first cut adj edge;
        v = first_voxel
        vertices.update(v)
        it = iter(adj_cut_edges)
        e = next(it)
        while True:
            f = next(it)
            #if f == e:
            #    f = next(it)
            w = get_neighbor_voxel(v, f, block_center)
            edges.update((v, tuple(w)))
            vertices.update(tuple(w))

            v = w
            e = f

            if np.array_equal(v, first_voxel):
                break

            adj_edges = get_center_adjacent_edges(v, block_center)
            adj_cut_edges = intersect_edges(cut_edges, adj_edges)
            it = iter(adj_cut_edges)

    vertices = np.array(vertices)
    edges = np.array([[get_index(vertices, list(edge)[0]), get_index(vertices, list(edge)[1])] for edge in edges])
    vertices = np.array(map(grid.get_voxel_center, vertices))
    return pymesh.wires.WireNetwork.create_from_data(vertices, edges)


"""def extract_mesh(s_opt, cut_edges, grid):
    vertices = set([])
    edges = set([])

    for voxel in s_opt:
        blocks = filter(lambda x: pred_intersect(s_opt, x), get_blocks(voxel))
        for b in blocks:
            block_center = get_block_center(b)
            # get first voxel of current block; current block is always non-empty
            for first_voxel in b:
                break
            adj_edges = get_center_adjacent_edges(first_voxel, block_center)
            adj_cut_edges = intersect_edges(cut_edges, adj_edges)
            # get first cut adj edge;
            v = first_voxel
            vertices.update(tuple(v))
            it = iter(adj_cut_edges)
            e = next(it)
            while True:
                f = next(it)
                if f == e:
                    f = next(it)
                w = get_neighbor_voxel(v, f, block_center)
                edges.update(set((tuple(v), tuple(w))))
                vertices.update(tuple(w))

                v = w
                e = f

                if v == first_voxel:
                    break

                adj_edges = get_center_adjacent_edges(v, block_center)
                adj_cut_edges = intersect_edges(cut_edges, adj_edges)
                it = iter(adj_cut_edges)

    vertices = np.array(vertices)
    edges = np.array([[get_index(vertices, list(edge)[0]), get_index(vertices, list(edge)[1])] for edge in edges])
    vertices = np.array(map(grid.get_voxel_center, vertices))
    return pymesh.wires.WireNetwork.create_from_data(vertices, edges)"""


def get_index(array, elem):
    return np.where(np.all(array == elem, axis=1))[0]


def get_neighbor_voxel(v, edge, center):
    a, b = edge
    center_norm = center / np.linalg.norm(center)
    # vec = (b - (v + Grid.center)) - (a - (v + Grid.center))
    vec = np.array(b) - np.array(a)
    vec = sign(np.dot(center_norm, vec)) * vec
    vec = vec / np.linalg.norm(vec)
    return v + vec


def intersect_edges(cut_edges, adj_edges):
    filtered_edges = []
    for a, b in adj_edges:
        if (a, b) in cut_edges or (b, a) in cut_edges:
            filtered_edges.append((a, b))
    return filtered_edges


def get_center_adjacent_edges(voxel, center):
    voxel_center = voxel + Grid.center
    vec = center - voxel
    vec /= np.linalg.norm(vec)
    x = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([1, 0, 0]))) * np.array([1, 0, 0]))
    y = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([0, 1, 0]))) * np.array([0, 1, 0]))
    z = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([0, 0, 1]))) * np.array([0, 0, 1]))
    return [(x, y), (x, z), (y, z)]


def get_block_center(b):
    return sum([v + Grid.center for v in b]) / len(b)


def intersect_block(s_opt, b):
    return s_opt.intersection(set(map(tuple, b)))

def pred_intersect(s_opt, b):
    return len(intersect_block(s_opt, b)) >= 3


def get_blocks(voxel):
    blocks = set()
    for i in [0, -1]:
        x = np.array([i, 0, 0])
        for j in [0, -1]:
            y = np.array([0, j, 0])
            for k in [0, -1]:
                z = np.array([0, 0, k])
                cur_block = set()
                for direction in block:
                    cur_block.update(tuple(voxel + x + y + z + direction))
                blocks.update(cur_block)
    return blocks
