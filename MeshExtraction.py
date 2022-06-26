import sys

import numpy as np

import Grid

def smoothing(grid, vertices, edges):
    np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=2000)
    max_vert = vertices.shape[0]

    A = np.zeros((max_vert, max_vert))
    A[tuple(edges.T)] = 1
    b = np.array((edges.T[1],edges.T[0]))
    A[tuple(b)] = 1


    D = np.sum(A, axis=1)
    lambda_v = (A @ D)/D + 1
    lambda_v = np.diag(1/lambda_v)
    D = np.diag(1 / D)
    L = np.identity(max_vert) - D @ A
    new_vertices = vertices - lambda_v @ (L @ (L @ vertices))
    # new_vertices = vertices - (L @ (L @ vertices))



    stopping_criterion = np.linalg.norm(new_vertices - vertices, axis=1)  \
        < grid.voxel_size * (grid.phi[tuple(vertices.astype(int).T)] + 1)

    for i,stop in enumerate(stopping_criterion):
        if not stop:
            new_vertices[i] = vertices[i]
    np.save('vertices_smooth.npy', new_vertices)

def extract_mesh(s_opt, cut_edges, grid):
    vertices = set([])
    faces = set([])
    edges = set([])
    first_block = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    ranges = np.arange(grid.resolution - 1)
    displacements = np.array(np.meshgrid(ranges, ranges, ranges)).T.reshape(-1, 3)
    all_blocks = first_block[np.newaxis, :] + displacements[:, np.newaxis, :]
    valid_blocks = filter(lambda x: pred_intersect(s_opt, x), all_blocks)
    for block in valid_blocks:
        start_voxel = block[0]
        start_edge = []
        next_edge = []
        block_center = get_block_center(block)

        # find the first voxel containing min_cut edges
        for voxel in block:
            adj_edges = get_center_adjacent_edges(voxel, block_center)
            adj_cut_edges = intersect_edges(cut_edges, adj_edges)
            if len(adj_cut_edges) == 2:
                start_voxel = voxel
                start_edge = from_cut_edge_to_normal_edges(adj_cut_edges[1])
                next_edge = from_cut_edge_to_normal_edges(adj_cut_edges[0])
                break

        v = start_voxel

        # first cut edges
        e = start_edge
        f = next_edge
        triangle = set([])
        triangle.add(tuple(start_voxel))
        # vertices.add(tuple(v))
        while (True):
            if f == start_edge:
                break
            # find next voxel
            w = get_neighbor_voxel(v, f, block, block_center, cut_edges)
            vertices.add(tuple(v))
            vertices.add(tuple(w))
            triangle.add(tuple(w))
            edges.add((tuple(v), tuple(w)))
            if tuple(w) != tuple(start_voxel):
                edges.add((tuple(start_voxel), tuple(w)))
            if len(triangle)==3:
                faces.add(tuple(triangle))
                triangle = set([])
                triangle.add(tuple(start_voxel))
                triangle.add(tuple(w))
            v = w
            e = f
            adj_edges = get_center_adjacent_edges(w, block_center)
            adj_cut_edges = intersect_edges(cut_edges, adj_edges)
            normal_edge1 = from_cut_edge_to_normal_edges(adj_cut_edges[0])
            normal_edge2 = from_cut_edge_to_normal_edges(adj_cut_edges[1])
            if e == normal_edge1:
                f = normal_edge2
            elif e == normal_edge2:
                f = normal_edge1

    vertices = np.array(list(vertices))
    edges = np.array([[get_index(vertices, a)[0], get_index(vertices, b)[0]] for (a, b) in edges])
    faces = np.array(
        [[get_index(vertices, a)[0], get_index(vertices, b)[0], get_index(vertices, c)[0]] for (a, b, c) in faces])
    vertices = grid.get_voxel_center(vertices)

    np.save('vertices.npy', vertices)
    np.save('faces.npy', faces)
    np.save('edges.npy', edges)
def remove_voxel_from_block(voxel, block):
    result = set([])
    for (x,y,z) in block:
        if x!=voxel[0] and y!=voxel[1] and z!=voxel[2]:
            result.add((x,y,z))
    return result
def pred_intersect(s_opt, b):
    return len(intersect_block(s_opt, b)) >= 3

def intersect_block(s_opt, b):
    return s_opt.intersection(set(map(tuple, b)))

def get_block_center(b):
    return np.mean(b, axis=0) + Grid.center

def intersect_edges(cut_edges, adj_edges):
    filtered_edges = []
    for (a, b) in adj_edges:
        if (a, b) in cut_edges or (b, a) in cut_edges:
            filtered_edges.append((a, b))
    return filtered_edges

def get_center_adjacent_edges(voxel, center):
    voxel_center = voxel + Grid.center
    vec = center - voxel_center
    vec /= np.linalg.norm(vec)
    x = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([1, 0, 0]))) * np.array([1, 0, 0]))
    y = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([0, 1, 0]))) * np.array([0, 1, 0]))
    z = tuple(voxel_center + 0.5 * sign(np.dot(vec, np.array([0, 0, 1]))) * np.array([0, 0, 1]))
    return [(x, y), (x, z), (y, z)]


def sign(x):
    return (1, -1)[x < 0]

def get_neighbor_voxel(v, current_edge, block, block_center, cut_edges):

    block_wo_v = block

    for next_voxel in block:
        if tuple(v) == tuple(next_voxel):
            continue
        adj_edges = get_center_adjacent_edges(next_voxel, block_center)
        adj_cut_edges = intersect_edges(cut_edges, adj_edges)
        if len(adj_cut_edges) < 2:
            continue
        normal_edge1 = from_cut_edge_to_normal_edges(adj_cut_edges[0])
        normal_edge2 = from_cut_edge_to_normal_edges(adj_cut_edges[1])
        if edges_equal(normal_edge1, current_edge) or edges_equal(normal_edge2,current_edge):
            return next_voxel

def from_cut_edge_to_normal_edges(cut_edge):
    a = cut_edge[0]
    b = cut_edge[1]
    r1 = np.zeros(3)
    r2 = np.zeros(3)
    for i in range(0,3):
        if a[i] == b[i]:
            r1[i] = float(a[i])-0.5
            r2[i] = float(a[i])+0.5
        elif (float(a[i])*2)%2 == 0:
            r1[i] = a[i]
            r2[i] = a[i]
        elif (float(b[i])*2)%2 == 0:
            r1[i] = b[i]
            r2[i] = b[i]
    return (tuple(r1), tuple(r2))

def get_natural_center_edges(voxel, center):
    midpoint = tuple(center)
    vec = center - voxel
    vec /= np.linalg.norm(vec)
    x = tuple(center - sign(np.dot(vec, np.array([1, 0, 0]))) * np.array([1, 0, 0]))
    y = tuple(center - sign(np.dot(vec, np.array([0, 1, 0]))) * np.array([0, 1, 0]))
    z = tuple(center - sign(np.dot(vec, np.array([0, 0, 1]))) * np.array([0, 0, 1]))
    return [(midpoint, x), (midpoint, y), (midpoint, z)]

def get_index(array, elem):
    return np.where(np.all(array == elem, axis=-1))[0]

def edges_equal(e1, e2):
    if e1[0] == e2[0] and e1[1] == e2[1]:
        return True
    elif e1[0] == e2[1] and e1[1] == e2[0]:
        return True
    return False