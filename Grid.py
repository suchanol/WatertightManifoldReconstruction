import numpy as np
import plyfile

up = + np.array([0, 1, 0])  # up
down = - np.array([0, 1, 0])  # down
front = + np.array([0, 0, 1])  # front
back = - np.array([0, 0, 1])  # back
right = + np.array([1, 0, 0])  # right
left = - np.array([1, 0, 0])  # left
neighbors = [up, down, front, back, right, left]
edges = [(front, up), (front, down), (front, right), (front, left),
         (right, back), (right, up), (right, down),
         (left, up), (left, down), (left, back),
         (back, up), (back, down)]

center = np.array([1, 1, 1]) / 2.0


class Grid:

    # def __init__(self, points=[], resolution=0):
    #     self.voxel_size = 0
    #     if points and resolution != 0:
    #         self.resolution = resolution
    #         self.make_grid(points, resolution)
    #     else:
    #         self.resolution = 0
    #         self.points = np.array([np.empty(3)], dtype=int)
    #         self.voxels = np.array([np.empty(3)], dtype=int)
    def __init__(self):
        self.voxel_size = 0
        self.points = []
        self.resolution = 0
        self.voxels = []

    def make_grid_from_file(self, file, resolution):
        plydata = plyfile.PlyData.read(file)
        point_cloud = np.zeros(shape=(plydata['vertex'].count, 3), dtype=np.float32)
        point_cloud[:, 0] = plydata['vertex'].data['x']
        point_cloud[:, 1] = plydata['vertex'].data['y']
        point_cloud[:, 2] = plydata['vertex'].data['z']
        self.make_grid(point_cloud, resolution)

    def make_grid(self, points, resolution):
        self.resolution = resolution
        max_coord = np.max(points)
        self.voxel_size = max_coord / self.resolution
        for p in points:
            self.insert_point(p)

    def get_voxel_center(self, v):
        return v * self.voxel_size + np.ones((3, 1)) * self.voxel_size / 2.0

    def get_voxels(self):
        return np.unique(self.voxels, axis=0)

    def get_voxel(self, p):
        return (d // self.voxel_size for d in p)

    def get_points(self):
        return self.points

    # p = (x,y,z)
    # divide the x y z by the dimension of the voxel cube, the rounded down number indicates the indices
    # of the voxel containing the points in a 3d matrix
    # return the index of the voxel
    def insert_point(self, p):
        # remark: the points should be stored somewhere I put a array to store them, you can change it to a different
        # container
        voxel = [d // self.voxel_size for d in p]
        self.voxels.append(voxel)
        self.points.append(p)
        return voxel

    def get_valid_neighbors(self, vi):
        neighbors = list(filter(self.is_valid, self.get_neighbors(vi)))
        return neighbors

    # (up, down, front, back, right, left)
    # vi = (x, y, z)
    # the input is the voxel index return a list of indexes
    @staticmethod
    def get_neighbors(vi):
        return np.array([vi + direction for direction in neighbors])

    def is_valid(self, v):
        return np.logical_and.reduce([0 <= x < self.resolution for x in v])

    def is_point(self, vi):
        if not self.is_occupied(tuple(vi)):
            return False
        itemindex = np.unique(np.where(self.voxels == vi)[0], axis=0)
        if np.array_equal(np.unique(self.points[itemindex], axis=0), [[-1, -1, -1]]):
            return False
        return True

    def is_occupied(self, vi):
        return vi in self.voxels

    def set_occupied(self, vi):
        self.voxels.append(vi)
        self.points.append([-1, -1, -1])

    # we avoid negative indexes therefore starting with 0 up to resolution
    # remark: not sure we need this, the resolution should be enough
    def get_bounds(self):
        return self.resolution * self.voxel_size
