import plyfile
import numpy as np

up = + np.array([0, 1, 0])  # up
down = - np.array([0, 1, 0])  # down
front = + np.array([0, 0, 1])  # front
back = - np.array([0, 0, 1])  # back
right = + np.array([1, 0, 0])  # right
left = - np.array([1, 0, 0])  # left
neighbors = np.array([up, down, front, back, right, left])
edges = np.array([[front, up], [front, down], [front, right], [front, left],
         [right, back], [right, up], [right, down],
         [left, up], [left, down], [left, back],
         [back, up], [back, down]])

center = np.array([1, 1, 1]) / 2.0

class Grid:

    def __init__(self):
        self.points = np.array([])
        self.voxels = np.array([])
        self.phi = np.array([])
        self.voxel_size = 0
        self.resolution = 0
        self.min_coord = 0
        self.center = np.array([])

    def change_grid(self, resolution, voxel_size,  center):
        self.resolution = resolution
        self.voxel_size = voxel_size
        self.center = center

        self.voxels = np.zeros([self.resolution] * 3, bool)
        self.phi = np.zeros(self.voxels.shape)

    def make_grid_from_file(self, file, resolution):
        plydata = plyfile.PlyData.read(file)
        point_cloud = np.zeros(shape=(plydata['vertex'].count, 3), dtype=np.float32)
        point_cloud[:, 0] = plydata['vertex'].data['x']
        point_cloud[:, 1] = plydata['vertex'].data['y']
        point_cloud[:, 2] = plydata['vertex'].data['z']
        """
        point_cloud = np.load(file)"""
        self.make_grid(point_cloud, resolution)

    def make_grid(self, points, resolution):
        self.resolution = resolution
        bbox = [np.min(points, axis=0), np.max(points, axis=0)]
        self.center = np.mean(bbox, axis=0)
        max_coord = np.max(points)
        min_coord = np.min(points)
        self.voxel_size = (max_coord - min_coord) / self.resolution
        self.points = np.array([])

        self.voxels = np.zeros([self.resolution] * 3, bool)
        self.phi = np.zeros(self.voxels.shape)
        self.insert_point(points)

    def insert_point(self, point):
        if self.points.size == 0:
            self.points = point.reshape((-1, 3))
        else:
            self.points = np.append(self.points, point, axis=-1)

        voxel_indices = self.get_voxel(point)
        self.voxels[tuple(voxel_indices.T)] = True
        self.phi[tuple(voxel_indices.T)] = 0

    def get_voxel_center(self, voxel):
        return voxel * self.voxel_size + self.voxel_size / 2.

    def get_voxel(self, point):
        return((point - self.center + self.voxel_size * self.resolution / 2) // self.voxel_size).astype(int)

    def get_valid_neighbors(self, voxel):
        all_neighbors = self.get_neighbors(voxel)
        valid_neighbors = self.is_valid(all_neighbors)
        return all_neighbors[valid_neighbors]

    def get_neighbors(self, voxel):
        return voxel.reshape((-1, 3))[:, np.newaxis] + neighbors

    def is_valid(self, voxel):
        return np.logical_and.reduce((0 <= voxel) & (voxel < self.resolution), axis=-1)

    def is_occupied(self, voxel):
        return self.voxels[tuple(voxel.T)] & True

    def is_point(self, voxel):
        return (self.get_voxel(self.points) == voxel[:, np.newaxis]).all(-1).any(1)

    def set_phi(self, voxel, value):
        self.phi[tuple(voxel.T)] = value

    def set_occupied(self, voxel):
        self.voxels[tuple(voxel.T)] = True

    def set_unoccupied(self, voxel):
        self.voxels[tuple(voxel.T)] = False

    def get_bounds(self):
        return self.resolution * self.voxel_size
