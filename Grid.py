class Grid:
    def __init__(self):
        voxel_size = 0
        points = []
        resolution = (0, 0, 0)

    def make_grid_from_file(self, file, resolution):
        pass

    def make_grid(self, points, resolution):
        pass

    def get_voxels(self):
        pass

    def get_voxel(self, p):
        pass

    # p = (x,y,z)
    # divide the x y z by the dimension of the voxel cube, the rounded down number indicates the indices
    # of the voxel containing the points in a 3d matrix
    # return the index of the voxel
    def insert_point(self, p):
        # remark: the points should be stored somewhere I put a array to store them, you can change it to a different
        # container
        voxel = (d // self.voxel_size for d in p)
        self.points[voxel] = p
        return voxel

    # (up, down, front, back, left, right)
    # vi = (x, y, z)
    # the input is the voxel index return a list of indexes
    def get_neighbors(self, vi):
        neighbors = []
        neighbors.append(vi + (0, 1, 0))  # up
        neighbors.append(vi - (0, 1, 0))  # down
        neighbors.append(vi + (0, 0, 1))  # front
        neighbors.append(vi - (0, 0, 1))  # back
        neighbors.append(vi - (1, 0, 0))  # left
        neighbors.append(vi + (1, 0, 0))  # right
        return neighbors

    def is_valid(self, v):
        pass

    # we avoid negative indexes therefore starting with 0 up to resolution
    # remark: not sure we need this, the resolution should be enough
    def get_bounds(self):
        return self.resolution * self.voxel_size
