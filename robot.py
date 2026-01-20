import trimesh
import numpy as np

class Robot:
    def __init__(self, size=0.05):
        self.body = trimesh.creation.box(extents=(size, size, size))
        self.path_points = []

    def move_to(self, position):
        self.body.apply_translation(position - self.body.centroid)
        self.path_points.append(position)

    def paint_trail(self):
        if len(self.path_points) < 2:
            return None

        return trimesh.load_path(self.path_points)
