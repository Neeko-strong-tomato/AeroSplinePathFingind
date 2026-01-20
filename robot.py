import trimesh
import numpy as np

class Robot:
    def __init__(self, size=0.05):
        self.body = trimesh.creation.box(extents=(size, size, size))
        self.trail_points = []

    def move_to(self, position):
        delta = position - self.body.centroid
        self.body.apply_translation(delta)
        self.trail_points.append(position)

    def paint_trail(self):
        if len(self.trail_points) < 2:
            return None

        return trimesh.load_path(self.trail_points)
