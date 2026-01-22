import numpy as np
from robot.actions import (
    ACTION_MOVE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
)

class Robot:
    def __init__(self, mesh):
        self.mesh = mesh
        self.face_centers = mesh.triangles_center
        self.face_normals = mesh.face_normals
        self.face_adjacency = self._build_adjacency()

        self.reset()

    def _build_adjacency(self):
        adj = [[] for _ in range(len(self.mesh.faces))]
        for i, j in self.mesh.face_adjacency:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def reset(self, face_id=None):
        if face_id is None:
            face_id = np.random.randint(len(self.mesh.faces))

        self.current_face = face_id
        self.position = self.face_centers[face_id]
        self.normal = self.face_normals[face_id]

        self.path = [face_id]
        self.prev_normal = self.normal

    def step(self, action):
        neighbors = self.face_adjacency[self.current_face]
        if not neighbors:
            return self.current_face

        if action == ACTION_MOVE:
            next_face = neighbors[0]

        elif action == ACTION_LEFT:
            next_face = neighbors[len(neighbors) // 2]

        elif action == ACTION_RIGHT:
            next_face = neighbors[-1]

        elif action == ACTION_STAY:
            next_face = self.current_face

        else:
            next_face = self.current_face

        self.current_face = next_face
        self.position = self.face_centers[next_face]
        self.normal = self.face_normals[next_face]
        self.path.append(next_face)

        return next_face

    def angle_change(self):
        dot = np.dot(self.normal, self.prev_normal)
        dot = np.clip(dot, -1.0, 1.0)
        angle = 1.0 - dot
        self.prev_normal = self.normal
        return angle
