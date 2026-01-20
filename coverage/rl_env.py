import random

class MeshRLEnv:
    def __init__(self, graph, faces_in_region):
        self.graph = graph
        self.faces = set(faces_in_region)
        self.covered = set()
        self.state = None

    def reset(self, start_face):
        self.state = start_face
        self.covered = set([start_face])
        return self._get_state()

    def _get_state(self):
        covered_ratio = len(self.covered) / len(self.faces)
        covered_bin = int(covered_ratio * 5)  # 0..4
        return (self.state, covered_bin)

    def actions(self):
        return [
            f for f in self.graph.neighbors(self.state)
            if f in self.faces
        ]

    def step(self, action):
        prev = self.state
        self.state = action

        reward = -0.1  # coût déplacement

        if action not in self.covered:
            reward += 10
            self.covered.add(action)
        else:
            reward -= 5  # recouvrement inutile

        done = len(self.covered) == len(self.faces)

        if done:
            reward += 50  # région entièrement couverte

        return self._get_state(), reward, done
