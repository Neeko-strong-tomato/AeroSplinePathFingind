import random

class MeshRLEnv:
    def __init__(self, graph, goal):
        self.graph = graph
        self.goal = goal
        self.state = None

    def reset(self, start):
        self.state = start
        return self.state

    def step(self, action):
        self.state = action

        reward = -1
        done = False

        if self.state == self.goal:
            reward = 100
            done = True

        return self.state, reward, done

    def actions(self, state):
        return list(self.graph.neighbors(state))
