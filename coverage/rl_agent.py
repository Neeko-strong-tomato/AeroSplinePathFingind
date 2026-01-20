import random
from collections import defaultdict

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)

        qs = [(self.q[(state, a)], a) for a in actions]
        return max(qs)[1]

    def update(self, s, a, r, s_next, actions_next):
        max_q = max([self.q[(s_next, a2)] for a2 in actions_next], default=0)
        self.q[(s, a)] += self.alpha * (r + self.gamma * max_q - self.q[(s, a)])
