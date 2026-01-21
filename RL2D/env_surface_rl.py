import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SurfaceCoverageEnv(gym.Env):
    """
    Environment 2D pour coverage / peinture avec formes fermées.
    Mask : "#" = zone peignable, "." = zone interdite
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, surface_map, max_steps=500):
        super().__init__()

        self.surface_map = surface_map
        self.height = len(surface_map)
        self.width = len(surface_map[0])
        self.max_steps = max_steps

        # Action : 0 = forward, 1 = turn left, 2 = turn right
        self.action_space = spaces.Discrete(3)

        # Observation : mask + grid + position + direction
        self.observation_space = spaces.Dict({
            "mask": spaces.Box(0, 1, shape=(self.height, self.width), dtype=np.int8),
            "grid": spaces.Box(0, 1, shape=(self.height, self.width), dtype=np.int8),
            "pos": spaces.Box(low=0, high=max(self.height, self.width),
                               shape=(2,), dtype=np.int32),
            "dir": spaces.Discrete(4)
        })

        self.reset()

    def _map_to_mask(self, surface_map):
        return np.array([[1 if c == "#" else 0 for c in row] for row in surface_map], dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.mask = self._map_to_mask(self.surface_map)
        self.grid = np.zeros_like(self.mask)  # cellules peintes

        # Position de départ : aléatoire dans la zone peignable
        ys, xs = np.where(self.mask == 1)
        idx = self.np_random.integers(len(xs))
        self.pos = np.array([ys[idx], xs[idx]], dtype=np.int32)

        self.dir = self.np_random.integers(4)  # 0: up, 1: right, 2: down, 3: left

        self.steps = 0
        self.covered = 0

        # pour encourager les segments droits
        self.straight_length = 0
        self.last_dir = self.dir

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "mask": self.mask.copy(),
            "grid": self.grid.copy(),
            "pos": self.pos.copy(),
            "dir": self.dir
        }

    def step(self, action):
        self.steps += 1
        reward = 0

        # Rotation
        if action == 1:  # left
            self.dir = (self.dir - 1) % 4
            reward -= 0.2
        elif action == 2:  # right
            self.dir = (self.dir + 1) % 4
            reward -= 0.2

        # Avance
        move = [(-1, 0), (0, 1), (1, 0), (0, -1)][self.dir]
        new_pos = self.pos + np.array(move)

        # Hors carte
        if not (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width):
            reward -= 5
            return self._get_obs(), reward, True, False, {}

        # Zone interdite
        if self.mask[new_pos[0], new_pos[1]] == 0:
            reward -= 2
            return self._get_obs(), reward, False, False, {}

        # Mouvement validé
        self.pos = new_pos

        # Bonus longueur de segment droit
        if self.dir == self.last_dir:
            self.straight_length += 1
            reward += min(0.05 * self.straight_length, 1.0)
        else:
            self.straight_length = 1
            reward -= 0.1

        self.last_dir = self.dir

        # Peinture
        if self.grid[self.pos[0], self.pos[1]] == 0:
            self.grid[self.pos[0], self.pos[1]] = 1
            reward += 1.0
            self.covered += 1
        else:
            reward -= 0.5

        # Fin si tout est peint
        if self.covered == self.mask.sum():
            reward += 20
            return self._get_obs(), reward, True, False, {}

        done = self.steps >= self.max_steps
        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        # simple print text
        display = np.full((self.height, self.width), " ")
        for y in range(self.height):
            for x in range(self.width):
                if self.mask[y, x] == 0:
                    display[y, x] = "."
                elif self.grid[y, x] == 1:
                    display[y, x] = "#"
                else:
                    display[y, x] = " "

        py, px = self.pos
        display[py, px] = "R"

        print("\n".join("".join(row) for row in display))
        print(f"step={self.steps} covered={self.covered}/{self.mask.sum()} dir={self.dir}")
