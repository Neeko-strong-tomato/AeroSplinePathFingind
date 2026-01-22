import random
import numpy as np
from collections import defaultdict
from uv_map import create_point_cloud
class UVMapRLEnv:
    """Environnement RL navigant dans l'espace 2D des UV maps"""
    def __init__(self, mesh_env, coverage_threshold=0.95, max_steps_per_episode=500, cell_size=0.02):
        self.mesh_env = mesh_env
        self.coverage_threshold = coverage_threshold
        self.max_steps = max_steps_per_episode
        self.cell_size = cell_size
        
        # Obtenir la UV map 2D
        self.face_uv_centers, _ = mesh_env.generate_uv_map()
        
        # État = position (x, y) en 2D dans l'espace UV
        self.state = None
        self.visited_cells = set()  # Ensemble de cellules couvertes
        self.path_history = []
        self.previous_direction = None
        self.steps = 0

    def _uv_to_grid_cell(self, uv_pos):
        """Convertit une position UV en cellule de grille discrète"""
        cell_x = int(np.clip(uv_pos[0] / self.cell_size, 0, 1000))
        cell_y = int(np.clip(uv_pos[1] / self.cell_size, 0, 1000))
        return (cell_x, cell_y)

    def reset(self, start_uv=None):
        """Réinitialise l'environnement avec position UV de départ"""
        if start_uv is None:
            start_uv = np.array([0.5, 0.5])
        
        self.state = np.array(start_uv, dtype=np.float32)
        self.visited_cells = {self._uv_to_grid_cell(self.state)}
        self.path_history = [self.state.copy()]
        self.previous_direction = None
        self.steps = 0
        return self.state

    def actions(self, state=None):
        """Retourne 8 actions possibles : déplacement 2D dans l'espace UV"""
        state = state if state is not None else self.state
        actions = []
        step = self.cell_size * 1.5
        
        # 8 directions de déplacement
        directions = [
            (step, 0),       # droite
            (-step, 0),      # gauche
            (0, step),       # haut
            (0, -step),      # bas
            (step, step),    # diag haut-droite
            (step, -step),   # diag bas-droite
            (-step, step),   # diag haut-gauche
            (-step, -step)   # diag bas-gauche
        ]
        
        for dx, dy in directions:
            new_pos = state + np.array([dx, dy], dtype=np.float32)
            # Rester dans [0, 1]²
            new_pos = np.clip(new_pos, 0.0, 1.0)
            actions.append(new_pos)
        
        return actions

    def _calculate_direction_2d(self, from_uv, to_uv):
        """Calcule la direction normalisée entre deux positions UV"""
        direction = to_uv - from_uv
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            return direction / norm
        return None

    def _calculate_orientation_penalty(self, from_uv, to_uv):
        """Pénalise les changements d'orientation brusques"""
        current_direction = self._calculate_direction_2d(from_uv, to_uv)
        if current_direction is None or self.previous_direction is None:
            return 0
        
        dot_product = np.clip(np.dot(current_direction, self.previous_direction), -1, 1)
        angle_change = np.arccos(dot_product)
        return angle_change / np.pi

    def _check_path_crossing(self, window=10):
        """Pénalise la revisite récente"""
        if len(self.path_history) < 3:
            return 0
        
        current_cell = self._uv_to_grid_cell(self.state)
        recent_cells = set([self._uv_to_grid_cell(p) for p in self.path_history[-window:]])
        
        if current_cell in recent_cells and self.path_history[-1] is not self.state:
            return 1.0
        return 0

    def _get_coverage_ratio(self):
        """Retourne le ratio de couverture"""
        max_cells = int((1.0 / self.cell_size) ** 2)
        return len(self.visited_cells) / max_cells

    def calculate_reward(self, action):
        """Calcule le reward pour maximiser la couverture"""
        reward = 0

        # Pénalité par pas
        reward -= 0.05

        # Récompense pour nouvelles cellules explorées
        action_cell = self._uv_to_grid_cell(action)
        if action_cell not in self.visited_cells:
            reward += 1.0

        # Pénalité pour recouvrements récents
        reward -= self._check_path_crossing() * 0.01

        # Pénalité pour changement d'orientation brusque
        reward -= self._calculate_orientation_penalty(self.state, action) * 0.5

        # Reward progressif pour couverture
        coverage_ratio = self._get_coverage_ratio()
        reward += coverage_ratio * 2.0

        # Bonus énorme pour couverture complète
        if coverage_ratio >= self.coverage_threshold:
            reward += 100

        return reward

    def step(self, action):
        """Effectue un pas dans l'environnement"""
        prev_state = self.state.copy()
        self.state = action.copy()
        
        reward = self.calculate_reward(action)
        
        action_cell = self._uv_to_grid_cell(action)
        self.visited_cells.add(action_cell)
        self.path_history.append(self.state.copy())
        
        self.previous_direction = self._calculate_direction_2d(prev_state, action)
        self.steps += 1

        coverage_ratio = self._get_coverage_ratio()
        done = (coverage_ratio >= self.coverage_threshold) or (self.steps >= self.max_steps)

        return self.state, reward, done
