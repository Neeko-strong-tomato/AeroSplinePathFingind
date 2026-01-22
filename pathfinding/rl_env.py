import random
import numpy as np
from collections import defaultdict
from uv_map import create_point_cloud
class UVMapRLEnv:
    """Environnement RL navigant dans l'espace 2D des UV maps"""
    def __init__(self, mesh_env, coverage_threshold=0.95, max_steps_per_episode=2000, cell_size=0.02):
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

    def _get_surface_normal_at_uv(self, uv_pos):
        """Obtient le vecteur normal de la surface au point UV"""
        # Trouver la face la plus proche de cette position UV
        distances = np.linalg.norm(self.face_uv_centers - uv_pos, axis=1)
        closest_face_idx = np.argmin(distances)
        
        # Retourner la normale de cette face
        face_normal = self.mesh_env.mesh.face_normals[closest_face_idx]
        return face_normal / np.linalg.norm(face_normal)

    def _calculate_orientation_penalty(self, from_uv, to_uv):
        """Calcule l'angle entre les vecteurs normales de départ et d'arrivée
        Pénalité exponentielle : plus forte pour les directions opposées"""
        # Obtenir les normales de surface à chaque position
        normal_from = self._get_surface_normal_at_uv(from_uv)
        normal_to = self._get_surface_normal_at_uv(to_uv)
        
        # Calculer l'angle entre les deux normales
        dot_product = np.clip(np.dot(normal_from, normal_to), -1, 1)
        angle = np.arccos(dot_product)
        
        # Pénalité quadratique pour punir fortement les directions opposées
        # angle/π va de 0 à 1, (angle/π)² va de 0 à 1 avec plus de poids sur les valeurs hautes
        normalized_angle = angle / np.pi
        return normalized_angle ** 2

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

    def _is_on_mesh(self, uv_pos):
        """Vérifie si la position UV correspond à une face réelle du mesh"""
        # On utilise une distance seuil très faible pour confirmer la présence d'une face
        distances = np.linalg.norm(self.face_uv_centers - uv_pos, axis=1)
        min_dist = np.min(distances)
        
        # Si la distance au centre de face le plus proche est trop grande, 
        # on est probablement dans le "vide" de l'UV map
        return min_dist < self.cell_size * 2.0

    def _check_mesh_integrity(self, from_uv, to_uv):
        """
        Pénalise si le mouvement traverse le vide ou saute entre des 
        parties du mesh non connectées.
        """
        # 1. Vérifier si l'arrivée est sur le mesh
        if not self._is_on_mesh(to_uv):
            return 2.0 # Forte pénalité pour sortie du mesh

        # 2. Vérifier l'adjacence physique (optionnel mais recommandé)
        # On trouve les faces de départ et d'arrivée
        dists_from = np.linalg.norm(self.face_uv_centers - from_uv, axis=1)
        dists_to = np.linalg.norm(self.face_uv_centers - to_uv, axis=1)
        
        face_from = np.argmin(dists_from)
        face_to = np.argmin(dists_to)

        # Si les faces ne sont pas les mêmes et ne sont pas adjacentes dans le graphe
        if face_from != face_to:
            if not self.mesh_env.graph.has_edge(face_from, face_to):
                return 1.5 # Pénalité pour "saut" téléporté entre deux îles UV
                
        return 0.0


    def calculate_reward(self, action):
        """Calcule le reward pour maximiser la couverture"""
        reward = 0

        # Pénalité par pas
        reward -= 0.05

        # --- NOUVELLE PÉNALITÉ DE TRAVERSÉE ---
        mesh_penalty = self._check_mesh_integrity(self.state, action)
        reward -= mesh_penalty * 2.0  # Multiplicateur d'importance
        
        if mesh_penalty > 0:
            # Si le mouvement est invalide, on peut décider de stopper l'exploration
            # ou simplement de donner une très mauvaise note.
            return reward - 5.0 

        # Récompense pour nouvelles cellules explorées
        action_cell = self._uv_to_grid_cell(action)
        if action_cell not in self.visited_cells:
            reward += 1.0
        else:
            reward -= 0.1

        # Pénalité pour recouvrements récents
        reward -= self._check_path_crossing() * 0.1

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
