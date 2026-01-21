import numpy as np
import trimesh
from collections import defaultdict
from scipy.spatial.distance import cdist


class MeshCoverageRLEnv:
    """
    Environnement RL pour couverture de surface.
    
    État: [position_xy, orientation, coverage_map_local]
    Actions: 8 directions + changement d'orientation
    Récompense: couverture + efficacité + lissage
    """
    
    def __init__(self, mesh, coverage_radius=0.1, max_steps=10000):
        """
        Args:
            mesh: trimesh.Mesh
            coverage_radius: rayon de couverture du robot
            max_steps: max étapes par épisode
        """
        self.mesh = mesh
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        
        # État du mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.face_centers = mesh.triangles_center
        self.normals = mesh.face_normals
        self.n_faces = len(self.faces)
        
        # État de couverture
        self.coverage_map = np.zeros(self.n_faces, dtype=bool)  # quelles faces sont couvertes
        self.visited_sequence = []  # historique des positions
        
        # Position actuelle
        self.current_pos = None
        self.current_orientation = np.array([1.0, 0.0, 0.0])  # vecteur direction
        self.step_count = 0
        
        # Stats pour récompense
        self.path_length = 0.0
        self.last_pos = None
        
    def reset(self, start_pos=None):
        """Réinitialise l'environnement"""
        self.coverage_map = np.zeros(self.n_faces, dtype=bool)
        self.visited_sequence = []
        self.step_count = 0
        self.path_length = 0.0
        
        # Position initiale
        if start_pos is None:
            start_pos = self.face_centers[np.random.randint(0, self.n_faces)]
        
        self.current_pos = start_pos.copy()
        self.last_pos = start_pos.copy()
        self.current_orientation = np.random.randn(3)
        self.current_orientation /= np.linalg.norm(self.current_orientation)
        
        # Marquer les faces initiales comme couvertes
        self._update_coverage(start_pos)
        
        return self._get_state()
    
    def _get_state(self):
        """
        Retourne l'état: [position + orientationUnitaire + local_coverage]
        
        Returns:
            state: array de taille variable
        """
        # Position normalisée
        pos_norm = self.current_pos / (np.linalg.norm(self.mesh.vertices.max(axis=0)) + 1e-6)
        
        # Orientation (3 composantes)
        orientation = self.current_orientation
        
        # Coverage local (faces couvertes dans le rayon)
        covered_local = self.coverage_map[self._get_nearby_faces()].astype(float)
        
        # Pourcentage global couvert
        coverage_pct = np.array([self.coverage_map.sum() / self.n_faces])
        
        # Nombre d'étapes
        steps_pct = np.array([self.step_count / self.max_steps])
        
        state = np.concatenate([
            pos_norm,           # 3
            orientation,        # 3
            coverage_pct,       # 1
            steps_pct          # 1
        ])
        
        return state
    
    def _get_nearby_faces(self, radius=None):
        """Retourne les indices des faces proches de la position courante"""
        if radius is None:
            radius = self.coverage_radius
        
        distances = cdist([self.current_pos], self.face_centers)[0]
        return np.where(distances <= radius)[0]
    
    def _update_coverage(self, pos):
        """Marque les faces couvertes à partir de pos"""
        nearby = self._get_nearby_faces()
        self.coverage_map[nearby] = True
    
    def _calculate_reward(self, new_pos):
        """
        Récompense multi-objectif:
        - Nouvelles zones couvertes (+)
        - Recouvrements (-) 
        - Lissage de trajectoire (-)
        - Efficacité (distance/couverture)
        """
        reward = 0.0
        
        # 1. Récompense pour nouvelles zones couvertes
        old_coverage = self.coverage_map.sum()
        self._update_coverage(new_pos)
        new_coverage = self.coverage_map.sum()
        new_area_covered = new_coverage - old_coverage
        reward += new_area_covered * 10.0  # Bonus pour nouvelles faces
        
        # 2. Pénalité pour recouvrements inutiles
        nearby = self._get_nearby_faces()
        already_covered = self.coverage_map[nearby].sum()
        total_nearby = len(nearby)
        if total_nearby > 0:
            overlap_ratio = already_covered / total_nearby
            reward -= overlap_ratio * 5.0  # Pénalité pour redondance
        
        # 3. Pénalité pour changements d'orientation trop brusques
        direction = new_pos - self.current_pos
        if np.linalg.norm(direction) > 1e-6:
            direction /= np.linalg.norm(direction)
            angle_change = np.arccos(np.clip(np.dot(direction, self.current_orientation), -1, 1))
            reward -= 0.1 * angle_change  # Lissage
            self.current_orientation = direction
        
        # 4. Pénalité pour distance parcourue (efficacité)
        step_distance = np.linalg.norm(new_pos - self.current_pos)
        self.path_length += step_distance
        reward -= 0.01 * step_distance  # Favorise chemins courts
        
        # 5. Bonus pour effor de couverture vs distance
        if step_distance > 1e-6:
            coverage_efficiency = new_area_covered / step_distance
            reward += coverage_efficiency * 2.0
        
        # Pénalité pour dépassement du max_steps
        if self.step_count >= self.max_steps:
            reward -= 50.0
        
        return reward
    
    def step(self, action):
        """
        Action: [dx, dy, dz] normalisée (direction unitaire)
        
        Returns:
            state, reward, done, info
        """
        if self.step_count >= self.max_steps:
            done = True
            return self._get_state(), 0.0, done, {"episode_end": "max_steps"}
        
        # Convertir action en déplacement
        if isinstance(action, (int, np.integer)):
            # Si action est un index discret (8 directions)
            action_vector = self._discrete_action_to_vector(action)
        else:
            # Si action continue
            action_vector = np.array(action)
            action_vector /= (np.linalg.norm(action_vector) + 1e-6)
        
        # Nouveau position
        step_size = 0.05  # Hyperparamètre
        new_pos = self.current_pos + action_vector * step_size
        
        # Garder sur la surface du mesh (projection simple)
        new_pos = self._project_on_mesh(new_pos)
        
        # Calculer récompense
        reward = self._calculate_reward(new_pos)
        
        # Update
        self.current_pos = new_pos
        self.visited_sequence.append(new_pos.copy())
        self.step_count += 1
        
        # Condition fin
        coverage_ratio = self.coverage_map.sum() / self.n_faces
        done = (coverage_ratio > 0.95) or (self.step_count >= self.max_steps)
        
        info = {
            "coverage": coverage_ratio,
            "path_length": self.path_length,
            "n_steps": self.step_count
        }
        
        return self._get_state(), reward, done, info
    
    def _discrete_action_to_vector(self, action_idx):
        """Convertit action discrète (0-7) en vecteur 3D"""
        # 8 directions cardinales sur la surface
        directions = [
            [1, 0, 0],    # +X
            [-1, 0, 0],   # -X
            [0, 1, 0],    # +Y
            [0, -1, 0],   # -Y
            [0, 0, 1],    # +Z
            [0, 0, -1],   # -Z
            [1, 1, 0],    # Diagonal
            [1, -1, 0],   # Diagonal
        ]
        direction = np.array(directions[action_idx % len(directions)], dtype=float)
        return direction / np.linalg.norm(direction)
    
    def _project_on_mesh(self, point):
        """Projette un point sur la surface du mesh (simple: nearest point)"""
        distances = cdist([point], self.face_centers)[0]
        nearest_face = np.argmin(distances)
        return self.face_centers[nearest_face].copy()
    
    def get_coverage_percentage(self):
        """Retourne le % de couverture"""
        return (self.coverage_map.sum() / self.n_faces) * 100
    
    def get_path_length(self):
        """Retourne la longueur totale du trajet"""
        return self.path_length
    
    def render(self):
        """Affiche la couverture actuelle (pour debug)"""
        # Colorer les faces couvertes/non couvertes
        colors = np.zeros((self.n_faces, 3))
        colors[self.coverage_map] = [0, 255, 0]  # Vert = couvert
        colors[~self.coverage_map] = [255, 0, 0]  # Rouge = non couvert
        
        return colors
