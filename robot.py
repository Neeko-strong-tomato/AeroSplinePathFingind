import trimesh
import numpy as np

class Robot:
    def __init__(self, size=0.05):
        self.size = size
        self.body = trimesh.creation.box(extents=(size, size, size))
        self.initial_pos = self.body.centroid.copy()
        self.path_points = []

    def move_to(self, position):
        """Déplace le robot à la position donnée"""
        try:
            position = np.array(position, dtype=np.float32)
            
            if position.shape != (3,):
                raise ValueError(f"Position invalide: {position.shape}, attendu (3,)")
            
            # Calculer le déplacement
            current_pos = self.body.centroid.copy()
            displacement = position - current_pos
            
            # Appliquer le déplacement
            self.body.apply_translation(displacement)
            self.path_points.append(position.copy())
        except Exception as e:
            print(f"❌ Erreur move_to: {e}")

    def paint_trail(self):
        """Crée une trace du chemin parcouru"""
        try:
            if len(self.path_points) < 2:
                return None
            
            trail_points = np.array(self.path_points, dtype=np.float32)
            trail = trimesh.load_path(trail_points)
            return trail
        except Exception as e:
            print(f"❌ Erreur paint_trail: {e}")
            return None

    def reset(self):
        """Réinitialise la position du robot"""
        self.body = trimesh.creation.box(extents=(self.size, self.size, self.size))
        self.path_points = []
