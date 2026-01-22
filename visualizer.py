import trimesh
import time
import numpy as np

class Visualizer:
    def __init__(self, mesh):
        self.scene = trimesh.Scene(mesh)
        self.mesh = mesh

    def add_path(self, points):
        """Ajoute un chemin à la scène"""
        try:
            points_array = np.array(points)
            if len(points_array) > 1:
                path = trimesh.load_path(points_array)
                self.scene.add_geometry(path)
                print(f"✅ Chemin ajouté avec {len(points_array)} points")
        except Exception as e:
            print(f"❌ Erreur ajout chemin: {e}")

    def add_robot(self, robot):
        """Ajoute le robot à la scène"""
        try:
            self.scene.add_geometry(robot.body)
            print(f"✅ Robot ajouté")
        except Exception as e:
            print(f"❌ Erreur ajout robot: {e}")

    def animate(self, robot, path, delay=0.05):
        """Affiche simplement le chemin sans animation"""
        try:
            print(f"🎨 Affichage du chemin avec {len(path)} points")
            
            # Peindre toute la trace
            for i, p in enumerate(path):
                position = np.array(p, dtype=np.float32)
                
                if position.shape != (3,):
                    print(f"⚠️  Point {i} invalide: {position.shape}, attendu (3,)")
                    continue
                
                robot.move_to(position)
                
                # Créer la trace
                if i % 5 == 0:
                    trail = robot.paint_trail()
                    if trail is not None:
                        self.scene.add_geometry(trail)
            
            # Placer le robot au point final
            if len(path) > 0:
                robot.move_to(np.array(path[-1], dtype=np.float32))
            
            print(f"✅ Chemin affiché")
            self.scene.show(block=True)
        except Exception as e:
            print(f"❌ Erreur affichage: {e}")
            import traceback
            traceback.print_exc()