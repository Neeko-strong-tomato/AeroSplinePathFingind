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
        """Anime le robot le long du chemin"""
        try:
            print(f"🎬 Démarrage animation avec {len(path)} points")
            
            for i, p in enumerate(path):
                # Convertir en numpy array
                position = np.array(p, dtype=np.float32)
                
                if position.shape != (3,):
                    print(f"⚠️  Point {i} invalide: {position.shape}, attendu (3,)")
                    continue
                
                # Bouger le robot
                robot.move_to(position)
                
                # Créer la trace (tous les 5 points pour éviter trop de géométries)
                if i % 5 == 0:
                    trail = robot.paint_trail()
                    if trail is not None:
                        self.scene.add_geometry(trail)
                
                # Afficher chaque 20 points pour aller plus vite
                if i % 20 == 0:
                    try:
                        self.scene.show()
                    except:
                        pass
                
                time.sleep(delay)
            
            print(f"✅ Animation terminée")
            self.scene.show()
        except Exception as e:
            print(f"❌ Erreur animation: {e}")
            import traceback
            traceback.print_exc()
