import trimesh
import os

class MeshEnvironment:
    def __init__(self, mesh_path=None):
        self.mesh = self._load_mesh_safe(mesh_path)

    def _load_mesh_safe(self, mesh_path):
        if mesh_path is None or not os.path.exists(mesh_path):
            print("⚠️ Mesh introuvable → sphère par défaut")
            return trimesh.creation.icosphere(radius=1.0, subdivisions=4)

        try:
            mesh = trimesh.load(mesh_path)
            print(f"✅ Mesh chargé : {mesh_path}")
            return mesh
        except Exception as e:
            print(f"❌ Erreur mesh ({e}) → sphère par défaut")
            return trimesh.creation.icosphere(radius=1.0, subdivisions=4)
