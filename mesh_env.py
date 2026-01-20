import trimesh
import numpy as np
import networkx as nx
import os

class MeshEnvironment:
    def __init__(self, mesh_path=None):
        self.mesh_path = mesh_path
        self.mesh = self._load_mesh_safe()
        self.graph = self._build_graph()

    def _load_mesh_safe(self):
        """
        Charge un mesh depuis un fichier.
        Si échec → génère une sphère de secours.
        """
        if self.mesh_path is None or not os.path.exists(self.mesh_path):
            print("⚠️  Mesh introuvable. Génération d'une sphère par défaut.")
            return self._fallback_sphere()

        try:
            mesh = trimesh.load(self.mesh_path)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("Objet chargé invalide")
            print(f"✅ Mesh chargé : {self.mesh_path}")
            return mesh

        except Exception as e:
            print(f"❌ Erreur chargement mesh : {e}")
            print("⚠️  Utilisation d'une sphère de secours.")
            return self._fallback_sphere()

    def _fallback_sphere(self):
        return trimesh.creation.icosphere(
            radius=1.0,
            subdivisions=3
        )

    def _build_graph(self):
        G = nx.Graph()
        centers = self.mesh.triangles_center
        adjacency = self.mesh.face_adjacency

        for i, c in enumerate(centers):
            G.add_node(i, pos=c)

        for f1, f2 in adjacency:
            p1, p2 = centers[f1], centers[f2]
            G.add_edge(
                f1, f2,
                weight=np.linalg.norm(p1 - p2)
            )

        return G

    def face_center(self, face_id):
        return self.mesh.triangles_center[face_id]
