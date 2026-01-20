import trimesh
import os
import networkx as nx


class MeshEnvironment:
    def __init__(self, mesh_path=None):
        self.mesh = self._load_mesh_safe(mesh_path)
        self.graph = self._build_face_graph()

    def _load_mesh_safe(self, mesh_path):
        if mesh_path is None or not os.path.exists(mesh_path):
            print(" Mesh introuvable → sphère par défaut")
            return trimesh.creation.icosphere(radius=1.0, subdivisions=4)

        try:
            mesh = trimesh.load(mesh_path)
            print(f" Mesh chargé : {mesh_path}")
            return mesh
        except Exception as e:
            print(f" Erreur mesh ({e}) → sphère par défaut")
            return trimesh.creation.icosphere(radius=1.0, subdivisions=4)

    def _build_face_graph(self):
        """
        Construit un graphe où chaque noeud est une face du mesh
        et chaque arête représente une adjacency géométrique.
        """
        graph = nx.Graph()

        # nombre de faces
        n_faces = len(self.mesh.faces)
        graph.add_nodes_from(range(n_faces))

        # adjacency fournie par trimesh
        for f1, f2 in self.mesh.face_adjacency:
            graph.add_edge(f1, f2)

        print(f" Graphe construit : {n_faces} faces, {graph.number_of_edges()} arêtes")
        return graph
