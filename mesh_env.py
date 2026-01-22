import trimesh
import numpy as np
import networkx as nx
import os
from scipy.spatial import distance_matrix
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
    
    def get_face_points(self, face_id, num_points=5):
        """
        Génère plusieurs points sur une face (centre + sommets + points interpolés).
        
        num_points:
        - 1: juste le centre
        - 3: les 3 sommets
        - 5: centre + 4 points interpolés (bords)
        - plus: points plus denses sur la face
        """
        face = self.mesh.faces[face_id]
        vertices = self.mesh.vertices[face]  # 3 sommets du triangle
        
        if num_points == 1:
            return [self.mesh.triangles_center[face_id]]
        
        if num_points == 3:
            return [vertices[0], vertices[1], vertices[2]]
        
        # Générer num_points uniformément distribués sur la face
        points = []
        
        # Ajouter le centre
        points.append(self.mesh.triangles_center[face_id])
        
        # Ajouter les sommets
        for vertex in vertices:
            points.append(vertex)
        
        # Ajouter des points d'interpolation sur les arêtes
        if num_points > 4:
            # Points sur les 3 arêtes
            for i in range(3):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 3]
                for t in np.linspace(0.25, 0.75, (num_points - 4) // 3 + 1):
                    points.append((1 - t) * v1 + t * v2)
        
        return points[:num_points]
    
    def get_all_face_points(self, num_points=5):
        """
        Retourne tous les points de toutes les faces.
        Retourne une liste de listes : pour chaque face, une liste de points.
        """
        all_points = []
        for face_id in range(len(self.mesh.faces)):
            face_points = self.get_face_points(face_id, num_points)
            all_points.append(face_points)
        return all_points

    def generate_uv_map(self, resolution=512):
        """
        Génère une UV map en déroulant le mesh 3D en 2D.
        Retourne les coordonnées UV pour chaque face (entre 0 et 1).
        """
        try:
            # Essayer d'utiliser la fonction unwrap de trimesh
            uv = self.mesh.UV
            
            if uv is not None and len(uv) > 0:
                # Normaliser les coordonnées UV entre 0 et 1
                uv = np.array(uv)
                min_uv = uv.min(axis=0)
                max_uv = uv.max(axis=0)
                range_uv = max_uv - min_uv
                range_uv[range_uv == 0] = 1
                
                uv_normalized = (uv - min_uv) / range_uv
                uv_normalized = np.clip(uv_normalized, 0, 1)
                
                # Obtenir le centre UV de chaque face
                face_uv_centers = self._compute_face_uv_centers(uv_normalized)
                print(f"✅ UV map générée avec trimesh")
                return face_uv_centers, resolution
        except:
            pass
        
        # Fallback: projection orthographique simple basée sur les coordonnées 3D
        print("⚠️  Utilisation d'une projection orthographique pour UV map")
        
        vertices = self.mesh.vertices
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        
        # Projection sur le plan XY
        uv_coords = vertices[:, :2].copy()
        uv_coords = (uv_coords - min_coords[:2]) / (max_coords[:2] - min_coords[:2] + 1e-6)
        uv_coords = np.clip(uv_coords, 0, 1)
        
        # Calculer le centre UV de chaque face
        face_uv_centers = self._compute_face_uv_centers(uv_coords)
        
        return face_uv_centers, resolution
    
    def _compute_face_uv_centers(self, uv_coords):
        """
        Calcule le centre UV de chaque face en moyennant les UV de ses sommets.
        """
        face_uv_centers = np.zeros((len(self.mesh.faces), 2))
        
        for face_idx, face in enumerate(self.mesh.faces):
            # Les 3 sommets de la face
            vert_uv = uv_coords[face]
            # Moyenne des UV des sommets
            face_uv_centers[face_idx] = vert_uv.mean(axis=0)
        
        return face_uv_centers
    
    def get_face_uv_bounds(self):
        """
        Retourne les limites UV de chaque face.
        """
        try:
            face_uv_centers, _ = self.generate_uv_map()
            
            bounds = []
            for face_idx in range(len(self.mesh.faces)):
                face = self.mesh.faces[face_idx]
                vertices = self.mesh.vertices[face]
                
                # Projection 2D
                min_coords = self.mesh.vertices.min(axis=0)
                max_coords = self.mesh.vertices.max(axis=0)
                
                uv_2d = vertices[:, :2]
                uv_2d = (uv_2d - min_coords[:2]) / (max_coords[:2] - min_coords[:2] + 1e-6)
                
                min_uv = uv_2d.min(axis=0)
                max_uv = uv_2d.max(axis=0)
                bounds.append((min_uv, max_uv))
            
            return bounds
        except:
            return None
    
    def pixel_to_face(self, uv_pos, face_uv_centers):
        """
        Trouve la face la plus proche pour une position UV donnée.
        """
        distances = np.linalg.norm(face_uv_centers - uv_pos, axis=1)
        return np.argmin(distances)
    
    def face_to_uv(self, face_id, face_uv_centers):
        """
        Retourne les coordonnées UV d'une face.
        """
        return face_uv_centers[face_id]

    def smooth_path_3d(self, uv_path, face_uv_centers, interpolation_steps=5):
        """
        Convertit un chemin UV en chemin 3D lissé.
        Préserve tous les points en les projetant directement sur le mesh.
        """
        path_3d = []
        
        # Pour chaque point UV, trouver la face la plus proche et interpoler
        for i, uv_pos in enumerate(uv_path):
            face_id = self.pixel_to_face(uv_pos, face_uv_centers)
            face_center = self.face_center(face_id)
            path_3d.append(face_center)
            
            # Interpoler vers le point suivant
            if i < len(uv_path) - 1:
                next_uv = uv_path[i + 1]
                next_face_id = self.pixel_to_face(next_uv, face_uv_centers)
                next_center = self.face_center(next_face_id)
                
                # Interpolation linéaire entre les deux faces
                if face_id != next_face_id:
                    for step in range(1, interpolation_steps):
                        t = step / interpolation_steps
                        interpolated = (1 - t) * face_center + t * next_center
                        path_3d.append(interpolated)
        
        return np.array(path_3d)
    
    def project_point_to_mesh_surface(self, point):
        """
        Projette un point sur la surface la plus proche du mesh.
        """
        closest_point, distance, face_id = self.mesh.nearest.on_surface([point])
        return closest_point[0], face_id
    
    def refine_path_on_surface(self, path_3d):
        """
        Affine le chemin 3D en s'assurant que tous les points sont sur la surface du mesh.
        """
        refined_path = []
        for point in path_3d:
            projected_point, _ = self.project_point_to_mesh_surface(point)
            refined_path.append(projected_point)
        return np.array(refined_path)
