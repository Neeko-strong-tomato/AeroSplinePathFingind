"""
Mesh segmentation module for dividing STL meshes into paintable faces.

A face is defined as a connected region of mesh faces where:
- Adjacent faces have an angle < 60 degrees between normals
- The overall bend angle of the grouped region is < 60 degrees
"""

import numpy as np
import trimesh
from scipy.spatial.distance import cdist
from collections import deque
from typing import List, Tuple, Set
from tqdm import tqdm


class MeshFaceSegmenter:
    """Segments a mesh into faces based on curvature and angle thresholds."""
    
    def __init__(
        self,
        angle_threshold: float = 60.0,
        bend_angle_threshold: float = 60.0
    ):
        """
        Initialize segmenter.
        
        Args:
            angle_threshold: Maximum angle (degrees) between adjacent face normals
            bend_angle_threshold: Maximum overall bend angle (degrees) for a face region
        """
        self.angle_threshold = np.radians(angle_threshold)
        self.bend_angle_threshold = np.radians(bend_angle_threshold)
        self.mesh = None
        self.face_normals = None
        self.adjacency = None
        self.face_segments = None
        
    def load_stl(self, stl_path: str) -> trimesh.Trimesh:
        """Load STL file and compute face properties."""
        print(f"Loading STL file: {stl_path}")
        self.mesh = trimesh.load(stl_path)
        print(f"Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        print("Computing face normals...")
        self.face_normals = self.mesh.face_normals
        print("Building face adjacency graph...")
        self._compute_adjacency()
        print("Adjacency graph complete")
        return self.mesh
    
    def _compute_adjacency(self) -> None:
        """Compute face adjacency graph."""
        num_faces = len(self.mesh.faces)
        self.adjacency = [[] for _ in range(num_faces)]
        
        # Build edge to faces mapping
        edge_to_faces = {}
        for face_idx, face in enumerate(tqdm(self.mesh.faces, desc="Building edge map", unit="face")):
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Build adjacency from shared edges
        for edge, faces in tqdm(edge_to_faces.items(), desc="Building adjacency", unit="edge"):
            if len(faces) == 2:
                f1, f2 = faces
                self.adjacency[f1].append(f2)
                self.adjacency[f2].append(f1)
    
    def _angle_between_normals(self, n1: np.ndarray, n2: np.ndarray) -> float:
        """Compute angle between two normal vectors."""
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def _compute_face_region_bend_angle(self, face_indices: Set[int]) -> float:
        """
        Compute overall bend angle for a region of faces.
        
        Uses the angle between the mean normal of the region and individual face normals.
        """
        if len(face_indices) <= 1:
            return 0.0
        
        region_normals = self.face_normals[list(face_indices)]
        mean_normal = region_normals.mean(axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-10)
        
        max_angle = 0.0
        for normal in region_normals:
            angle = self._angle_between_normals(mean_normal, normal)
            max_angle = max(max_angle, angle)
        
        return max_angle
    
    def _is_valid_merge(
        self,
        face_idx: int,
        existing_region: Set[int]
    ) -> bool:
        """Check if a face can be merged with an existing region."""
        # Check angle with all neighbors in the region
        for neighbor_idx in existing_region:
            if neighbor_idx in self.adjacency[face_idx]:
                angle = self._angle_between_normals(
                    self.face_normals[face_idx],
                    self.face_normals[neighbor_idx]
                )
                if angle > self.angle_threshold:
                    return False
        
        # Check overall bend angle after merge
        test_region = existing_region | {face_idx}
        bend_angle = self._compute_face_region_bend_angle(test_region)
        if bend_angle > self.bend_angle_threshold:
            return False
        
        return True
    
    def segment(self) -> List[Set[int]]:
        """
        Segment mesh into faces using region growing.
        
        Returns:
            List of sets, each containing face indices that form a face region
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded. Call load_stl() first.")
        
        num_faces = len(self.mesh.faces)
        visited = [False] * num_faces
        self.face_segments = []
        
        pbar = tqdm(total=num_faces, desc="Segmenting mesh", unit="face")
        
        for start_idx in range(num_faces):
            if visited[start_idx]:
                continue
            
            # Start region growing from unvisited face
            region = {start_idx}
            queue = deque([start_idx])
            visited[start_idx] = True
            pbar.update(1)
            
            while queue:
                current_idx = queue.popleft()
                
                # Try to add adjacent faces to region
                for neighbor_idx in self.adjacency[current_idx]:
                    if not visited[neighbor_idx]:
                        if self._is_valid_merge(neighbor_idx, region):
                            region.add(neighbor_idx)
                            visited[neighbor_idx] = True
                            pbar.update(1)
                            queue.append(neighbor_idx)
            
            self.face_segments.append(region)
        
        pbar.close()
        return self.face_segments
    
    def get_face_geometry(self, face_indices: Set[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract geometry for a face region.
        
        Returns:
            (vertices, faces): Vertices of the region and face indices
        """
        face_list = list(face_indices)
        mesh_faces = self.mesh.faces[face_list]
        
        # Get unique vertices in this region
        unique_vertices_idx = np.unique(mesh_faces)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices_idx)}
        
        # Remap faces to new vertex indices
        remapped_faces = np.array([
            [vertex_map[v] for v in face] for face in mesh_faces
        ])
        
        vertices = self.mesh.vertices[unique_vertices_idx]
        
        return vertices, remapped_faces
    
    def get_face_bounds(self, face_indices: Set[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of a face region."""
        vertices, _ = self.get_face_geometry(face_indices)
        return vertices.min(axis=0), vertices.max(axis=0)
