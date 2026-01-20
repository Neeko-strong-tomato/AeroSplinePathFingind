"""
Path planning module for generating optimal painting paths on mesh faces.

Uses a snake/raster pattern for efficient coverage of face regions.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree


class PathPlanner:
    """Generates painting coverage paths for mesh faces."""
    
    def __init__(
        self,
        spray_height: float = 0.20,
        spray_radius: float = 0.10,
        path_spacing: float = 0.05
    ):
        """
        Initialize path planner.
        
        Args:
            spray_height: Height of arm head above surface (meters)
            spray_radius: Radius of spray pattern (meters)
            path_spacing: Spacing between adjacent path lines (meters)
        """
        self.spray_height = spray_height
        self.spray_radius = spray_radius
        self.path_spacing = path_spacing
    
    def plan_path(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_normal: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate adaptive coverage path that wraps around the actual 3D geometry.
        
        Creates a sample grid on the face surface and generates optimized 
        waypoints that follow the actual mesh topology, not flat projections.
        
        Args:
            vertices: Face vertices (N x 3)
            faces: Face indices (M x 3)
            face_normal: Normal vector of the face
            
        Returns:
            List of waypoints defining the spray path
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Sample points across the face surface itself
        waypoints = self._generate_surface_coverage(mesh, face_normal)
        
        return waypoints
    
    def _get_u_vector(self, normal: np.ndarray) -> np.ndarray:
        """Get first orthogonal vector in the plane defined by normal."""
        # Find vector not parallel to normal
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        
        u_vec = np.cross(normal, arbitrary)
        u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-10)
        return u_vec
    
    def _project_to_2d(
        self,
        vertices: np.ndarray,
        center: np.ndarray,
        u_vec: np.ndarray,
        v_vec: np.ndarray
    ) -> np.ndarray:
        """Project 3D vertices onto 2D plane."""
        relative = vertices - center
        x_coords = np.dot(relative, u_vec)
        y_coords = np.dot(relative, v_vec)
        return np.column_stack([x_coords, y_coords])
    
    def _generate_surface_coverage(
        self,
        mesh: trimesh.Trimesh,
        face_normal: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate coverage waypoints by sampling directly on the mesh surface.
        
        Creates sample points uniformly across triangles and connects them
        in an optimized order that wraps around the geometry.
        """
        # Sample points directly from mesh triangles using barycentric coords
        sample_points, triangle_indices = self._sample_mesh_triangles(mesh)
        
        if len(sample_points) < 2:
            # Fallback if sampling fails
            center = mesh.vertices.mean(axis=0)
            return [center + self.spray_height * mesh.vertex_normals.mean(axis=0)]
        
        # Sort points into efficient coverage order
        sorted_points, sorted_triangles = self._sort_points_by_coverage(
            sample_points, triangle_indices
        )
        
        # Generate waypoints with proper height offset using triangle normals
        waypoints = []
        for point, tri_idx in zip(sorted_points, sorted_triangles):
            # Use the actual triangle normal for this point
            tri_normal = mesh.face_normals[tri_idx]
            # Offset point by spray height along the surface normal
            waypoint = point + self.spray_height * tri_normal
            waypoints.append(waypoint)
        
        return waypoints
    
    def _sample_mesh_triangles(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points uniformly across mesh triangles using barycentric coordinates.
        
        Returns:
            (sample_points, triangle_indices): Arrays of sampled 3D points and their source triangle indices
        """
        # Calculate number of samples based on mesh area
        num_samples = max(10, int(mesh.area / (self.path_spacing ** 2)))
        
        # Calculate face areas directly
        face_areas = []
        for i, face in enumerate(mesh.faces):
            v0, v1, v2 = mesh.vertices[face]
            # Cross product gives 2x the area
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            face_areas.append(area)
        
        face_areas = np.array(face_areas)
        
        # Normalize areas for probability distribution
        total_area = face_areas.sum()
        if total_area > 1e-10:
            face_probs = face_areas / total_area
        else:
            face_probs = np.ones(len(mesh.faces)) / len(mesh.faces)
        
        # Sample triangles based on area
        sampled_face_indices = np.random.choice(
            len(mesh.faces),
            size=num_samples,
            p=face_probs
        )
        
        sample_points = []
        triangle_indices = []
        
        # For each sampled triangle, get a random point on it
        for face_idx in sampled_face_indices:
            face = mesh.faces[face_idx]
            v0, v1, v2 = mesh.vertices[face]
            
            # Random barycentric coordinates
            r1 = np.random.random()
            r2 = np.random.random()
            
            # Ensure they're in the triangle
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            
            # Compute point using barycentric coords
            point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
            sample_points.append(point)
            triangle_indices.append(face_idx)
        
        return np.array(sample_points), np.array(triangle_indices)
    
    def _sort_points_by_coverage(
        self,
        points: np.ndarray,
        triangle_indices: np.ndarray
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Sort sample points into an efficient coverage order.
        
        Uses greedy nearest neighbor to minimize travel distance.
        """
        if len(points) == 0:
            return [], []
        
        # Use greedy nearest neighbor to order points efficiently
        remaining = list(range(len(points)))
        sorted_indices = [remaining.pop(0)]
        current_point = points[sorted_indices[0]]
        
        while remaining:
            # Find nearest unvisited point
            distances = np.linalg.norm(points[remaining] - current_point, axis=1)
            nearest_local_idx = np.argmin(distances)
            nearest_idx = remaining.pop(nearest_local_idx)
            sorted_indices.append(nearest_idx)
            current_point = points[nearest_idx]
        
        sorted_points = [points[i] for i in sorted_indices]
        sorted_triangles = [triangle_indices[i] for i in sorted_indices]
        
        return sorted_points, sorted_triangles
    
    def optimize_waypoints(
        self,
        waypoints_list: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Optimize the order of face painting to minimize travel distance.
        
        Simple greedy approach: always go to the nearest unvisited face center.
        
        Args:
            waypoints_list: List of waypoint lists for each face
            
        Returns:
            Optimized concatenated waypoint list
        """
        if not waypoints_list:
            return []
        
        num_faces = len(waypoints_list)
        visited = [False] * num_faces
        optimized = []
        
        # Start with first face
        current = 0
        visited[0] = True
        optimized.extend(waypoints_list[0])
        
        # Greedy nearest neighbor
        current_pos = waypoints_list[0][-1]
        
        for _ in range(num_faces - 1):
            # Find nearest unvisited face
            min_dist = float('inf')
            nearest = -1
            
            for i in range(num_faces):
                if not visited[i]:
                    face_start = waypoints_list[i][0]
                    dist = np.linalg.norm(current_pos - face_start)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i
            
            # Move to nearest face
            if nearest != -1:
                visited[nearest] = True
                optimized.extend(waypoints_list[nearest])
                current_pos = waypoints_list[nearest][-1]
        
        return optimized


class PathVisualizer:
    """Utilities for visualizing generated paths."""
    
    @staticmethod
    def export_waypoints_to_csv(
        waypoints: List[np.ndarray],
        output_path: str
    ) -> None:
        """Export waypoints to CSV file."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z'])
            for waypoint in waypoints:
                writer.writerow(waypoint)
    
    @staticmethod
    def visualize_mesh_with_paths(
        mesh: trimesh.Trimesh,
        face_segments: List[set],
        waypoints_list: List[List[np.ndarray]]
    ) -> None:
        """Visualize mesh with planned paths (requires visualization backend)."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                triangles=mesh.faces,
                alpha=0.3,
                color='cyan'
            )
            
            # Plot paths for each face
            colors = plt.cm.rainbow(np.linspace(0, 1, len(waypoints_list)))
            for waypoints, color in zip(waypoints_list, colors):
                waypoints_array = np.array(waypoints)
                ax.plot(
                    waypoints_array[:, 0],
                    waypoints_array[:, 1],
                    waypoints_array[:, 2],
                    color=color,
                    linewidth=2,
                    label=f'Face'
                )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Mesh Painting Paths')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
