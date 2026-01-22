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
        Generate systematic coverage path using organized grid pattern.
        
        Creates a structured raster pattern across the face that methodically
        covers the entire surface with organized, parallel paths.
        
        Args:
            vertices: Face vertices (N x 3)
            faces: Face indices (M x 3)
            face_normal: Normal vector of the face
            
        Returns:
            List of waypoints defining the spray path
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Generate organized grid-based coverage
        waypoints = self._generate_grid_coverage(mesh, vertices, faces, face_normal)
        
        return waypoints
    
    def _generate_grid_coverage(
        self,
        mesh: trimesh.Trimesh,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_normal: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate methodical coverage using a 2D grid projected onto the 3D face.
        
        Creates a regular raster pattern (like painting horizontal lines) that
        covers the face systematically from one side to the other.
        """
        # Create local coordinate system for the face
        face_center = vertices.mean(axis=0)
        u_vec = self._get_u_vector(face_normal)
        v_vec = np.cross(face_normal, u_vec)
        v_vec = v_vec / (np.linalg.norm(v_vec) + 1e-10)
        
        # Project vertices to 2D for bounds
        vertices_2d = self._project_to_2d(vertices, face_center, u_vec, v_vec)
        min_x = vertices_2d[:, 0].min()
        max_x = vertices_2d[:, 0].max()
        min_y = vertices_2d[:, 1].min()
        max_y = vertices_2d[:, 1].max()
        
        # Create grid of points following raster pattern
        waypoints = []
        y = min_y
        direction = 1  # Alternating direction for snake pattern
        
        while y <= max_y:
            # Generate sweep line at this y coordinate
            sweep_points = self._generate_grid_sweep(
                mesh, face_center, u_vec, v_vec, face_normal,
                y, min_x, max_x, direction, vertices_2d
            )
            waypoints.extend(sweep_points)
            
            y += self.path_spacing
            direction *= -1
        
        return waypoints
    
    def _generate_grid_sweep(
        self,
        mesh: trimesh.Trimesh,
        face_center: np.ndarray,
        u_vec: np.ndarray,
        v_vec: np.ndarray,
        face_normal: np.ndarray,
        y: float,
        min_x: float,
        max_x: float,
        direction: int,
        vertices_2d: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate a single sweep line with adaptive bounds following the face geometry.
        """
        # Find actual x extent at this y coordinate
        y_tolerance = self.path_spacing * 1.0
        y_indices = np.abs(vertices_2d[:, 1] - y) < y_tolerance
        
        if y_indices.any():
            x_at_y = vertices_2d[y_indices, 0]
            local_min_x = x_at_y.min()
            local_max_x = x_at_y.max()
        else:
            local_min_x = min_x
            local_max_x = max_x

        # print(local_max_x, local_min_x)

        local_max_x = max_x
        local_min_x = min_x
        
        # Apply alternating direction
        if direction < 0:
            local_min_x, local_max_x = local_max_x, local_min_x
        
        # Generate evenly-spaced points along this line
        num_points = max(5, int((abs(local_max_x - local_min_x) / self.path_spacing)))
        # num_points = 5
        x_coords = np.linspace(local_min_x, local_max_x, num_points)
        
        sweep_waypoints = []

        mesh.face_normals  # Ensure face normals are calculated
        vertex_normals = mesh.vertex_normals

        for x in x_coords:
            # Create 3D point in the face plane
            point_3d = face_center + x * u_vec + y * v_vec
            
            # Project to mesh surface
            projected_point = self._project_to_mesh_surface(mesh, point_3d)

            # # _, _, face_index = mesh.nearest.on_surface(projected_point.reshape(1, -1))
            # # local_normal = mesh.face_normals[face_index[0]]
            
            # # _, _, face_index = trimesh.proximity.closest_point(mesh, [projected_point])
            # face_index = trimesh.proximity.nearby_faces(mesh, [projected_point])
            # vertex_normals = trimesh.geometry.mean_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals)
            # # local_normal = mesh.face_normals[face_index[0][0]]
            # local_normal = vertex_normals[mesh.faces[face_index[0][0]]].mean(axis=0)

            _, _, face_index = mesh.nearest.on_surface(projected_point.reshape(1, 3))
            idx = face_index[0]

            face_vertices = mesh.faces[idx]
            local_normal = vertex_normals[face_vertices].mean(axis=0)

            local_normal /= np.linalg.norm(local_normal)

            print(local_normal)

            # Offset by spray height
            waypoint = projected_point + self.spray_height * local_normal
            # waypoint = projected_point
            sweep_waypoints.append(waypoint)
        
        return sweep_waypoints
    
    def _project_to_mesh_surface(
        self,
        mesh: trimesh.Trimesh,
        point: np.ndarray
    ) -> np.ndarray:
        """
        Project a point onto the mesh surface using efficient spatial lookup.
        
        Uses KDTree to find nearby vertices first, then checks adjacent faces.
        """
        # Build KDTree on first call for full mesh vertices
        mesh_id = id(mesh)
        if not hasattr(self, '_kdtree_cache'):
            self._kdtree_cache = {}
        
        if mesh_id not in self._kdtree_cache:
            self._kdtree_cache[mesh_id] = KDTree(mesh.vertices)
        
        kdtree = self._kdtree_cache[mesh_id]
        
        # Find k nearest vertices
        _, indices = kdtree.query(point, k=min(10, len(mesh.vertices)))
        
        min_dist = float('inf')
        closest_point = point.copy()
        
        # Check faces adjacent to nearest vertices
        checked_faces = set()
        for vertex_idx in indices:
            vertex_idx = int(vertex_idx)  # Ensure it's an int
            
            # Get faces adjacent to this vertex (safe indexing)
            if vertex_idx < len(mesh.vertex_faces):
                for face_idx in mesh.vertex_faces[vertex_idx]:
                    if face_idx < 0 or face_idx in checked_faces:
                        continue
                    
                    checked_faces.add(face_idx)
                    face = mesh.faces[face_idx]
                    tri_verts = mesh.vertices[face]
                    proj, dist = self._project_to_triangle(point, tri_verts)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = proj
        
        # Fallback: if no faces found, check all faces (shouldn't happen)
        if len(checked_faces) == 0:
            for face in mesh.faces:
                tri_verts = mesh.vertices[face]
                proj, dist = self._project_to_triangle(point, tri_verts)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = proj
        
        return closest_point
    
    def _project_to_triangle(
        self,
        point: np.ndarray,
        tri_verts: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Project point onto triangle surface.
        
        Returns (projected_point, distance_to_plane).
        """
        v0, v1, v2 = tri_verts
        
        # Triangle edges and normal
        edge0 = v1 - v0
        edge1 = v2 - v0
        normal = np.cross(edge0, edge1)
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-10:
            # Degenerate triangle
            return tri_verts.mean(axis=0), np.linalg.norm(point - tri_verts.mean(axis=0))
        
        normal = normal / norm_len
        
        # Project onto plane
        to_point = point - v0
        dist_to_plane = np.dot(to_point, normal)
        proj_plane = point - dist_to_plane * normal
        
        # Barycentric coordinates
        e0_e0 = np.dot(edge0, edge0)
        e0_e1 = np.dot(edge0, edge1)
        e1_e1 = np.dot(edge1, edge1)
        e0_to_proj = np.dot(edge0, proj_plane - v0)
        e1_to_proj = np.dot(edge1, proj_plane - v0)
        
        denom = e0_e0 * e1_e1 - e0_e1 * e0_e1
        
        if abs(denom) < 1e-10:
            return proj_plane, abs(dist_to_plane)
        
        u = (e1_e1 * e0_to_proj - e0_e1 * e1_to_proj) / denom
        v = (e0_e0 * e1_to_proj - e0_e1 * e0_to_proj) / denom
        
        # Check if inside triangle
        if u >= -1e-10 and v >= -1e-10 and u + v <= 1.0 + 1e-10:
            return proj_plane, abs(dist_to_plane)
        
        # Clamp to triangle edges
        closest = proj_plane
        min_edge_dist = float('inf')
        
        for edge_start, edge_end in [(v0, v1), (v1, v2), (v2, v0)]:
            edge = edge_end - edge_start
            edge_len_sq = np.dot(edge, edge)
            
            if edge_len_sq < 1e-10:
                continue
            
            t = np.clip(np.dot(proj_plane - edge_start, edge) / edge_len_sq, 0, 1)
            edge_pt = edge_start + t * edge
            d = np.linalg.norm(proj_plane - edge_pt)
            
            if d < min_edge_dist:
                min_edge_dist = d
                closest = edge_pt
        
        return closest, abs(dist_to_plane) + min_edge_dist
    
    def _get_u_vector(self, normal: np.ndarray) -> np.ndarray:
        """Get first orthogonal vector in the plane defined by normal."""
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
