"""
Visualization module for mesh painter process steps.

Provides comprehensive visualizations for:
1. Original mesh
2. Mesh segmentation into faces
3. Path planning for each face
4. Final optimized path
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from typing import List, Set, Tuple, Optional
from pathlib import Path


class MeshVisualizer:
    """Visualization utilities for mesh painting process."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_original_mesh(
        self,
        mesh: trimesh.Trimesh,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize the original mesh."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh surface
        vertices = mesh.vertices
        faces = mesh.faces
        
        mesh_plot = Poly3DCollection(
            vertices[faces],
            alpha=0.7,
            facecolor='cyan',
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_collection3d(mesh_plot)
        
        # Set limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Original Mesh')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(str(self.output_dir / "01_original_mesh.png"), dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_segmentation(
        self,
        mesh: trimesh.Trimesh,
        face_segments: List[Set[int]],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize mesh with segmented faces in different colors."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Generate colors for each segment
        colors = plt.cm.tab20(np.linspace(0, 1, len(face_segments)))
        
        # Plot each face segment with different color
        for seg_idx, face_indices in enumerate(face_segments):
            face_list = list(face_indices)
            mesh_faces = faces[face_list]
            
            mesh_plot = Poly3DCollection(
                vertices[mesh_faces],
                alpha=0.8,
                facecolor=colors[seg_idx % len(colors)],
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_collection3d(mesh_plot)
        
        # Set limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Mesh Segmentation ({len(face_segments)} faces)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(str(self.output_dir / "02_segmentation.png"), dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_individual_faces(
        self,
        mesh: trimesh.Trimesh,
        face_segments: List[Set[int]],
        save_dir: Optional[str] = None
    ) -> None:
        """Visualize each face individually."""
        save_dir = Path(save_dir) if save_dir else self.output_dir / "03_individual_faces"
        save_dir.mkdir(exist_ok=True)
        
        for face_idx, face_indices in enumerate(face_segments):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Get faces for this segment
            face_list = list(face_indices)
            mesh_faces = faces[face_list]
            
            # Plot this segment
            mesh_plot = Poly3DCollection(
                vertices[mesh_faces],
                alpha=0.8,
                facecolor='lightblue',
                edgecolor='navy',
                linewidth=1
            )
            ax.add_collection3d(mesh_plot)
            
            # Get bounds for this face
            face_vertices = vertices[mesh_faces.flatten()]
            ax.set_xlim(face_vertices[:, 0].min() - 0.1, face_vertices[:, 0].max() + 0.1)
            ax.set_ylim(face_vertices[:, 1].min() - 0.1, face_vertices[:, 1].max() + 0.1)
            ax.set_zlim(face_vertices[:, 2].min() - 0.1, face_vertices[:, 2].max() + 0.1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Face {face_idx} ({len(face_indices)} triangles)')
            
            plt.tight_layout()
            plt.savefig(str(save_dir / f"face_{face_idx:02d}.png"), dpi=150, bbox_inches='tight')
            plt.close()
    
    def visualize_face_with_path(
        self,
        mesh: trimesh.Trimesh,
        face_indices: Set[int],
        waypoints: List[np.ndarray],
        face_idx: int = 0,
        save_dir: Optional[str] = None
    ) -> None:
        """Visualize a single face with its painting path."""
        save_dir = Path(save_dir) if save_dir else self.output_dir / "04_paths_per_face"
        save_dir.mkdir(exist_ok=True)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Get faces for this segment
        face_list = list(face_indices)
        mesh_faces = faces[face_list]
        
        # Plot face
        mesh_plot = Poly3DCollection(
            vertices[mesh_faces],
            alpha=0.6,
            facecolor='lightblue',
            edgecolor='navy',
            linewidth=0.5
        )
        ax.add_collection3d(mesh_plot)
        
        # Plot path
        if waypoints:
            waypoints_array = np.array(waypoints)
            ax.plot(
                waypoints_array[:, 0],
                waypoints_array[:, 1],
                waypoints_array[:, 2],
                'r-',
                linewidth=2,
                label='Spray Path'
            )
            
            # Mark start and end
            ax.scatter(*waypoints[0], color='green', s=100, marker='o', label='Start')
            ax.scatter(*waypoints[-1], color='red', s=100, marker='s', label='End')
        
        # Get bounds
        face_vertices = vertices[mesh_faces.flatten()]
        if waypoints:
            all_points = np.vstack([face_vertices, waypoints_array])
        else:
            all_points = face_vertices
        
        margin = 0.05 * (all_points.max(axis=0) - all_points.min(axis=0))
        ax.set_xlim(all_points[:, 0].min() - margin[0], all_points[:, 0].max() + margin[0])
        ax.set_ylim(all_points[:, 1].min() - margin[1], all_points[:, 1].max() + margin[1])
        ax.set_zlim(all_points[:, 2].min() - margin[2], all_points[:, 2].max() + margin[2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Face {face_idx} with Painting Path ({len(waypoints)} waypoints)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(str(save_dir / f"path_face_{face_idx:02d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_all_paths(
        self,
        mesh: trimesh.Trimesh,
        face_segments: List[Set[int]],
        waypoints_list: List[List[np.ndarray]],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize mesh with all painting paths overlaid."""
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot mesh
        mesh_plot = Poly3DCollection(
            vertices[faces],
            alpha=0.3,
            facecolor='lightgray',
            edgecolor='gray',
            linewidth=0.3
        )
        ax.add_collection3d(mesh_plot)
        
        # Plot paths for each face
        colors = plt.cm.rainbow(np.linspace(0, 1, len(waypoints_list)))
        
        for waypoints, color in zip(waypoints_list, colors):
            if waypoints:
                waypoints_array = np.array(waypoints)
                ax.plot(
                    waypoints_array[:, 0],
                    waypoints_array[:, 1],
                    waypoints_array[:, 2],
                    color=color,
                    linewidth=1.5,
                    alpha=0.8
                )
        
        # Set limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'All Painting Paths ({len(waypoints_list)} faces)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(str(self.output_dir / "05_all_paths.png"), dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_optimized_path(
        self,
        mesh: trimesh.Trimesh,
        waypoints: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize the optimized complete path."""
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot mesh
        mesh_plot = Poly3DCollection(
            vertices[faces],
            alpha=0.2,
            facecolor='lightgray',
            edgecolor='gray',
            linewidth=0.3
        )
        ax.add_collection3d(mesh_plot)
        
        # Plot optimized path
        if waypoints:
            waypoints_array = np.array(waypoints)
            ax.plot(
                waypoints_array[:, 0],
                waypoints_array[:, 1],
                waypoints_array[:, 2],
                'b-',
                linewidth=1.5,
                alpha=0.9,
                label='Optimized Path'
            )
            
            # Mark key points
            ax.scatter(*waypoints[0], color='green', s=150, marker='o', label='Start', zorder=5)
            ax.scatter(*waypoints[-1], color='red', s=150, marker='s', label='End', zorder=5)
        
        # Set limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Optimized Painting Path ({len(waypoints)} waypoints)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(str(self.output_dir / "06_optimized_path.png"), dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_path_statistics(
        self,
        face_segments: List[Set[int]],
        waypoints_list: List[List[np.ndarray]],
        save_path: Optional[str] = None
    ) -> None:
        """Generate statistics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Face sizes (number of triangles)
        face_sizes = [len(seg) for seg in face_segments]
        axes[0, 0].bar(range(len(face_sizes)), face_sizes, color='steelblue')
        axes[0, 0].set_xlabel('Face Index')
        axes[0, 0].set_ylabel('Number of Triangles')
        axes[0, 0].set_title('Face Sizes')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Number of waypoints per face
        waypoint_counts = [len(wp) for wp in waypoints_list]
        axes[0, 1].bar(range(len(waypoint_counts)), waypoint_counts, color='coral')
        axes[0, 1].set_xlabel('Face Index')
        axes[0, 1].set_ylabel('Number of Waypoints')
        axes[0, 1].set_title('Waypoints Per Face')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Total path length per face
        path_lengths = []
        for waypoints in waypoints_list:
            if len(waypoints) > 1:
                waypoints_array = np.array(waypoints)
                distances = np.linalg.norm(
                    np.diff(waypoints_array, axis=0),
                    axis=1
                )
                path_lengths.append(distances.sum())
            else:
                path_lengths.append(0)
        
        axes[1, 0].bar(range(len(path_lengths)), path_lengths, color='mediumseagreen')
        axes[1, 0].set_xlabel('Face Index')
        axes[1, 0].set_ylabel('Total Path Length (m)')
        axes[1, 0].set_title('Path Length Per Face')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""
Segmentation Statistics:

Total Faces: {len(face_segments)}
Total Triangles: {sum(face_sizes)}

Waypoints:
  Per Face: {np.mean(waypoint_counts):.1f} ± {np.std(waypoint_counts):.1f}
  Total: {sum(waypoint_counts)}

Path Length:
  Per Face: {np.mean(path_lengths):.3f} ± {np.std(path_lengths):.3f} m
  Total: {sum(path_lengths):.3f} m

Face Sizes:
  Min: {min(face_sizes)}
  Max: {max(face_sizes)}
  Avg: {np.mean(face_sizes):.1f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(str(self.output_dir / "07_statistics.png"), dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(
        self,
        mesh: trimesh.Trimesh,
        face_segments: List[Set[int]],
        waypoints_list: List[List[np.ndarray]],
        optimized_waypoints: List[np.ndarray]
    ) -> None:
        """Generate complete visualization report."""
        print("Generating visualizations...")
        print("  [1/7] Original mesh")
        self.visualize_original_mesh(mesh)
        
        print("  [2/7] Segmentation")
        self.visualize_segmentation(mesh, face_segments)
        
        print("  [3/7] Individual faces")
        self.visualize_individual_faces(mesh, face_segments)
        
        print("  [4/7] Paths per face")
        for i, (face_indices, waypoints) in enumerate(zip(face_segments, waypoints_list)):
            self.visualize_face_with_path(mesh, face_indices, waypoints, i)
        
        print("  [5/7] All paths overlay")
        self.visualize_all_paths(mesh, face_segments, waypoints_list)
        
        print("  [6/7] Optimized path")
        self.visualize_optimized_path(mesh, optimized_waypoints)
        
        print("  [7/7] Statistics")
        self.visualize_path_statistics(face_segments, waypoints_list)
        
        print(f"Visualizations saved to: {self.output_dir}")
