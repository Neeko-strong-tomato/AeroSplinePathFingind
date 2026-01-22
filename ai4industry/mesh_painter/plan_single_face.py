#!/usr/bin/env python3
"""
Plan and optimize painting paths for a single face STL file.

Use this to iteratively improve path planning for individual faces.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import trimesh
import numpy as np
from path_planning import PathPlanner, PathVisualizer


def plan_single_face(face_stl: str, output_csv: str = None,
                    spray_height: float = 0.20,
                    spray_radius: float = 0.10,
                    path_spacing: float = 0.05,
                    visualize: bool = False,
                    viz_dir: str = "face_visualizations"):
    """Plan painting path for a single face."""
    
    face_path = Path(face_stl)
    if not face_path.exists():
        print(f"Error: Face STL file not found: {face_stl}")
        return
    
    print("\n" + "="*60)
    print("SINGLE FACE PATH PLANNING")
    print("="*60)
    
    # Load face mesh
    mesh = trimesh.load(face_stl)
    print(f"Loaded face mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Extract vertices and triangles
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Use first face normal as reference
    face_normal = mesh.face_normals[0]
    print(f"Face normal: [{face_normal[0]:.4f}, {face_normal[1]:.4f}, {face_normal[2]:.4f}]")
    
    # Generate path
    print("\n" + "="*60)
    print("PATH PLANNING")
    print("="*60)
    print(f"Parameters:")
    print(f"  Spray height: {spray_height}m")
    print(f"  Spray radius: {spray_radius}m")
    print(f"  Path spacing: {path_spacing}m")
    
    planner = PathPlanner(
        spray_height=spray_height,
        spray_radius=spray_radius,
        path_spacing=path_spacing
    )
    
    waypoints = planner.plan_path(vertices, faces, face_normal)
    waypoints = np.array(waypoints)
    
    print(f"\n✓ Generated {len(waypoints)} waypoints")
    print(f"  Bounds:")
    print(f"    X: [{waypoints[:, 0].min():.3f}, {waypoints[:, 0].max():.3f}]")
    print(f"    Y: [{waypoints[:, 1].min():.3f}, {waypoints[:, 1].max():.3f}]")
    print(f"    Z: [{waypoints[:, 2].min():.3f}, {waypoints[:, 2].max():.3f}]")
    
    # Calculate path metrics
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    path_length = np.sum(distances)
    
    print(f"\nPath Metrics:")
    print(f"  Total length: {path_length:.2f}m")
    print(f"  Mean waypoint spacing: {distances.mean():.4f}m")
    print(f"  Max jump: {distances.max():.4f}m")
    print(f"  Large jumps (>0.2m): {np.sum(distances > 0.2)}")
    
    # Analyze grid pattern
    y_coords = waypoints[:, 1]
    y_sorted = np.sort(np.unique(np.round(y_coords, decimals=3)))
    
    if len(y_sorted) > 1:
        y_spacing = np.diff(y_sorted)
        print(f"\nGrid Pattern:")
        print(f"  Sweep lines: {len(y_sorted)}")
        print(f"  Y-spacing: min={y_spacing.min():.4f}m, max={y_spacing.max():.4f}m, avg={y_spacing.mean():.4f}m")
    
    # Export waypoints
    output_file = output_csv
    if output_file is None:
        face_name = face_path.stem
        output_file = f"{face_name}_waypoints.csv"
    
    print("\n" + "="*60)
    print("EXPORT")
    print("="*60)
    print(f"Exporting waypoints to: {output_file}")
    PathVisualizer.export_waypoints_to_csv(waypoints, output_file)
    print("✓ Waypoints exported")
    
    # Generate visualization if requested
    if visualize:
        print("\n" + "="*60)
        print("VISUALIZATION")
        print("="*60)
        
        viz_path = Path(viz_dir)
        viz_path.mkdir(parents=True, exist_ok=True)
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D visualization
        fig = plt.figure(figsize=(14, 6))
        
        # Subplot 1: Mesh with waypoints
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c='darkblue', s=1, alpha=0.3, label='Mesh vertices')
        ax1.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                'r-', linewidth=1, label='Path', alpha=0.8)
        ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                   c='red', s=5, alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Face Mesh with Planned Path')
        ax1.legend()
        
        # Subplot 2: Top-down view (2D path)
        ax2 = fig.add_subplot(122)
        ax2.scatter(vertices[:, 0], vertices[:, 1], c='darkblue', s=2, alpha=0.3, label='Mesh')
        ax2.plot(waypoints[:, 0], waypoints[:, 1], 'r-', linewidth=0.5, label='Path')
        ax2.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=2, alpha=0.6)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-Down Path View')
        ax2.legend()
        ax2.axis('equal')
        
        plt.tight_layout()
        viz_file = viz_path / f"{face_path.stem}_path.png"
        plt.savefig(str(viz_file), dpi=150)
        print(f"✓ Visualization saved to {viz_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plan and optimize painting paths for a single face STL file"
    )
    parser.add_argument(
        "face_stl",
        type=str,
        help="Path to extracted face STL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for waypoints (default: {face}_waypoints.csv)"
    )
    parser.add_argument(
        "--spray-height",
        type=float,
        default=0.20,
        help="Height of spray head above surface (meters, default: 0.20)"
    )
    parser.add_argument(
        "--spray-radius",
        type=float,
        default=0.10,
        help="Radius of spray pattern (meters, default: 0.10)"
    )
    parser.add_argument(
        "--path-spacing",
        type=float,
        default=0.05,
        help="Spacing between path lines (meters, default: 0.05)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="face_visualizations",
        help="Directory to save visualizations (default: face_visualizations)"
    )
    
    args = parser.parse_args()
    
    plan_single_face(
        args.face_stl,
        output_csv=args.output,
        spray_height=args.spray_height,
        spray_radius=args.spray_radius,
        path_spacing=args.path_spacing,
        visualize=args.visualize,
        viz_dir=args.viz_dir
    )


if __name__ == "__main__":
    main()
