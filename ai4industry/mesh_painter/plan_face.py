#!/usr/bin/env python3
"""
Plan and visualize painting paths for a single face mesh.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import numpy as np
import trimesh
from path_planning import PathPlanner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plan_single_face(face_stl: str, spray_height: float = 0.20, 
                     spray_radius: float = 0.10, path_spacing: float = 0.05,
                     visualize: bool = True):
    """
    Plan painting path for a single face mesh.
    
    Args:
        face_stl: Path to face STL file
        spray_height: Height of spray head above surface
        spray_radius: Radius of spray pattern
        path_spacing: Spacing between path lines
        visualize: Whether to visualize the result
    """
    
    face_path = Path(face_stl)
    if not face_path.exists():
        print(f"Error: {face_stl} not found")
        return None
    
    print(f"\nLoading face mesh: {face_stl}")
    mesh = trimesh.load(str(face_path))
    
    print(f"✓ Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Area: {mesh.area:.4f} m²")
    print(f"  Bounds: X=[{mesh.bounds[0,0]:.3f}, {mesh.bounds[1,0]:.3f}]")
    print(f"          Y=[{mesh.bounds[0,1]:.3f}, {mesh.bounds[1,1]:.3f}]")
    print(f"          Z=[{mesh.bounds[0,2]:.3f}, {mesh.bounds[1,2]:.3f}]")
    
    # Plan path
    print(f"\nPlanning path with:")
    print(f"  Spray height: {spray_height:.3f}m")
    print(f"  Spray radius: {spray_radius:.3f}m")
    print(f"  Path spacing: {path_spacing:.3f}m")
    
    planner = PathPlanner(
        spray_height=spray_height,
        spray_radius=spray_radius,
        path_spacing=path_spacing
    )
    
    # Get face normal
    face_normal = mesh.face_normals[0]
    
    waypoints = planner.plan_path(mesh.vertices, mesh.faces, face_normal)
    waypoints = np.array(waypoints)
    
    print(f"\n✓ Path planning complete:")
    print(f"  Total waypoints: {len(waypoints)}")
    print(f"  Path length: {np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)):.2f}m")
    print(f"  Bounds:")
    print(f"    X: [{waypoints[:, 0].min():.3f}, {waypoints[:, 0].max():.3f}]")
    print(f"    Y: [{waypoints[:, 1].min():.3f}, {waypoints[:, 1].max():.3f}]")
    print(f"    Z: [{waypoints[:, 2].min():.3f}, {waypoints[:, 2].max():.3f}]")
    
    # Analyze waypoint distribution
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    print(f"  Waypoint distances:")
    print(f"    Mean: {distances.mean():.4f}m")
    print(f"    Min: {distances.min():.4f}m")
    print(f"    Max: {distances.max():.4f}m")
    
    if visualize:
        visualize_path(mesh, waypoints, face_path.stem)
    
    # Save waypoints to CSV
    output_csv = face_path.parent / f"{face_path.stem}_waypoints.csv"
    np.savetxt(output_csv, waypoints, delimiter=',', header='X,Y,Z', comments='')
    print(f"\n✓ Saved waypoints to: {output_csv}")
    
    return waypoints


def visualize_path(mesh, waypoints, title="Face Path"):
    """Visualize mesh and planned path."""
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D view with mesh and path
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot mesh
    vertices = mesh.vertices
    faces = mesh.faces
    
    poly = [[vertices[faces[j]] for j in range(len(faces))]]
    ax1.add_collection3d(Poly3DCollection(poly[0], alpha=0.3, 
                                         edgecolor='lightgray', linewidth=0.2,
                                         facecolor='cyan'))
    
    # Plot path
    ax1.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
            'r-', linewidth=1, label='Path', alpha=0.8)
    ax1.scatter(waypoints[0, 0], waypoints[0, 1], waypoints[0, 2], 
               c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], 
               c='red', s=100, marker='s', label='End', zorder=5)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title}\n3D Path View')
    ax1.legend()
    
    # 2D top view (X-Y)
    ax2 = fig.add_subplot(132)
    ax2.plot(waypoints[:, 0], waypoints[:, 1], 'r-', linewidth=1.5, label='Path')
    ax2.scatter(waypoints[0, 0], waypoints[0, 1], c='green', s=100, marker='o', label='Start')
    ax2.scatter(waypoints[-1, 0], waypoints[-1, 1], c='red', s=100, marker='s', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # Waypoint distance analysis
    ax3 = fig.add_subplot(133)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    ax3.plot(distances, 'b-', linewidth=1.5, label='Distance between waypoints')
    ax3.axhline(y=distances.mean(), color='r', linestyle='--', label=f'Mean: {distances.mean():.4f}m')
    ax3.set_xlabel('Waypoint Index')
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('Waypoint Spacing Analysis')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plan path for a single face")
    parser.add_argument("face_stl", type=str, help="Path to face STL file")
    parser.add_argument("--spray-height", type=float, default=0.20, 
                       help="Spray height (m, default: 0.20)")
    parser.add_argument("--spray-radius", type=float, default=0.10,
                       help="Spray radius (m, default: 0.10)")
    parser.add_argument("--path-spacing", type=float, default=0.05,
                       help="Path spacing (m, default: 0.05)")
    parser.add_argument("--no-vis", action="store_true", help="Don't visualize")
    
    args = parser.parse_args()
    
    plan_single_face(
        args.face_stl,
        spray_height=args.spray_height,
        spray_radius=args.spray_radius,
        path_spacing=args.path_spacing,
        visualize=not args.no_vis
    )


if __name__ == "__main__":
    main()
