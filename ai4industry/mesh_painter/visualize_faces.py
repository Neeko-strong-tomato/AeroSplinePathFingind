#!/usr/bin/env python3
"""
Interactive visualization of all extracted faces in a matplotlib window.
Use the GUI to inspect each face and decide which ones to keep.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import numpy as np


def load_all_faces(faces_dir: str):
    """Load all face meshes from directory."""
    faces_dir = Path(faces_dir)
    metadata_file = faces_dir / "faces_metadata.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    faces = []
    for face_info in metadata['faces']:
        face_file = faces_dir / face_info['file']
        mesh = trimesh.load(str(face_file))
        faces.append({
            'id': face_info['face_id'],
            'file': face_info['file'],
            'mesh': mesh,
            'triangles': face_info['num_triangles'],
            'area': face_info['area_m2'],
            'info': face_info
        })
    
    return faces, metadata


def visualize_all_faces(faces_dir: str):
    """Display all faces in an interactive matplotlib grid."""
    faces, metadata = load_all_faces(faces_dir)
    
    # Calculate grid dimensions (roughly square)
    n_faces = len(faces)
    n_cols = int(np.ceil(np.sqrt(n_faces)))
    n_rows = int(np.ceil(n_faces / n_cols))
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'All Extracted Faces ({n_faces} total) - Close window when done reviewing', 
                 fontsize=16, fontweight='bold')
    
    # Color map: green for large faces, yellow for medium, red for small
    def get_color(area):
        if area > 0.5:
            return '#2ecc71'  # Green - good for painting
        elif area > 0.01:
            return '#f39c12'  # Orange - medium
        else:
            return '#e74c3c'  # Red - too small
    
    for idx, face_info in enumerate(faces):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        mesh = face_info['mesh']
        verts = mesh.vertices
        faces_tri = mesh.faces
        
        # Create 3D polygon collection
        poly = [[verts[faces_tri[j]] for j in range(len(faces_tri))]]
        ax.add_collection3d(Poly3DCollection(poly[0], alpha=0.7, 
                                           edgecolor='black', linewidth=0.5,
                                           facecolor=get_color(face_info['area'])))
        
        # Set limits
        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        
        # Labels and title
        area = face_info['area']
        title_color = 'green' if area > 0.5 else 'orange' if area > 0.01 else 'red'
        ax.set_title(f"Face {face_info['id']}: {area:.4f}m² ({face_info['triangles']} tri)", 
                    fontweight='bold', color=title_color, fontsize=10)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Large (>0.5 m²) - Good for painting'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Medium (0.01-0.5 m²)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Small (<0.01 m²) - Noise/artifacts'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()
    
    # Print summary after window closes
    print("\n" + "="*70)
    print("FACE INSPECTION COMPLETE")
    print("="*70)
    print("\nDecide which faces to keep:")
    print("\nTo keep specific faces, run:")
    print(f"  poetry run python extract_and_visualize_faces.py stl/test_part_m.stl --keep FACE_IDS")
    print("\nExamples:")
    print("  poetry run python extract_and_visualize_faces.py stl/test_part_m.stl --keep 0 1 2 3")
    print("  poetry run python extract_and_visualize_faces.py stl/test_part_m.stl --keep 0 1")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize all extracted faces")
    parser.add_argument("--faces-dir", type=str, default="stl/faces", 
                       help="Path to faces directory (default: stl/faces)")
    
    args = parser.parse_args()
    visualize_all_faces(args.faces_dir)
