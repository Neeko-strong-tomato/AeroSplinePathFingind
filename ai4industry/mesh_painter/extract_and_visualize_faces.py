#!/usr/bin/env python3
"""
Extract individual face meshes from STL file and visualize them for selection.
Allows user to see all faces and decide which ones to work on.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import trimesh
import numpy as np
from mesh_segmentation import MeshFaceSegmenter
import json

def extract_faces(stl_file: str, angle_threshold: float = 60, bend_threshold: float = 60):
    """
    Extract individual face meshes and save them as separate STL files.
    Also create a summary JSON with face metadata.
    """
    stl_path = Path(stl_file)
    faces_dir = stl_path.parent / "faces"
    faces_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading mesh: {stl_file}")
    segmenter = MeshFaceSegmenter(
        angle_threshold=angle_threshold,
        bend_angle_threshold=bend_threshold
    )
    mesh = segmenter.load_stl(stl_file)
    
    print(f"Segmenting mesh with angles: {angle_threshold}°, {bend_threshold}°")
    face_segments = segmenter.segment()
    
    print(f"\n✓ Found {len(face_segments)} faces")
    print("=" * 70)
    
    face_metadata = []
    
    for face_idx, face_indices in enumerate(face_segments):
        # Get vertices and triangles for this face
        vertices, triangles = segmenter.get_face_geometry(face_indices)
        
        # Create submesh for this face
        face_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # Calculate metrics
        area = face_mesh.area
        bounds = face_mesh.bounds
        bounds_size = bounds[1] - bounds[0]
        centroid = face_mesh.centroid
        
        # Save STL
        face_file = faces_dir / f"face_{face_idx:03d}.stl"
        face_mesh.export(str(face_file))
        
        metadata = {
            "face_id": face_idx,
            "file": str(face_file.name),
            "num_triangles": len(triangles),
            "num_vertices": len(vertices),
            "area_m2": float(area),
            "bounds": {
                "min": [float(x) for x in bounds[0]],
                "max": [float(x) for x in bounds[1]],
                "size": [float(x) for x in bounds_size]
            },
            "centroid": [float(x) for x in centroid],
            "volume": float(face_mesh.volume)
        }
        face_metadata.append(metadata)
        
        # Print summary
        print(f"\nFace {face_idx:2d}:")
        print(f"  File: {face_file.name}")
        print(f"  Triangles: {len(triangles):5d}")
        print(f"  Vertices: {len(vertices):5d}")
        print(f"  Area: {area:10.4f} m²")
        print(f"  Bounds: X=[{bounds[0,0]:7.3f}, {bounds[1,0]:7.3f}] m")
        print(f"           Y=[{bounds[0,1]:7.3f}, {bounds[1,1]:7.3f}] m")
        print(f"           Z=[{bounds[0,2]:7.3f}, {bounds[1,2]:7.3f}] m")
        print(f"  Size: {bounds_size[0]:6.3f} × {bounds_size[1]:6.3f} × {bounds_size[2]:6.3f} m")
    
    # Save metadata
    metadata_file = faces_dir / "faces_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "source_file": stl_file,
            "angle_threshold": angle_threshold,
            "bend_threshold": bend_threshold,
            "num_faces": len(face_metadata),
            "faces": face_metadata
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Saved {len(face_metadata)} face meshes to: {faces_dir}")
    print(f"✓ Metadata saved to: {metadata_file}")
    print("\nTo visualize a face, open one of the STL files in:")
    print(f"  {faces_dir}")
    print("\nEdit faces_metadata.json to mark which faces to keep (set 'keep': true/false)")
    
    return faces_dir, metadata_file


def visualize_faces_summary(metadata_file: str):
    """
    Print a summary table of all faces with their properties.
    """
    metadata_file = Path(metadata_file)
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 120)
    print("FACE SUMMARY TABLE")
    print("=" * 120)
    print(f"{'ID':3} {'File':20} {'Tris':6} {'Verts':6} {'Area':10} {'Size X':8} {'Size Y':8} {'Size Z':8} {'Volume':10}")
    print("-" * 120)
    
    total_area = 0
    for face in data['faces']:
        size = face['bounds']['size']
        area = face['area_m2']
        total_area += area
        
        print(f"{face['face_id']:3d} {face['file']:20} {face['num_triangles']:6d} {face['num_vertices']:6d} "
              f"{area:10.4f} {size[0]:8.3f} {size[1]:8.3f} {size[2]:8.3f} {face['volume']:10.4f}")
    
    print("-" * 120)
    print(f"{'TOTAL':3} {' ':20} {' ':6} {' ':6} {total_area:10.4f}")
    print("=" * 120)


def delete_uninteresting_faces(metadata_file: str, keep_indices: list):
    """
    Delete face files that are not in the keep_indices list.
    """
    metadata_file = Path(metadata_file)
    faces_dir = metadata_file.parent
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    faces_to_delete = []
    for face in data['faces']:
        if face['face_id'] not in keep_indices:
            faces_to_delete.append(face['face_id'])
    
    print(f"\nDeleting {len(faces_to_delete)} faces: {faces_to_delete}")
    
    for face_id in faces_to_delete:
        face_file = faces_dir / f"face_{face_id:03d}.stl"
        if face_file.exists():
            face_file.unlink()
            print(f"  Deleted: {face_file.name}")
    
    # Update metadata to only keep selected faces
    data['faces'] = [f for f in data['faces'] if f['face_id'] in keep_indices]
    data['num_faces'] = len(keep_indices)
    
    with open(metadata_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Kept {len(keep_indices)} faces")
    print(f"✓ Updated metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize mesh faces")
    parser.add_argument("stl_file", type=str, help="Path to STL file")
    parser.add_argument("--angle-threshold", type=float, default=60, help="Angle threshold (default: 60)")
    parser.add_argument("--bend-threshold", type=float, default=60, help="Bend threshold (default: 60)")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary of existing faces")
    parser.add_argument("--keep", type=int, nargs="+", help="Face IDs to keep (delete others)")
    
    args = parser.parse_args()
    
    if args.summary_only:
        # Show summary of existing faces
        stl_path = Path(args.stl_file)
        metadata_file = stl_path.parent / "faces" / "faces_metadata.json"
        if metadata_file.exists():
            visualize_faces_summary(str(metadata_file))
        else:
            print(f"No metadata found at {metadata_file}")
    elif args.keep is not None:
        # Delete uninteresting faces
        stl_path = Path(args.stl_file)
        metadata_file = stl_path.parent / "faces" / "faces_metadata.json"
        delete_uninteresting_faces(str(metadata_file), args.keep)
        visualize_faces_summary(str(metadata_file))
    else:
        # Extract faces
        faces_dir, metadata_file = extract_faces(
            args.stl_file,
            angle_threshold=args.angle_threshold,
            bend_threshold=args.bend_threshold
        )
        visualize_faces_summary(metadata_file)


if __name__ == "__main__":
    main()
