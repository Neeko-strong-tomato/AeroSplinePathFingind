#!/usr/bin/env python3
"""Mesh simplification utility to reduce polygon count and file size."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import trimesh
import numpy as np


def simplify_mesh_saliency(mesh, target_count=None, target_reduction=0.5):
    """Simplify mesh by removing low-saliency faces."""
    original_faces = len(mesh.faces)
    
    if target_count is None:
        target_count = max(4, int(original_faces * target_reduction))
    
    print(f"Using saliency-based simplification...")
    
    simplified = mesh.copy()
    pbar = tqdm(total=100, desc="Simplifying", unit="%")
    
    while len(simplified.faces) > target_count:
        try:
            # Calculate face areas
            areas = simplified.area_faces
            
            if len(areas) == 0:
                break
            
            # Find faces with smallest areas
            threshold_pct = (target_count / len(simplified.faces))
            num_to_keep = max(target_count, int(len(simplified.faces) * threshold_pct))
            
            # Keep faces above median area
            median_area = np.median(areas)
            keep_mask = areas >= median_area * 0.5
            
            # Remove very small faces
            min_area = np.min(areas[areas > 0])
            keep_mask = keep_mask | (areas >= min_area * 2)
            
            if keep_mask.sum() >= len(keep_mask) * 0.95:
                # Not enough progress, try more aggressive
                keep_mask = areas >= np.percentile(areas, 30)
            
            if keep_mask.sum() < len(keep_mask):
                simplified = trimesh.Trimesh(
                    vertices=simplified.vertices,
                    faces=simplified.faces[keep_mask],
                    process=False
                )
                simplified.remove_unreferenced_vertices()
                
                reduction_pct = (1 - len(simplified.faces) / original_faces) * 100
                pbar.n = min(int(reduction_pct), 99)
                pbar.refresh()
            else:
                break
            
            if len(simplified.faces) <= target_count:
                break
                
        except Exception as e:
            print(f"\nSimplification error: {e}")
            break
    
    pbar.n = 100
    pbar.close()
    return simplified


def main():
    parser = argparse.ArgumentParser(description="Simplify STL meshes to reduce polygon count")
    parser.add_argument("input_file", type=str, help="Input STL file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output STL file")
    parser.add_argument("--reduction", "-r", type=float, default=0.5, help="Reduction ratio 0-1")
    parser.add_argument("--target-faces", "-t", type=int, default=None, help="Target number of faces")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_simplified.stl"
    
    print("\n" + "="*60)
    print("LOADING MESH")
    print("="*60)
    print(f"Loading: {input_path}")
    mesh = trimesh.load(input_path)
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    input_size = input_path.stat().st_size / 1024
    print(f"File size: {input_size:.2f} KB")
    
    print("\n" + "="*60)
    print("SIMPLIFICATION")
    print("="*60)
    print(f"Target reduction: {args.reduction * 100:.0f}%")
    if args.target_faces:
        print(f"Target faces: {args.target_faces}")
    
    mesh = simplify_mesh_saliency(mesh, target_count=args.target_faces, target_reduction=args.reduction)
    
    print("\n" + "="*60)
    print("EXPORT")
    print("="*60)
    print(f"Saving to: {output_path}")
    mesh.export(str(output_path))
    
    output_size = output_path.stat().st_size / 1024
    original_mesh = trimesh.load(input_path)
    original_face_count = len(original_mesh.faces)
    final_face_count = len(mesh.faces)
    
    print(f"\n✓ Simplified mesh: {len(mesh.vertices)} vertices, {final_face_count} faces")
    print(f"✓ File size: {output_size:.2f} KB")
    
    face_reduction = (1 - final_face_count / original_face_count) * 100
    size_reduction = (1 - output_size / input_size) * 100
    
    print(f"\n✓ Face reduction: {face_reduction:.1f}%")
    print(f"✓ Size reduction: {size_reduction:.1f}%")
    
    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)
    print(f"\nNext: poetry run mesh-painter \"{output_path}\" --visualize")


if __name__ == "__main__":
    main()
