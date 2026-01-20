#!/usr/bin/env python3
"""
Main entry point for mesh painter application.

Loads STL file, segments it into faces, and generates optimal painting paths.
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mesh_segmentation import MeshFaceSegmenter
from path_planning import PathPlanner, PathVisualizer
from visualization import MeshVisualizer


def _get_cache_key(stl_file: str, angle_threshold: float, bend_threshold: float) -> str:
    """Generate a unique cache key for segmentation parameters."""
    key_str = f"{stl_file}_{angle_threshold}_{bend_threshold}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_file(stl_file: str, angle_threshold: float, bend_threshold: float) -> Path:
    """Get the cache file path for segmentation results."""
    cache_dir = Path(stl_file).parent / ".mesh_painter_cache"
    cache_key = _get_cache_key(stl_file, angle_threshold, bend_threshold)
    return cache_dir / f"{cache_key}.json"


def _load_segmentation_cache(cache_file: Path) -> tuple:
    """Load cached segmentation results. Returns (face_segments, file_hash) or (None, None)."""
    if not cache_file.exists():
        return None, None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            # Convert list of lists back to list of sets
            face_segments = [set(face) for face in data['segments']]
            return face_segments, data['file_hash']
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return None, None


def _save_segmentation_cache(cache_file: Path, face_segments: list, file_hash: str) -> None:
    """Save segmentation results to cache."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data = {
            'segments': [list(face) for face in face_segments],
            'file_hash': file_hash
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def _get_file_hash(file_path: str) -> str:
    """Compute hash of file to detect changes."""
    hash_obj = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Paint objects mapped as 3D meshes with robotic arm"
    )
    parser.add_argument(
        "stl_file",
        type=str,
        help="Path to STL file to process"
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Angle threshold for face segmentation (degrees, default: 30)"
    )
    parser.add_argument(
        "--bend-threshold",
        type=float,
        default=90.0,
        help="Bend angle threshold for face segmentation (degrees, default: 90)"
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
        "--output",
        type=str,
        default=None,
        help="Output CSV file for waypoints (default: stl_basename_waypoints.csv)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comprehensive visualizations for all steps"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations (default: visualizations)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached segmentation, recalculate from scratch"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.stl_file).exists():
        print(f"Error: STL file not found: {args.stl_file}", file=sys.stderr)
        sys.exit(1)
    
    # Segment mesh
    print("\n" + "="*60)
    print("MESH SEGMENTATION")
    print("="*60)
    
    # Try to load from cache
    cache_file = _get_cache_file(args.stl_file, args.angle_threshold, args.bend_threshold)
    file_hash = _get_file_hash(args.stl_file)
    face_segments = None
    
    if not args.no_cache:
        cached_segments, cached_hash = _load_segmentation_cache(cache_file)
        if cached_segments is not None and cached_hash == file_hash:
            print(f"Loading cached segmentation from: {cache_file}")
            face_segments = cached_segments
    
    # If not cached, compute segmentation
    if face_segments is None:
        segmenter = MeshFaceSegmenter(
            angle_threshold=args.angle_threshold,
            bend_angle_threshold=args.bend_threshold
        )
        mesh = segmenter.load_stl(args.stl_file)
        face_segments = segmenter.segment()
        
        # Save to cache
        _save_segmentation_cache(cache_file, face_segments, file_hash)
    else:
        # Still need to load mesh for later steps
        segmenter = MeshFaceSegmenter(
            angle_threshold=args.angle_threshold,
            bend_angle_threshold=args.bend_threshold
        )
        mesh = segmenter.load_stl(args.stl_file)
    
    print(f"\n✓ Segmentation complete: {len(face_segments)} faces found")
    for i, face in enumerate(face_segments):
        print(f"  Face {i}: {len(face)} mesh triangles")
    
    # Plan paths
    print("\n" + "="*60)
    print("PATH PLANNING")
    print("="*60)
    planner = PathPlanner(
        spray_height=args.spray_height,
        spray_radius=args.spray_radius,
        path_spacing=args.path_spacing
    )
    
    waypoints_list = []
    for i, face_indices in enumerate(tqdm(face_segments, desc="Planning paths", unit="face")):
        vertices, faces = segmenter.get_face_geometry(face_indices)
        face_normal = mesh.face_normals[list(face_indices)[0]]
        
        waypoints = planner.plan_path(vertices, faces, face_normal)
        waypoints_list.append(waypoints)
    
    print(f"✓ Path planning complete")
    
    # Optimize overall path
    print("\n" + "="*60)
    print("PATH OPTIMIZATION")
    print("="*60)
    print("Optimizing overall path order...")
    optimized_waypoints = planner.optimize_waypoints(waypoints_list)
    print(f"✓ Total waypoints: {len(optimized_waypoints)}")
    
    # Export waypoints
    output_file = args.output
    if output_file is None:
        stl_name = Path(args.stl_file).stem
        output_file = f"{stl_name}_waypoints.csv"
    
    print("\n" + "="*60)
    print("EXPORT")
    print("="*60)
    print(f"Exporting waypoints to: {output_file}")
    PathVisualizer.export_waypoints_to_csv(optimized_waypoints, output_file)
    print("✓ Waypoints exported")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\n" + "="*60)
        print("VISUALIZATION")
        print("="*60)
        visualizer = MeshVisualizer(output_dir=args.viz_dir)
        visualizer.generate_report(
            mesh, face_segments, waypoints_list, optimized_waypoints
        )
    
    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
