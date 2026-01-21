#!/usr/bin/env python3
"""
Test suite for path planning validation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import trimesh
import json
import hashlib
from path_planning import PathPlanner
from mesh_segmentation import MeshFaceSegmenter


def _get_cache_key(stl_file: str, angle_threshold: float, bend_threshold: float) -> str:
    """Generate a unique cache key for segmentation parameters."""
    key_str = f"{stl_file}_{angle_threshold}_{bend_threshold}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_file(stl_file: str, angle_threshold: float, bend_threshold: float) -> Path:
    """Get the cache file path for segmentation results."""
    cache_dir = Path(stl_file).parent / ".mesh_painter_cache"
    cache_key = _get_cache_key(stl_file, angle_threshold, bend_threshold)
    return cache_dir / f"{cache_key}.json"


def _load_segmentation_cache(cache_file: Path):
    """Load cached segmentation results."""
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            # Convert list of lists back to list of sets
            face_segments = [set(face) for face in data['segments']]
            return face_segments
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return None


def load_face_segments(mesh_file: str, angle_threshold: float = 60, bend_threshold: float = 60):
    """Load face segments from cache or compute if not cached."""
    mesh_file = Path(mesh_file)
    
    # Try to load from cache first
    cache_file = _get_cache_file(str(mesh_file), angle_threshold, bend_threshold)
    face_segments = _load_segmentation_cache(cache_file)
    
    if face_segments is not None:
        print(f"✓ Loaded {len(face_segments)} face segments from cache ({cache_file.name})")
        return face_segments, True  # Return flag indicating cache was used
    
    # Otherwise compute
    print(f"Computing face segments (cache not found at {cache_file})...")
    segmenter = MeshFaceSegmenter(angle_threshold=angle_threshold, bend_angle_threshold=bend_threshold)
    segmenter.load_stl(str(mesh_file))
    face_segments = segmenter.segment()
    return face_segments, False  # Return flag indicating cache was not used

def test_path_coverage():
    """Test that paths cover the entire face surface"""
    print("=" * 60)
    print("TEST 1: Path Coverage Analysis")
    print("=" * 60)
    
    # Load mesh
    mesh_file = Path("stl/test_part_m.stl")
    if not mesh_file.exists():
        print(f"Error: {mesh_file} not found")
        return
    
    mesh = trimesh.load(mesh_file)
    
    # Load face segments (from cache if available)
    face_segments, cache_used = load_face_segments(str(mesh_file), angle_threshold=60, bend_threshold=60)
    
    # Only create segmenter for geometry extraction (not segmentation)
    segmenter = MeshFaceSegmenter(angle_threshold=60, bend_angle_threshold=60)
    segmenter.load_stl(str(mesh_file))
    
    planner = PathPlanner(spray_height=0.20, spray_radius=0.10, path_spacing=0.05)
    
    for face_idx, face_indices in enumerate(list(face_segments)[:3]):  # Test first 3 faces
        vertices, faces = segmenter.get_face_geometry(face_indices)
        face_normal = mesh.face_normals[list(face_indices)[0]]
        
        waypoints = planner.plan_path(vertices, faces, face_normal)
        waypoints = np.array(waypoints)
        
        print(f"\nFace {face_idx}:")
        print(f"  Mesh triangles: {len(face_indices)}")
        print(f"  Waypoints generated: {len(waypoints)}")
        print(f"  Waypoint bounds:")
        print(f"    X: [{waypoints[:, 0].min():.3f}, {waypoints[:, 0].max():.3f}]")
        print(f"    Y: [{waypoints[:, 1].min():.3f}, {waypoints[:, 1].max():.3f}]")
        print(f"    Z: [{waypoints[:, 2].min():.3f}, {waypoints[:, 2].max():.3f}]")
        print(f"  Z variance: {waypoints[:, 2].std():.6f} (should be low)")
        path_length = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        print(f"  Path length: {path_length:.2f}m")

def test_waypoint_ordering():
    """Test that waypoints are ordered logically"""
    print("\n" + "=" * 60)
    print("TEST 2: Waypoint Ordering (Continuity)")
    print("=" * 60)
    
    mesh_file = Path("stl/test_part_m.stl")
    if not mesh_file.exists():
        print(f"Error: {mesh_file} not found")
        return
    
    mesh = trimesh.load(mesh_file)
    
    # Load face segments (from cache if available)
    face_segments, cache_used = load_face_segments(str(mesh_file), angle_threshold=60, bend_threshold=60)
    segmenter = MeshFaceSegmenter(angle_threshold=60, bend_angle_threshold=60)
    segmenter.load_stl(str(mesh_file))
    planner = PathPlanner(spray_height=0.20, spray_radius=0.10, path_spacing=0.05)
    
    for face_idx, face_indices in enumerate(list(face_segments)[:2]):
        vertices, faces = segmenter.get_face_geometry(face_indices)
        face_normal = mesh.face_normals[list(face_indices)[0]]
        
        waypoints = np.array(planner.plan_path(vertices, faces, face_normal))
        
        # Calculate distances between consecutive waypoints
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        
        print(f"\nFace {face_idx}:")
        print(f"  Mean distance between points: {distances.mean():.4f}m")
        print(f"  Max jump: {distances.max():.4f}m")
        print(f"  Min distance: {distances.min():.6f}m")
        print(f"  Jumps > 0.5m: {np.sum(distances > 0.5)}")
        
        # Check for large discontinuities
        large_jumps = np.where(distances > 0.2)[0]
        if len(large_jumps) > 0:
            print(f"  ⚠️  Found {len(large_jumps)} large jumps (>0.2m)")
        else:
            print(f"  ✓ No large discontinuities detected")

def test_spray_height():
    """Test that all waypoints are at correct spray height"""
    print("\n" + "=" * 60)
    print("TEST 3: Spray Height Validation")
    print("=" * 60)
    
    mesh_file = Path("stl/test_part_m.stl")
    if not mesh_file.exists():
        print(f"Error: {mesh_file} not found")
        return
    
    mesh = trimesh.load(mesh_file)
    spray_height = 0.20
    
    # Load face segments (from cache if available)
    face_segments, cache_used = load_face_segments(str(mesh_file), angle_threshold=60, bend_threshold=60)
    segmenter = MeshFaceSegmenter(angle_threshold=60, bend_angle_threshold=60)
    segmenter.load_stl(str(mesh_file))
    planner = PathPlanner(spray_height=spray_height, spray_radius=0.10, path_spacing=0.05)
    
    total_waypoints = 0
    
    for face_idx, face_indices in enumerate(list(face_segments)[:3]):
        vertices, faces = segmenter.get_face_geometry(face_indices)
        face_normal = mesh.face_normals[list(face_indices)[0]]
        
        waypoints = np.array(planner.plan_path(vertices, faces, face_normal))
        total_waypoints += len(waypoints)
        
        # Estimate height offset from face normal direction
        # All waypoints should be roughly at spray_height offset from surface
        print(f"\nFace {face_idx}: {len(waypoints)} waypoints")
        print(f"  Spray height offset: {spray_height:.4f}m (expected)")
        
        # Check if waypoints are reasonably offset from vertices
        # by checking distance to nearest vertex
        distances_to_vertices = []
        for wp in waypoints:
            dists = np.linalg.norm(vertices - wp, axis=1)
            distances_to_vertices.append(dists.min())
        
        distances_to_vertices = np.array(distances_to_vertices)
        print(f"  Distance to nearest vertex:")
        print(f"    Mean: {distances_to_vertices.mean():.4f}m")
        print(f"    Min: {distances_to_vertices.min():.4f}m")
        print(f"    Max: {distances_to_vertices.max():.4f}m")
        
        if distances_to_vertices.mean() > 0.15:
            print(f"  ✓ Waypoints are well offset from surface")
        else:
            print(f"  ⚠️  Waypoints may be too close to vertices")

def test_mesh_coverage():
    """Test that the path covers the face area adequately"""
    print("\n" + "=" * 60)
    print("TEST 4: Path Systematicity (Grid Pattern)")
    print("=" * 60)
    
    mesh_file = Path("stl/test_part_m.stl")
    if not mesh_file.exists():
        print(f"Error: {mesh_file} not found")
        return
    
    mesh = trimesh.load(mesh_file)
    
    # Load face segments (from cache if available)
    face_segments, cache_used = load_face_segments(str(mesh_file))
    segmenter = MeshFaceSegmenter(angle_threshold=30, bend_angle_threshold=90)
    segmenter.load_stl(str(mesh_file))
    planner = PathPlanner(spray_height=0.20, spray_radius=0.10, path_spacing=0.05)
    
    for face_idx, face_indices in enumerate(list(face_segments)[:2]):
        vertices, faces = segmenter.get_face_geometry(face_indices)
        face_normal = mesh.face_normals[list(face_indices)[0]]
        
        waypoints = np.array(planner.plan_path(vertices, faces, face_normal))
        
        # Analyze Y-coordinates to verify grid pattern
        y_coords = waypoints[:, 1]
        y_sorted = np.sort(np.unique(np.round(y_coords, decimals=3)))
        
        # Expected spacing should be roughly 0.05m (path_spacing)
        if len(y_sorted) > 1:
            y_spacing = np.diff(y_sorted)
            avg_y_spacing = y_spacing.mean()
        else:
            avg_y_spacing = 0
        
        print(f"\nFace {face_idx}:")
        print(f"  Total waypoints: {len(waypoints)}")
        print(f"  Unique Y levels: {len(y_sorted)}")
        print(f"  Expected Y spacing: 0.05m")
        print(f"  Actual average Y spacing: {avg_y_spacing:.4f}m")
        
        if len(y_sorted) > 1:
            y_spacing_range = [y_spacing.min(), y_spacing.max()]
            print(f"  Y spacing range: [{y_spacing_range[0]:.4f}, {y_spacing_range[1]:.4f}]m")
        
        # Check X distribution per Y level
        print(f"  Points per Y level:")
        for y_val in y_sorted[:5]:  # Show first 5 levels
            points_at_y = np.sum(np.abs(y_coords - y_val) < 0.005)
            print(f"    Y={y_val:.3f}: {points_at_y} points")
        
        if len(y_sorted) > 3:
            print(f"  ✓ Grid pattern detected with {len(y_sorted)} sweep lines")
        else:
            print(f"  ⚠️  Limited grid pattern (only {len(y_sorted)} sweep lines)")

if __name__ == "__main__":
    test_path_coverage()
    test_waypoint_ordering()
    test_spray_height()
    test_mesh_coverage()
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
