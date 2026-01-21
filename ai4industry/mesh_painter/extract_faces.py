#!/usr/bin/env python3
"""
Extract individual segmented faces from a mesh and save them as separate STL files.

This allows selective path planning optimization on individual faces.
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import trimesh
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


def _get_file_hash(file_path: str) -> str:
    """Compute hash of file to detect changes."""
    hash_obj = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def extract_faces(stl_file: str, output_dir: str = "extracted_faces", 
                 angle_threshold: float = 60, bend_threshold: float = 60,
                 no_cache: bool = False):
    """Extract individual faces and save as separate STL files."""
    
    stl_path = Path(stl_file)
    if not stl_path.exists():
        print(f"Error: STL file not found: {stl_file}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("EXTRACTING INDIVIDUAL FACES")
    print("="*60)
    
    # Load mesh
    mesh = trimesh.load(stl_file)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Load segmentation (from cache if available)
    cache_file = _get_cache_file(stl_file, angle_threshold, bend_threshold)
    file_hash = _get_file_hash(stl_file)
    face_segments = None
    
    if not no_cache:
        cached_segments, cached_hash = _load_segmentation_cache(cache_file)
        if cached_segments is not None and cached_hash == file_hash:
            print(f"Loading cached segmentation from cache")
            face_segments = cached_segments
    
    # If not cached, compute segmentation
    if face_segments is None:
        print("Computing segmentation...")
        segmenter = MeshFaceSegmenter(
            angle_threshold=angle_threshold,
            bend_angle_threshold=bend_threshold
        )
        segmenter.load_stl(stl_file)
        face_segments = segmenter.segment()
    
    print(f"\n✓ Segmentation complete: {len(face_segments)} faces found\n")
    
    # Extract and save each face
    print("Extracting faces...")
    for face_idx, face_indices in enumerate(tqdm(face_segments, desc="Extracting", unit="face")):
        # Create submesh for this face
        face_indices_list = list(face_indices)
        face_mesh = mesh.submesh([face_indices_list], append=True)
        
        # Save as STL
        face_file = output_path / f"face_{face_idx:03d}.stl"
        face_mesh.export(str(face_file))
        
        print(f"  Face {face_idx}: {len(face_indices_list)} triangles → {face_file.name}")
    
    # Create a metadata file with face information
    metadata = {
        "source_stl": str(stl_path),
        "angle_threshold": angle_threshold,
        "bend_threshold": bend_threshold,
        "num_faces": len(face_segments),
        "faces": {
            f"face_{i:03d}": {
                "index": i,
                "triangle_count": len(list(indices))
            }
            for i, indices in enumerate(face_segments)
        }
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Extracted {len(face_segments)} faces to {output_path}")
    print(f"✓ Metadata saved to {metadata_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual segmented faces from a mesh as separate STL files"
    )
    parser.add_argument(
        "stl_file",
        type=str,
        help="Path to STL file to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_faces",
        help="Directory to save extracted faces (default: extracted_faces)"
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=60.0,
        help="Angle threshold for face segmentation (degrees, default: 60)"
    )
    parser.add_argument(
        "--bend-threshold",
        type=float,
        default=60.0,
        help="Bend angle threshold for face segmentation (degrees, default: 60)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached segmentation, recalculate from scratch"
    )
    
    args = parser.parse_args()
    
    extract_faces(
        args.stl_file,
        output_dir=args.output_dir,
        angle_threshold=args.angle_threshold,
        bend_threshold=args.bend_threshold,
        no_cache=args.no_cache
    )


if __name__ == "__main__":
    main()
