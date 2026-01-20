# Mesh Painter

Paint objects mapped as 3D meshes with a robotic arm. This project focuses on dividing STL meshes into faces and creating optimal painting paths for the arm to cover.

## Features

- **Mesh Segmentation**: Divides STL models into paintable faces based on:
  - 60° angle threshold between adjacent face normals
  - 60° overall bend angle threshold for face regions
  
- **Path Planning**: Generates efficient coverage paths:
  - Snake/raster pattern for minimal travel distance
  - 20cm spray head height from surface
  - 20cm spray radius coverage

- **Waypoint Generation**: Exports optimized waypoint lists for robotic arm control

## Installation

```bash
cd mesh_painter
pip install -r requirements.txt
```

## Usage

### Basic usage
```bash
python main.py model.stl
```

### Advanced options
```bash
python main.py model.stl \
    --angle-threshold 60 \
    --bend-threshold 60 \
    --spray-height 0.20 \
    --spray-radius 0.10 \
    --path-spacing 0.05 \
    --output waypoints.csv \
    --visualize
```

### Options
- `--angle-threshold`: Angle threshold for face segmentation (degrees)
- `--bend-threshold`: Overall bend angle threshold (degrees)
- `--spray-height`: Height of spray head above surface (meters)
- `--spray-radius`: Radius of spray pattern (meters)
- `--path-spacing`: Spacing between parallel path lines (meters)
- `--output`: Output CSV file for waypoints
- `--visualize`: Display 3D visualization of mesh and paths

## Output

The tool generates a CSV file containing waypoints with columns:
- X, Y, Z: 3D coordinates of each waypoint

These waypoints represent:
1. Start/end points of each spray line
2. Offset by spray height above the mesh surface
3. Ordered to minimize travel distance between faces

## Project Structure

```
mesh_painter/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── src/
    ├── __init__.py
    ├── mesh_segmentation.py    # Face segmentation algorithm
    └── path_planning.py        # Coverage path planning
```

## Algorithm Details

### Mesh Segmentation
1. Load STL and compute face normals
2. Build face adjacency graph from shared edges
3. Region growing: merge adjacent faces if:
   - Angle between normals < 60°
   - Overall region bend angle < 60°

### Path Planning
1. For each face, compute local coordinate system
2. Project face onto 2D plane
3. Generate parallel raster lines with specified spacing
4. Convert back to 3D with spray height offset
5. Optimize face order using greedy nearest-neighbor

## Future Enhancements

- [ ] Improved bend angle calculation using geodesic distances
- [ ] Collision detection with workspace
- [ ] Support for joint constraints
- [ ] Adaptive path spacing based on surface curvature
- [ ] Support for different spray patterns (spiral, zigzag)
- [ ] Integration with robotic arm control system
