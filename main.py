import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from trimesh.path.entities import Line
from uv_map import create_point_cloud, calculate_path, scaling_transform
from segmentation import segmentation

meshes = segmentation("./3d_models/surface.stl")
paths_points = []
path_entities = []
for mesh in meshes:
    mesh = mesh.unwrap()
    points = create_point_cloud(mesh,resolution=32)

    # distance from point to surface of meshdistances
    # create a PointCloud object out of each (n,3) list of points
    cloud = trimesh.points.PointCloud(points[0])

    # Convertir faces → positions 3D
    (path_points,jump_list) = calculate_path(mesh,points[0],points[1])

    # Visualisation
    #path = trimesh.load_path(path_points)
    #path.colors = [trimesh.visual.random_color()]

    for i in range(len(jump_list)-1):
        paths_points += path_points
        path_entities.append(Line(points=np.arange(jump_list[i]+1,jump_list[i+1])))

path = trimesh.path.Path3D(
vertices=paths_points,
entities=path_entities,
colors = [trimesh.visual.random_color() for e in path_entities]
)

# create a scene containing the mesh and two sets of points
scene = trimesh.Scene([meshes,path])

# show the scene we are using
scene.show()
