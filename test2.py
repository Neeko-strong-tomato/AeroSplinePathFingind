import numpy as np
from PIL import Image
from uv_map import uv_to_3d, uv_to_face_id
import trimesh
# load a large- ish PLY model with colors
mesh = trimesh.load("./3d_models/dome.stl")

# Load texture image
image = Image.open("./UVmap.jpg")
image = image.convert("RGBA")
#image_array = np.array(image)

mesh = mesh.unwrap(image)

# we can sample the volume of Box primitives
resolution = 64
total_points = resolution * resolution
uv_points = []
for i in range(resolution):
    for j in range(resolution):
        uv_points.append(np.array([0.5 - float(i)/float(resolution),0.5 - float(j)/float(resolution)]))
        print(np.array([0.5 - float(i)/float(resolution),0.5 - float(j)/float(resolution)]))

ref_points = []
ref_face_ids = []


for p in uv_points:
    face_id = uv_to_face_id(mesh,p)
    if(face_id != None):
        ref_points.append(uv_to_3d(mesh,p))
        ref_face_ids.append(ref_face_ids)
points = np.array(np.array(ref_points),np.array(ref_face_ids))

# distance from point to surface of meshdistances
# create a PointCloud object out of each (n,3) list of points
cloud_close = trimesh.points.PointCloud(points[0])

# create a unique color for each point
cloud_colors = np.array([trimesh.visual.random_color() for i in points])

# set the colors on the random point and its nearest point to be the same
cloud_close.vertices_color = cloud_colors

# create a scene containing the mesh and two sets of points
scene = trimesh.Scene([mesh, cloud_close])

# show the scene we are using
scene.show()