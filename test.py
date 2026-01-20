import numpy as np
import trimesh

mesh = trimesh.load_mesh("test_part.stl")
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1.0 / mesh.scale)

V = mesh.vertices
path = []
for k in range (1000):
    path.append(trimesh.load_path(np.array([V[k], V[k+1]])))

scene = trimesh.Scene()
for line in path:
    scene.add_geometry(line)
scene.add_geometry(mesh)

scene.show()