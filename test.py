import numpy as np
import trimesh
from trimesh.viewer import SceneViewer
import time


MESH_PATH= "test_part.stl"
MIN_DISTANCE_POINTS = 0.001

def load_mesh(path):
    mesh = trimesh.load_mesh(path)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)
    mesh.visual.face_colors = [0, 0, 0, 100]
    return mesh

def create_path(V):
    path = []
    covered = []
    available_point = [True]*len(V)
    i_path = [0,1]
    for k in range(1000):
        print(f"{k}/1000")
        norm_min = 100
        for i in range(1,len(V)):
            if available_point[i] :
                diff = V[i_path[-1]]-V[i]
                norm_i = diff[0]**2 + diff[1]**2 + diff[2]**2
                if(norm_i<=MIN_DISTANCE_POINTS):
                    covered.append(i)
                    available_point[i] = False
                elif(norm_i<norm_min):
                    norm_min = norm_i
                    i_min = i
        i_path.append(i_min)
        available_point[i_min] =False
        path.append(trimesh.load_path(np.array([V[i_path[-2]], V[i_path[-1]]])))
    
    return path

def print_scene(mesh):
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    V = mesh.vertices
    path = create_path(V)
    for line in path:
        scene.add_geometry(line)
    viewer = SceneViewer(scene)
    return viewer, scene

if __name__ == "__main__":
    mesh = load_mesh(MESH_PATH)
    viewer, scene = print_scene(mesh)

    try:
        None
    except KeyboardInterrupt:
        viewer.close()