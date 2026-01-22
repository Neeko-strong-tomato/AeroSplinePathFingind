import numpy as np

def face_centers(mesh):
    return mesh.triangles_center

def face_normals(mesh):
    n = mesh.face_normals
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

def face_neighbors(mesh):
    adj = [[] for _ in range(len(mesh.faces))]
    for i, j in mesh.face_adjacency:
        adj[i].append(j)
        adj[j].append(i)
    return adj
