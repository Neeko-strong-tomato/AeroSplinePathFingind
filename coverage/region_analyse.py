def analyze_region(mesh, region):
    """
    Retourne :
    - centre de la région
    - normale moyenne
    - aire totale
    """
    faces = mesh.faces[region]
    vertices = mesh.vertices[faces].reshape(-1, 3)

    center = vertices.mean(axis=0)
    normals = mesh.face_normals[region]
    n_mean = np.mean(normals, axis=0)
    n_mean /= np.linalg.norm(n_mean)
    
    area = mesh.area_faces[region].sum()
    return {
        "center": center,
        "normal": n_mean,
        "area": area,
        "vertices": vertices
    }


import numpy as np

def local_frame(vertices, normal):
    """
    Renvoie le repère (u,v,n) pour la région.
    """
    verts_centered = vertices - vertices.mean(axis=0)
    cov = np.cov(verts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    u = eigvecs[:, np.argmax(eigvals)]
    if abs(np.dot(u, normal)) > 0.9:  # éviter u proche de n
        u = eigvecs[:, np.argmin(eigvals)]

    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v, normal


def project_to_2d(vertices, center, u, v):
    """
    Transforme les sommets en coordonnées 2D locales sur le plan tangent.
    """
    rel = vertices - center
    x = rel @ u
    y = rel @ v
    return np.stack([x, y], axis=1)
