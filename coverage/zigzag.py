import numpy as np
from mesh.mesh_utils import face_centers, face_normals


def zigzag_coverage(mesh):
    """
    Génère une trajectoire zigzag simple sur un sous-mesh
    Retourne une liste d'indices de faces
    """

    centers = face_centers(mesh)
    normals = face_normals(mesh)

    # Normale moyenne
    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal) + 1e-12

    # Base locale
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, mean_normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    tangent1 = np.cross(mean_normal, ref)
    tangent1 /= np.linalg.norm(tangent1) + 1e-12
    tangent2 = np.cross(mean_normal, tangent1)

    # Projection
    proj = np.stack([
        centers @ tangent1,
        centers @ tangent2
    ], axis=1)

    # Tri zigzag
    order = np.argsort(proj[:, 1])
    rows = np.array_split(order, int(len(order) ** 0.5) + 1)

    path = []
    for i, row in enumerate(rows):
        row_sorted = row[np.argsort(proj[row, 0])]
        if i % 2 == 1:
            row_sorted = row_sorted[::-1]
        path.extend(row_sorted.tolist())

    return path
