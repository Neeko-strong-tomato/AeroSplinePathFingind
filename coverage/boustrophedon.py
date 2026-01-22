import numpy as np
from mesh.mesh_utils import face_centers


def boustrophedon_coverage(mesh):
    """
    Variante simple du boustrophédon
    """
    centers = face_centers(mesh)

    # Projection XY
    proj = centers[:, :2]

    order = np.argsort(proj[:, 1])
    path = []

    current_dir = 1
    last_y = None
    row = []

    for idx in order:
        y = proj[idx, 1]
        if last_y is None or abs(y - last_y) < 1e-3:
            row.append(idx)
        else:
            row = sorted(row, key=lambda i: proj[i, 0], reverse=(current_dir < 0))
            path.extend(row)
            row = [idx]
            current_dir *= -1
        last_y = y

    if row:
        row = sorted(row, key=lambda i: proj[i, 0], reverse=(current_dir < 0))
        path.extend(row)

    return path
