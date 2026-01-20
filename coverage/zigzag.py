import numpy as np

def zigzag_on_region(mesh, face_ids, step=0.05):
    centers = mesh.triangles_center[face_ids]

    mean = centers.mean(axis=0)
    U, S, Vt = np.linalg.svd(centers - mean)

    u, v = Vt[:2]

    coords_2d = np.column_stack([
        (centers - mean) @ u,
        (centers - mean) @ v
    ])

    path = []
    y_vals = np.arange(coords_2d[:,1].min(),
                        coords_2d[:,1].max(),
                        step)

    for i, y in enumerate(y_vals):
        band = coords_2d[np.abs(coords_2d[:,1] - y) < step]

        band = sorted(band, key=lambda p: p[0], reverse=i % 2)

        for p in band:
            p3d = mean + p[0]*u + p[1]*v
            path.append(p3d)

    return path
