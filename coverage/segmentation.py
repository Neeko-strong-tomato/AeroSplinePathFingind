import numpy as np

def segment_by_normals(mesh, angle_threshold_deg=30):
    normals = mesh.face_normals
    cos_th = np.cos(np.radians(angle_threshold_deg))

    used = set()
    regions = []

    for i, n in enumerate(normals):
        if i in used:
            continue

        region = [i]
        used.add(i)

        for j, m in enumerate(normals):
            if j in used:
                continue
            if np.dot(n, m) > cos_th:
                region.append(j)
                used.add(j)

        regions.append(region)

    print(f" {len(regions)} régions détectées")
    return regions
