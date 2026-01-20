import numpy as np
from collections import deque

def angle_ok(n1, n2, cos_th):
    return np.dot(n1, n2) > cos_th


def segment_by_normals(mesh, angle_threshold_deg=30):
    normals = mesh.face_normals
    adjacency = mesh.face_adjacency  # (N, 2)
    cos_th = np.cos(np.radians(angle_threshold_deg))

    # Construire graphe de voisinage
    neighbors = [[] for _ in range(len(normals))]
    for f1, f2 in adjacency:
        neighbors[f1].append(f2)
        neighbors[f2].append(f1)

    visited = np.zeros(len(normals), dtype=bool)
    regions = []

    for start_face in range(len(normals)):
        if visited[start_face]:
            continue

        region = []
        queue = deque([start_face])
        visited[start_face] = True

        while queue:
            f = queue.popleft()
            region.append(f)

            for nb in neighbors[f]:
                if visited[nb]:
                    continue
                if angle_ok(normals[f], normals[nb], cos_th):
                    visited[nb] = True
                    queue.append(nb)

        regions.append(region)

    print(f"✅ {len(regions)} régions segmentées")
    return regions
