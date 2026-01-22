import os
import numpy as np
import trimesh
from collections import deque
import pyvista as pv


def segmentation(mesh_path: str, angle_deg: float = 30.0):
    """
    Segmentation conforme + plus utile:
    - split des pièces (fan)
    - voisins = face_adjacency
    - critère: une face peut entrer dans un segment si son angle avec la NORMALE MOYENNE du segment <= angle_deg
    => coupe naturellement les surfaces courbes (corps), pas seulement les cassures.
    """
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    if mesh.is_empty or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh vide ou sans faces.")

    mesh.merge_vertices()

    parts = mesh.split(only_watertight=False)

    cos_thr = float(np.cos(np.deg2rad(angle_deg)))

    all_meshes = []
    all_labels = []
    label_offset = 0

    for part in parts:
        if part.is_empty or len(part.faces) == 0:
            continue

        n_faces = len(part.faces)
        edges = getattr(part, "face_adjacency", None)

        # normales unitaires
        fn = part.face_normals.astype(np.float64)
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

        if edges is None or len(edges) == 0 or n_faces < 2:
            labels = np.zeros(n_faces, dtype=np.int32) + label_offset
            all_meshes.append(part)
            all_labels.append(labels)
            label_offset += 1
            continue

        adj = [[] for _ in range(n_faces)]
        for i, j in edges:
            i = int(i); j = int(j)
            adj[i].append(j)
            adj[j].append(i)

        labels = -np.ones(n_faces, dtype=np.int32)
        seg_id = 0

        for seed in range(n_faces):
            if labels[seed] != -1:
                continue

            # démarre une région
            labels[seed] = seg_id
            q = deque([seed])

            # normale moyenne du segment (mise à jour incrémentale)
            mean = fn[seed].copy()
            mean /= (np.linalg.norm(mean) + 1e-12)

            count = 1

            while q:
                u = q.popleft()
                for v in adj[u]:
                    if labels[v] != -1:
                        continue

                    if float(np.dot(fn[v], mean)) >= cos_thr:
                        labels[v] = seg_id
                        q.append(v)

                        mean = mean * count + fn[v]
                        mean /= (np.linalg.norm(mean) + 1e-12)
                        count += 1

            seg_id += 1

        labels = labels + label_offset
        label_offset += seg_id

        all_meshes.append(part)
        all_labels.append(labels)

    merged = trimesh.util.concatenate(all_meshes)
    merged_labels = np.concatenate(all_labels)
    return merged, merged_labels, int(merged_labels.max() + 1)


def show_segmentation(mesh: trimesh.Trimesh, face_labels: np.ndarray, title="Segmentation"):

    faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64),
                       mesh.faces.astype(np.int64)]).ravel()
    poly = pv.PolyData(mesh.vertices.astype(np.float64), faces)

    nseg = int(face_labels.max() + 1)
    rng = np.random.default_rng(0)
    colors = rng.integers(30, 255, size=(nseg, 3), dtype=np.uint8)

    rgb = colors[face_labels]  # (n_faces, 3)
    poly.cell_data["rgb"] = rgb
    poly.cell_data["rgb"] = poly.cell_data["rgb"].astype(np.uint8)

    pl = pv.Plotter(title=title)
    pl.add_mesh(poly, scalars="rgb", rgb=True, show_edges=False)
    pl.add_axes()
    pl.show()


def export_segmentation(mesh: trimesh.Trimesh,
                                  face_labels: np.ndarray,
                                  out_dir: str,
                                  prefix: str = "segment",
                                  min_faces: int = 1,
                                  reindex_vertices: bool = True,
                                  binary: bool = True):

    if mesh.is_empty or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh vide ou sans faces.")
    if face_labels is None or len(face_labels) != len(mesh.faces):
        raise ValueError("face_labels doit avoir la même longueur que mesh.faces.")

    os.makedirs(out_dir, exist_ok=True)

    labels = np.asarray(face_labels, dtype=np.int64)
    uniq = np.unique(labels)

    exported = {}
    stl_encoding = "binary" if binary else "ascii"

    for lab in uniq:
        face_idx = np.where(labels == lab)[0]
        if face_idx.size < min_faces:
            continue

        # Sous-maillage
        sub = mesh.submesh([face_idx], append=True, repair=False)

        # submesh peut renvoyer une liste selon la version; append=True renvoie normalement un Trimesh.
        if isinstance(sub, (list, tuple)):
            sub = sub[0] if len(sub) > 0 else None

        if sub is None or sub.is_empty or len(sub.faces) == 0:
            continue

        # Nettoyage / compaction
        if reindex_vertices:
            sub.remove_unreferenced_vertices()

        # Nom de fichier
        fname = f"{prefix}_{int(lab):03d}.stl"
        path = os.path.join(out_dir, fname)

        # Export STL
        sub.export(path, file_type="stl")
        exported[int(lab)] = path

    return exported


def export_segments_for_rl(mesh_path, out_dir, angle_deg=30, min_faces=20):
    mesh, labels, nseg = segmentation(mesh_path, angle_deg)
    exported = export_segmentation(
        mesh,
        labels,
        out_dir=out_dir,
        prefix="seg",
        min_faces=min_faces,
        reindex_vertices=True,
        binary=True
    )
    return list(exported.values())
