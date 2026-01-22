import trimesh

def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if mesh.is_empty or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh invalide : {path}")
    mesh.merge_vertices()
    return mesh
