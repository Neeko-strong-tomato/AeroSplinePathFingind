import trimesh
from coverage.segmentation import segment_by_normals
from coverage.region_analyse import analyze_region, local_frame, project_to_2d
from coverage.zigzag import discretize, zigzag_path, reproject_to_3d, visualize_path



# --------------------------------------------------
# 1. Chargement du mesh
# --------------------------------------------------
mesh = trimesh.load("3d_models/test_part.stl")

# --------------------------------------------------
# 2. Segmentation
# --------------------------------------------------
regions = segment_by_normals(mesh, angle_threshold_deg=30)

# --------------------------------------------------
# 3. Boucle sur les régions
# --------------------------------------------------
global_path = []

for region_id, region_faces in enumerate(regions):
    print(f"Région {region_id} | {len(region_faces)} faces")

    # Ignorer les petites régions
    if len(region_faces) < 5:
        continue

    # ------------------------------
    # Analyse région + repère local
    # ------------------------------
    info = analyze_region(mesh, region_faces)
    u, v, n = local_frame(info["vertices"], info["normal"])

    # ------------------------------
    # Projection 3D → 2D
    # ------------------------------
    points_2d = project_to_2d(info["vertices"], info["center"], u, v)

    # ------------------------------
    # Discrétisation
    # ------------------------------
    grid, min_xy, step = discretize(points_2d, step=0.02)  # 2cm


    # ------------------------------
    # Trajectoire zigzag
    # ------------------------------
    path_2d = zigzag_path(grid)

    # ------------------------------
    # Reprojection 2D → 3D
    # ------------------------------
    points_3d = reproject_to_3d(path_2d, min_xy, step, info["center"], u, v)

    # Ajouter au chemin global
    global_path.extend(points_3d)

    # ------------------------------
    # Visualisation région (optionnelle)
    # ------------------------------
    visualize_path(points_3d)

# --------------------------------------------------
# 4. Visualisation finale (optionnelle)
# --------------------------------------------------
print(f"\nPoints de trajectoire générés : {len(global_path)}")
