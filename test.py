import trimesh
import numpy as np

# =========================
# 1. Création / Chargement
# =========================

# Charger un maillage depuis un fichier
# mesh = trimesh.load("modele.stl")

# OU créer une primitive
mesh = trimesh.creation.icosphere(
    radius=1.0,
    subdivisions=3
)

# =========================
# 2. Informations générales
# =========================

print("Vertices :", len(mesh.vertices))
print("Faces :", len(mesh.faces))
print("Surface :", mesh.area)
print("Volume :", mesh.volume)
print("Watertight :", mesh.is_watertight)

# =========================
# 3. Transformations
# =========================

# Translation
mesh.apply_translation([1.0, 0.0, 0.0])

# Mise à l’échelle
mesh.apply_scale(0.5)

# Rotation (45° autour de l’axe Z)
angle = np.radians(45)
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle,
    direction=[0, 0, 1],
    point=mesh.centroid
)
mesh.apply_transform(rotation_matrix)

# =========================
# 4. Calculs géométriques
# =========================

# Distance d’un point au maillage
point = np.array([[0.0, 0.0, 0.0]])
distance = mesh.nearest.signed_distance(point)

print("Distance au point (0,0,0) :", distance[0])

# =========================
# 5. Opération booléenne
# =========================

cube = trimesh.creation.box(extents=(1, 1, 1))

# Soustraction sphère du cube
try:
    result = cube.difference(mesh)
    result.show()
except Exception as e:
    print("Booléen indisponible (blender/openscad requis)")

# =========================
# 6. Visualisation
# =========================

mesh.show()

# =========================
# 7. Export
# =========================

mesh.export("mesh_resultat.stl")
mesh.export("mesh_resultat.obj")
