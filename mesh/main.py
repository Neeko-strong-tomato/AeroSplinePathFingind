##############################################
## Exemple d'utilisation de la segmentation 
###############################################
from segmentation import (
    segmentation,
    show_segmentation,
    export_segmentation,
)

mesh, labels, nseg = segmentation("fan.stl", angle_deg=35)
print("FAN - Segments:", nseg)
print("FAN - Faces:", len(mesh.faces))

show_segmentation(mesh, labels, title="fan - 30deg")

exported = export_segmentation (
    mesh,
    labels,
    out_dir="out_fan_segments",
    prefix="fan_seg",
    min_faces=20,          
    reindex_vertices=True,
    binary=True
)
print("FAN - Nb STL exportés:", len(exported))




mesh, labels, nseg = segmentation("dome.stl", angle_deg=30)
print("DOME - Segments:", nseg)
print("DOME - Faces:", len(mesh.faces))

show_segmentation(mesh, labels, title="dome - 30deg")
exported = export_segmentation(
    mesh,
    labels,
    out_dir="out_dome_segments",
    prefix="dome_seg",
    min_faces=20,
    reindex_vertices=True,
    binary=True
)
print("DOME - Nb STL exportés:", len(exported))




mesh, labels, nseg = segmentation("cube.stl", angle_deg=30)
print("CUBE - Segments:", nseg)
print("CUBE - Faces:", len(mesh.faces))

show_segmentation(mesh, labels, title="cube - 30deg")
exported = export_segmentation(
    mesh,
    labels,
    out_dir="out_cube_segments",
    prefix="cube_seg",
    min_faces=1,
    reindex_vertices=True,
    binary=True
)
print("CUBE - Nb STL exportés:", len(exported))
