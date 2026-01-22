import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from trimesh.path.entities import Line

def raycast_uv(mesh: trimesh.Trimesh,
               ray_origin: np.ndarray,
               ray_direction: np.ndarray):
    """
    Perform a raycast on a mesh and return the UV coordinate of the intersection.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh with UV coordinates
    ray_origin : (3,) array
        Ray origin in world space
    ray_direction : (3,) array
        Ray direction (does not need to be normalized)

    Returns
    -------
    uv : (2,) array or None
        UV coordinate at intersection, or None if no hit
    hit_point : (3,) array or None
        3D intersection point
    face_index : int or None
        Index of intersected face
    """

    # Normalize direction
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Perform raycast
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction],
        multiple_hits=False
    )

    if len(locations) == 0:
        return None, None, None

    hit_point = locations[0]
    face_index = index_tri[0]

    # Get triangle vertices
    triangle_vertices = mesh.vertices[mesh.faces[face_index]]

    # Compute barycentric coordinates
    bary = trimesh.triangles.points_to_barycentric(
        triangle_vertices[None, :, :],
        hit_point[None, :]
    )[0]

    # Get triangle UVs
    if mesh.visual.uv is None:
        raise ValueError("Mesh does not contain UV coordinates")

    triangle_uvs = mesh.visual.uv[mesh.faces[face_index]]

    # Interpolate UV
    uv = (
        triangle_uvs[0] * bary[0] +
        triangle_uvs[1] * bary[1] +
        triangle_uvs[2] * bary[2]
    )

    return uv, hit_point, face_index


def build_uv_acceleration(mesh):
    """
    Precompute a UV-space acceleration structure.
    """

    uv_faces = mesh.visual.uv[mesh.faces]

    polygons = []
    face_ids = []

    for i, tri_uv in enumerate(uv_faces):
        poly = Polygon(tri_uv)
        if poly.is_valid and poly.area > 0:
            polygons.append(poly)
            face_ids.append(i)

    tree = STRtree(polygons)

    return {
        "tree": tree,
        "polygons": polygons,
        "face_ids": face_ids
    }

def uv_to_3d_fast(mesh, uv, uv_accel):
    """
    Fast UV to 3D lookup using spatial indexing (Shapely 2.x compatible).
    """

    uv = np.asarray(uv, dtype=np.float64)
    point = Point(uv)

    tree = uv_accel["tree"]
    polygons = uv_accel["polygons"]
    face_ids = uv_accel["face_ids"]

    # Query returns indices in Shapely 2.x
    candidate_indices = tree.query(point)

    for idx in candidate_indices:
        poly = polygons[idx]

        if poly.contains(point) or poly.touches(point):
            face_index = face_ids[idx]

            tri_uv = mesh.visual.uv[mesh.faces[face_index]]
            tri_verts = mesh.vertices[mesh.faces[face_index]]

            bary = trimesh.triangles.points_to_barycentric(
                tri_uv[None, :, :],
                uv[None, :]
            )[0]

            pos_3d = (
                tri_verts[0] * bary[0] +
                tri_verts[1] * bary[1] +
                tri_verts[2] * bary[2]
            )

            return pos_3d, face_index

    return None, None

def uv_to_3d(mesh: trimesh.Trimesh, uv: np.ndarray, eps=1e-8):
    """
    Convert UV coordinates to a 3D position on a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh with UVs
    uv : (2,) array-like
        UV coordinate (0–1 range)

    Returns
    -------
    point_3d : (3,) np.ndarray or None
        3D position on the mesh, or None if UV not found
    face_index : int or None
        Index of the face containing the UV
    """

    if mesh.visual.uv is None:
        raise ValueError("Mesh does not contain UV coordinates")

    uv = np.asarray(uv)

    # UVs per face (F, 3, 2)
    uv_faces = mesh.visual.uv[mesh.faces]

    # Iterate faces (simple + robust)
    for face_index, tri_uv in enumerate(uv_faces):
        
        
        # Compute UV triangle area (2D cross product)
        v0 = tri_uv[1] - tri_uv[0]
        v1 = tri_uv[2] - tri_uv[0]
        area2 = abs(v0[0] * v1[1] - v0[1] * v1[0])

        # Skip degenerate UV triangles
        if area2 < eps:
            continue

        # Check if UV is inside triangle
        bary = trimesh.triangles.points_to_barycentric(
            tri_uv[None, :, :],
            uv[None, :]
        )[0]

        if np.all(bary >= -1e-6):
            # Interpolate 3D position
            tri_verts = mesh.vertices[mesh.faces[face_index]]

            point_3d = (
                tri_verts[0] * bary[0] +
                tri_verts[1] * bary[1] +
                tri_verts[2] * bary[2]
            )

            return point_3d, face_index

    return None, None

import numpy as np
import trimesh

def uv_to_face_id(mesh: trimesh.Trimesh, uv: np.ndarray, eps=1e-8):
    """
    Find the face index whose UV triangle contains the given UV coordinate.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh with UVs
    uv : (2,) array-like
        UV coordinate (usually in [0, 1])
    eps : float
        Numerical tolerance

    Returns
    -------
    face_index : int or None
        Face index if found, otherwise None
    """

    if mesh.visual.uv is None:
        raise ValueError("Mesh does not contain UV coordinates")

    uv = np.asarray(uv, dtype=np.float64)

    # UV triangles per face: (F, 3, 2)
    uv_faces = mesh.visual.uv[mesh.faces]

    for face_index, tri_uv in enumerate(uv_faces):

        # Compute UV triangle area (2D cross product)
        v0 = tri_uv[1] - tri_uv[0]
        v1 = tri_uv[2] - tri_uv[0]
        area2 = abs(v0[0] * v1[1] - v0[1] * v1[0])

        # Skip degenerate UV triangles
        if area2 < eps:
            continue

        # Compute barycentric coordinates in UV space
        bary = trimesh.triangles.points_to_barycentric(
            tri_uv[None, :, :],
            uv[None, :]
        )[0]

        # Check if inside triangle
        if np.all(bary >= -eps) and np.all(bary <= 1.0 + eps):
            return face_index

    return None

def create_point_cloud(mesh,resolution = 64):
    uv_tree = build_uv_acceleration(mesh)

    # we can sample the volume of Box primitives
    total_points = resolution * resolution
    uv_points = []
    for i in range(resolution):
        for j in range(resolution):
            uv_points.append(np.array([float(i)/float(resolution),float(j)/float(resolution)]))

    ref_points = []
    ref_face_ids = []
    for p in uv_points:
        p, face_id = uv_to_3d_fast(mesh,p,uv_tree)
        if(face_id != None):
            ref_points.append(p)
            ref_face_ids.append(face_id)
    points = [np.array(ref_points),np.array(ref_face_ids)]
    return points

def calculate_path(mesh,points,face_ids,starting_pos_index = 0):
    def distance_normal(cur_normal,next_normal):
        # Orientation distance
        cross_product = np.clip(np.dot(next_normal, cur_normal), -1, 1)
        return 2 - (cross_product + 1) 
    
    def distance_position(cur_pos,next_pos):
        return np.linalg.norm(next_pos-cur_pos)
    
    def distance_previous_direction(previous_dir,cur_dir):
        # Orientation distance
        cross_product = np.clip(np.dot(previous_dir, cur_dir), -1, 1)
        return 2 - (cross_product + 1) 

    jump_list = [0]
    path = []
    point_amount = points.shape[0]
    print(point_amount)
    exploration_map = np.zeros((points.shape[0]))

    current_index = starting_pos_index
    previous_dir = np.array([0,0,0])
    step = 0
    while(True):
        exploration_map[current_index] = 1
        path.append(points[current_index])
        min_distance = 999
        best_index = current_index
        for i in range(point_amount):
            if(exploration_map[i] < 1):
                d = distance_position(points[current_index],points[i])
                dir = ((points[i] - points[current_index]) / d)
                dist = 20*d + 5*distance_normal(mesh.face_normals[face_ids[current_index]],mesh.face_normals[face_ids[i]]) + 0.0 * distance_previous_direction(previous_dir,dir)
                if(dist < min_distance):
                    best_index = i
                    min_distance = dist
        step += 1
        if(step % 100 == 0):
            print(step," out of ",point_amount)
        if(min_distance > 0.5): # Jump
            jump_list.append(step)
        if(best_index == current_index):
            print("Blocked couldn't proceed")
            break
        else:
            previous_dir = ((points[best_index] - points[current_index]) / distance_position(points[current_index],points[best_index]))
        current_index = best_index
        
    
    return (path,jump_list)


if __name__ == "__main__":
    # load a large- ish PLY model with colors
    mesh = trimesh.load("./3d_models/dome_uv_mapped.obj")

    # Load texture image
    image = Image.open("./UVmap.jpg")
    image = image.convert("RGBA")
    #image_array = np.array(image)

    #Unwrap the image if needed
    if(mesh.visual.uv is None):
        mesh = mesh.unwrap(image)

    points = create_point_cloud(mesh,resolution=64)

    # distance from point to surface of meshdistances
    # create a PointCloud object out of each (n,3) list of points
    cloud_close = trimesh.points.PointCloud(points[0])

    # Convertir faces → positions 3D
    (path_points,jump_list) = calculate_path(mesh,points[0],points[1])

    # Visualisation
    #path = trimesh.load_path(path_points)
    #path.colors = [trimesh.visual.random_color()]

    path_entities = []

    for i in range(len(jump_list)-1):
        path_entities.append(Line(points=np.arange(jump_list[i],jump_list[i+1])))

    path = trimesh.path.Path3D(
    vertices=path_points,
    entities=path_entities,
    colors = [trimesh.visual.random_color() for e in path_entities]
    )

    # create a scene containing the mesh and two sets of points
    scene = trimesh.Scene([mesh, cloud_close,path])

    # show the scene we are using
    scene.show()


