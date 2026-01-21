import numpy as np
import trimesh


class ProceduralMeshGenerator:
    """
    Génère procéduralement des meshes pour l'entraînement.
    Complexité croissante:
    - Primitives simples (sphère, cube, cylindre)
    - Primitives avec perturbations
    - Objets composés
    - Surfaces complexes
    """
    
    @staticmethod
    def simple_sphere(subdivisions=2, radius=1.0):
        """Sphère simple"""
        return trimesh.creation.icosphere(
            subdivisions=subdivisions,
            radius=radius
        )
    
    @staticmethod
    def perturbed_sphere(subdivisions=2, radius=1.0, perturbation=0.1):
        """Sphère perturbée aléatoirement"""
        mesh = trimesh.creation.icosphere(
            subdivisions=subdivisions,
            radius=radius
        )
        
        # Perturber les vertices
        perturbation_vec = np.random.randn(*mesh.vertices.shape) * perturbation
        mesh.vertices += perturbation_vec
        
        return mesh
    
    @staticmethod
    def bumpy_sphere(subdivisions=3, radius=1.0, bump_scale=0.15):
        """Sphère avec bosses (bruit Perlin simplifié)"""
        mesh = trimesh.creation.icosphere(
            subdivisions=subdivisions,
            radius=radius
        )
        
        # Ajouter des bosses avec variation sinusoïdale
        for i, vertex in enumerate(mesh.vertices):
            # Calcul pseudo-Perlin (oscillations harmoniques)
            r = np.linalg.norm(vertex)
            theta = np.arctan2(vertex[1], vertex[0])
            phi = np.arccos(np.clip(vertex[2] / r, -1, 1))
            
            # Combinaison de fréquences
            bumps = (np.sin(theta * 3) * np.sin(phi * 3) + 
                    np.sin(theta * 7) * np.sin(phi * 5))
            
            direction = vertex / (r + 1e-6)
            mesh.vertices[i] = vertex + direction * bump_scale * bumps
        
        return mesh
    
    @staticmethod
    def cube_with_details(size=1.0, subdivisions=1):
        """Cube subdivis"""
        mesh = trimesh.creation.box(extents=[size, size, size])
        
        # Subdiviser en ajoutant des points aux arêtes
        for _ in range(subdivisions):
            mesh = mesh.subdivide()
        
        return mesh
    
    @staticmethod
    def cylinder(radius=0.5, height=2.0, sections=32):
        """Cylindre"""
        return trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=sections
        )
    
    @staticmethod
    def torus(major_radius=1.0, minor_radius=0.3):
        """Tore (donut)"""
        return trimesh.creation.torus(
            major_radius=major_radius,
            minor_radius=minor_radius
        )
    
    @staticmethod
    def compound_shape():
        """Objet composé: sphère + cube"""
        sphere = trimesh.creation.icosphere(radius=0.5).apply_translation([0.5, 0, 0])
        cube = trimesh.creation.box(extents=[0.8, 0.8, 0.8]).apply_translation([-0.5, 0, 0])
        
        # Union (si possible)
        try:
            return sphere.union(cube)
        except:
            # Sinon, combiner comme deux meshes
            return trimesh.util.concatenate([sphere, cube])
    
    @staticmethod
    def complex_surface():
        """Surface complexe: wavelet"""
        # Créer grille
        x = np.linspace(-np.pi, np.pi, 30)
        y = np.linspace(-np.pi, np.pi, 30)
        X, Y = np.meshgrid(x, y)
        
        # Fonction complexe
        Z = 0.5 * np.sin(X) * np.cos(Y) + 0.3 * np.sin(2*X) * np.cos(3*Y)
        
        # Créer mesh
        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Créer faces (grille)
        faces = []
        n_x, n_y = X.shape
        for i in range(n_x - 1):
            for j in range(n_y - 1):
                v0 = i * n_y + j
                v1 = i * n_y + j + 1
                v2 = (i + 1) * n_y + j
                v3 = (i + 1) * n_y + j + 1
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return trimesh.Mesh(vertices, np.array(faces))
    
    @staticmethod
    def random_mesh(complexity='simple'):
        """
        Génère un mesh aléatoire de complexité variable.
        
        Args:
            complexity: 'simple', 'medium', 'hard', 'expert'
        """
        if complexity == 'simple':
            # Primitives simples
            choice = np.random.randint(0, 3)
            if choice == 0:
                return ProceduralMeshGenerator.simple_sphere(subdivisions=1)
            elif choice == 1:
                return ProceduralMeshGenerator.cube_with_details(subdivisions=0)
            else:
                return ProceduralMeshGenerator.cylinder(sections=16)
        
        elif complexity == 'medium':
            # Primitives perturbées
            choice = np.random.randint(0, 4)
            if choice == 0:
                return ProceduralMeshGenerator.perturbed_sphere(subdivisions=2)
            elif choice == 1:
                return ProceduralMeshGenerator.bumpy_sphere(subdivisions=2)
            elif choice == 2:
                return ProceduralMeshGenerator.torus()
            else:
                return ProceduralMeshGenerator.cube_with_details(subdivisions=1)
        
        elif complexity == 'hard':
            # Formes complexes
            choice = np.random.randint(0, 3)
            if choice == 0:
                return ProceduralMeshGenerator.bumpy_sphere(subdivisions=3, bump_scale=0.2)
            elif choice == 1:
                return ProceduralMeshGenerator.complex_surface()
            else:
                return ProceduralMeshGenerator.compound_shape()
        
        else:  # expert
            # Combinaison de tout
            return ProceduralMeshGenerator.bumpy_sphere(
                subdivisions=3, 
                radius=1.0,
                bump_scale=np.random.uniform(0.1, 0.3)
            )
    
    @staticmethod
    def dataset_iterator(n_meshes=100, difficulty_schedule=None):
        """
        Générateur de dataset d'entraînement.
        
        Args:
            n_meshes: nombre de meshes à générer
            difficulty_schedule: None ou liste de complexités
        
        Yields:
            mesh: trimesh.Mesh
        """
        if difficulty_schedule is None:
            # Progression linéaire
            complexity_list = []
            n_simple = n_meshes // 4
            n_medium = n_meshes // 4
            n_hard = n_meshes // 4
            n_expert = n_meshes - n_simple - n_medium - n_hard
            
            complexity_list = (
                ['simple'] * n_simple +
                ['medium'] * n_medium +
                ['hard'] * n_hard +
                ['expert'] * n_expert
            )
        else:
            complexity_list = difficulty_schedule
        
        for complexity in complexity_list:
            yield ProceduralMeshGenerator.random_mesh(complexity)
