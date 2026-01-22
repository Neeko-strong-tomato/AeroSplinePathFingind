import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

def visualize_stl_sequentially(folder_path):
    # Get and sort all .stl files
    stl_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.stl')])

    if not stl_files:
        print(f"No STL files found in {folder_path}")
        return

    for i, file_name in enumerate(stl_files):
        print(f"[{i+1}/{len(stl_files)}] Visualizing: {file_name}")
        file_path = os.path.join(folder_path, file_name)
        
        # Load the mesh
        your_mesh = mesh.Mesh.from_file(file_path)
        
        # Setup Figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the 3D object
        poly_collection = mplot3d.art3d.Poly3DCollection(your_mesh.vectors, alpha=0.8)
        
        # Set a distinct color (e.g., light blue with black edges)
        poly_collection.set_facecolor('skyblue')
        poly_collection.set_edgecolor('black')
        poly_collection.set_linewidth(0.1)
        
        ax.add_collection3d(poly_collection)

        # Scale axes based on the specific mesh
        scale = your_mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Labels and Title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f"File: {file_name}\n(Close window to see next)")
        
        plt.show()

# Run the sequential viewer
visualize_stl_sequentially('stl/faces')