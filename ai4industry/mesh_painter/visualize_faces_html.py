#!/usr/bin/env python3
"""
Create interactive 3D HTML visualization of all extracted faces.
Open in web browser to inspect and decide which faces to keep.
"""

import sys
from pathlib import Path
import json
import numpy as np

def visualize_faces_html(faces_dir: str):
    """Create interactive HTML visualization of all faces."""
    import trimesh
    
    faces_dir = Path(faces_dir)
    metadata_file = faces_dir / "faces_metadata.json"
    output_file = faces_dir / "visualize_faces.html"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Start HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Face Mesh Visualization</title>
    <script src="https://threejs.org/build/three.min.js"></script>
    <script src="https://threejs.org/examples/js/controls/OrbitControls.js"></script>
    <style>
        * { margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f0f0; }
        #container { display: flex; height: 100vh; }
        #canvas { flex: 1; }
        #info { 
            width: 300px; 
            background: white; 
            padding: 20px; 
            overflow-y: auto; 
            border-left: 1px solid #ccc;
        }
        .face-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .face-item:hover { background: #f9f9f9; }
        .face-item.large { border-left: 4px solid #2ecc71; }
        .face-item.medium { border-left: 4px solid #f39c12; }
        .face-item.small { border-left: 4px solid #e74c3c; }
        .face-item.selected { background: #e3f2fd; }
        .face-title { font-weight: bold; color: #333; }
        .face-stats { font-size: 12px; color: #666; margin-top: 5px; }
        #selectedList { margin-top: 20px; padding-top: 20px; border-top: 2px solid #ccc; }
        .selected-id { 
            display: inline-block; 
            background: #2ecc71; 
            color: white; 
            padding: 5px 10px; 
            margin: 5px 5px 5px 0; 
            border-radius: 3px;
        }
        .selected-id .remove { cursor: pointer; margin-left: 5px; font-weight: bold; }
        button {
            background: #2ecc71;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
            width: 100%;
        }
        button:hover { background: #27ae60; }
        #instructions {
            background: #fff3cd;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 3px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="canvas"></div>
        <div id="info">
            <h2>Face Selection</h2>
            <div id="instructions">
                Click on faces in the list to select/deselect them. Green = large (good), Orange = medium, Red = small (artifacts).
            </div>
            <div id="facesList"></div>
            <div id="selectedList">
                <strong>Selected Faces:</strong>
                <div id="selectedIds"></div>
                <button onclick="deleteUnselected()">DELETE UNSELECTED FACES</button>
                <button onclick="keepSelected()">KEEP SELECTED FACES</button>
            </div>
        </div>
    </div>

    <script>
        const metadata = """ + json.dumps(metadata) + """;
        const selectedFaces = new Set();
        
        // Initialize UI
        const facesList = document.getElementById('facesList');
        metadata.faces.forEach(face => {
            const div = document.createElement('div');
            div.className = 'face-item';
            if (face.area_m2 > 0.5) div.classList.add('large');
            else if (face.area_m2 > 0.01) div.classList.add('medium');
            else div.classList.add('small');
            
            div.innerHTML = `
                <div class="face-title">Face ${face.face_id}</div>
                <div class="face-stats">
                    Area: ${face.area_m2.toFixed(4)} m²<br>
                    Triangles: ${face.num_triangles}<br>
                    File: ${face.file}
                </div>
            `;
            
            div.onclick = (e) => {
                e.stopPropagation();
                if (selectedFaces.has(face.face_id)) {
                    selectedFaces.delete(face.face_id);
                    div.classList.remove('selected');
                } else {
                    selectedFaces.add(face.face_id);
                    div.classList.add('selected');
                }
                updateSelectedList();
            };
            
            facesList.appendChild(div);
        });
        
        function updateSelectedList() {
            const selectedIds = document.getElementById('selectedIds');
            selectedIds.innerHTML = '';
            Array.from(selectedFaces).sort((a,b) => a-b).forEach(id => {
                const span = document.createElement('span');
                span.className = 'selected-id';
                span.innerHTML = `${id} <span class="remove" onclick="removeFace(${id})">×</span>`;
                selectedIds.appendChild(span);
            });
        }
        
        function removeFace(id) {
            selectedFaces.delete(id);
            document.querySelectorAll('.face-item').forEach(item => {
                const title = item.querySelector('.face-title').textContent;
                const faceId = parseInt(title.split(' ')[1]);
                if (faceId === id) item.classList.remove('selected');
            });
            updateSelectedList();
        }
        
        function deleteUnselected() {
            const keep = Array.from(selectedFaces).sort((a,b) => a-b).join(' ');
            const cmd = `poetry run python extract_and_visualize_faces.py stl/test_part_m.stl --keep ${keep}`;
            alert(`Run this command to keep selected faces:\\n\\n${cmd}`);
        }
        
        function keepSelected() {
            const keep = Array.from(selectedFaces).sort((a,b) => a-b).join(' ');
            const cmd = `poetry run python extract_and_visualize_faces.py stl/test_part_m.stl --keep ${keep}`;
            alert(`Run this command to keep selected faces:\\n\\n${cmd}`);
        }
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ Created interactive visualization: {output_file}")
    print(f"\nOpen in your web browser:")
    print(f"  file://{output_file.absolute()}")
    print(f"\nor run:")
    print(f"  xdg-open {output_file.absolute()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize faces in HTML")
    parser.add_argument("--faces-dir", type=str, default="stl/faces", 
                       help="Path to faces directory (default: stl/faces)")
    
    args = parser.parse_args()
    visualize_faces_html(args.faces_dir)
