#!/usr/bin/env python3
"""
Create interactive 3D HTML visualization of all extracted faces using Three.js.
Open in web browser to inspect and decide which faces to keep.
"""

import sys
from pathlib import Path
import json
import trimesh
import numpy as np

def visualize_faces_3d_html(faces_dir: str):
    """Create interactive 3D HTML visualization of all faces."""
    
    faces_dir = Path(faces_dir)
    metadata_file = faces_dir / "faces_metadata.json"
    output_file = faces_dir / "visualize_faces_3d.html"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Generate Three.js mesh data for each face
    meshes_data = []
    for face_info in metadata['faces']:
        face_file = faces_dir / face_info['file']
        mesh = trimesh.load(str(face_file))
        
        verts = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
        
        meshes_data.append({
            'id': face_info['face_id'],
            'file': face_info['file'],
            'vertices': verts,
            'faces': faces,
            'area': face_info['area_m2'],
            'triangles': face_info['num_triangles']
        })
    
    # Start HTML with Three.js
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>3D Face Mesh Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }
        #container { display: flex; height: 100vh; }
        #canvas { flex: 1; background: #0a0a0a; }
        #info { 
            width: 350px; 
            background: #2a2a2a; 
            padding: 20px; 
            overflow-y: auto; 
            border-left: 1px solid #444;
        }
        h2 { margin-bottom: 15px; color: #4CAF50; }
        .instructions {
            background: #3a3a3a;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 3px;
            font-size: 12px;
            border-left: 3px solid #4CAF50;
        }
        .face-item {
            padding: 12px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 4px solid #666;
        }
        .face-item:hover { background: #3a3a3a; }
        .face-item.large { border-left-color: #4CAF50; }
        .face-item.medium { border-left-color: #FF9800; }
        .face-item.small { border-left-color: #f44336; }
        .face-item.selected { 
            background: #1a4d1a; 
            border-left-color: #4CAF50;
            box-shadow: inset 0 0 10px rgba(76, 175, 80, 0.3);
        }
        .face-item.small.selected { background: #4d1a1a; border-left-color: #f44336; }
        .face-title { font-weight: bold; color: #4CAF50; }
        .face-stats { font-size: 11px; color: #aaa; margin-top: 5px; }
        #selectedSection { 
            margin-top: 20px; 
            padding-top: 20px; 
            border-top: 2px solid #444; 
        }
        .selected-badge { 
            display: inline-block; 
            background: #4CAF50; 
            color: #000; 
            padding: 5px 10px; 
            margin: 5px 5px 5px 0; 
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
        }
        .selected-badge .remove { cursor: pointer; margin-left: 5px; }
        button {
            background: #4CAF50;
            color: #000;
            border: none;
            padding: 12px 15px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            margin-top: 10px;
            width: 100%;
            transition: background 0.3s;
        }
        button:hover { background: #45a049; }
        #command {
            background: #1a1a1a;
            border: 1px solid #444;
            padding: 10px;
            margin-top: 10px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 11px;
            word-break: break-all;
            color: #4CAF50;
        }
        .legend {
            background: #3a3a3a;
            padding: 10px;
            margin-top: 15px;
            border-radius: 3px;
            font-size: 12px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            margin-right: 8px;
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="canvas"></div>
        <div id="info">
            <h2>🎯 Face Selection</h2>
            <div class="instructions">
                Click faces to select/deselect. Large faces are good for painting. Small faces are noise.
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50;"></div>
                    Large (>0.5m²) - Paint
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800;"></div>
                    Medium (0.01-0.5m²)
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f44336;"></div>
                    Small (<0.01m²) - Noise
                </div>
            </div>
            <div id="facesList"></div>
            <div id="selectedSection">
                <strong>✓ Selected Faces:</strong>
                <div id="selectedIds"></div>
                <button onclick="keepSelected()">✓ APPLY SELECTION</button>
                <div id="command"></div>
            </div>
        </div>
    </div>

    <script>
        const meshesData = """ + json.dumps(meshes_data) + """;
        const selectedFaces = new Set();
        let scene, camera, renderer, meshes = {};
        
        // Initialize Three.js
        function initThree() {
            const canvas = document.getElementById('canvas');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 10000);
            camera.position.set(0, 0, 3);
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            canvas.appendChild(renderer.domElement);
            
            // Lighting
            const light = new THREE.DirectionalLight(0xffffff, 0.8);
            light.position.set(5, 5, 5);
            scene.add(light);
            scene.add(new THREE.AmbientLight(0x404040));
            
            // Load meshes
            meshesData.forEach((data, idx) => {
                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(data.vertices.flat());
                const indices = new Uint32Array(data.faces.flat());
                
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                geometry.computeVertexNormals();
                
                const material = new THREE.MeshPhongMaterial({
                    color: 0x4CAF50,
                    emissive: 0x000000,
                    side: THREE.DoubleSide
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.faceId = data.id;
                mesh.userData.area = data.area;
                
                scene.add(mesh);
                meshes[data.id] = mesh;
            });
            
            // Auto-fit camera
            const box = new THREE.Box3().setFromObject(scene);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            camera.position.z = cameraZ * 1.5;
            camera.lookAt(scene.position);
            
            // Mouse interaction
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            
            renderer.domElement.addEventListener('click', (event) => {
                mouse.x = (event.clientX / renderer.domElement.clientWidth) * 2 - 1;
                mouse.y = -(event.clientY / renderer.domElement.clientHeight) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(scene.children);
                
                if (intersects.length > 0) {
                    const obj = intersects[0].object;
                    if (obj.userData.faceId !== undefined) {
                        toggleFace(obj.userData.faceId);
                    }
                }
            });
            
            // Render loop
            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }
            animate();
            
            // Handle window resize
            window.addEventListener('resize', () => {
                const w = canvas.clientWidth;
                const h = canvas.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            });
        }
        
        // UI Functions
        const facesList = document.getElementById('facesList');
        meshesData.forEach(data => {
            const div = document.createElement('div');
            div.className = 'face-item';
            if (data.area > 0.5) div.classList.add('large');
            else if (data.area > 0.01) div.classList.add('medium');
            else div.classList.add('small');
            
            div.innerHTML = `
                <div class="face-title">Face ${data.id}</div>
                <div class="face-stats">
                    Area: ${data.area.toFixed(4)}m² | ${data.triangles} tri
                </div>
            `;
            
            div.onclick = () => toggleFace(data.id);
            facesList.appendChild(div);
        });
        
        function toggleFace(id) {
            if (selectedFaces.has(id)) {
                selectedFaces.delete(id);
                meshes[id].material.color.setHex(0x4CAF50);
            } else {
                selectedFaces.add(id);
                meshes[id].material.color.setHex(0x81C784);
            }
            
            document.querySelectorAll('.face-item').forEach((item, idx) => {
                if (meshesData[idx].id === id) {
                    item.classList.toggle('selected');
                }
            });
            updateSelectedList();
        }
        
        function updateSelectedList() {
            const selectedIds = document.getElementById('selectedIds');
            selectedIds.innerHTML = '';
            Array.from(selectedFaces).sort((a,b) => a-b).forEach(id => {
                const span = document.createElement('span');
                span.className = 'selected-badge';
                span.innerHTML = `${id} <span class="remove" onclick="removeFace(${id})">✕</span>`;
                selectedIds.appendChild(span);
            });
            updateCommand();
        }
        
        function removeFace(id) {
            toggleFace(id);
        }
        
        function updateCommand() {
            const keep = Array.from(selectedFaces).sort((a,b) => a-b).join(' ');
            const cmd = keep ? `poetry run python extract_and_visualize_faces.py stl/test_part_m.stl --keep ${keep}` : '(select faces first)';
            document.getElementById('command').textContent = cmd;
        }
        
        function keepSelected() {
            const keep = Array.from(selectedFaces).sort((a,b) => a-b).join(' ');
            if (keep) {
                alert(`Run this command to keep selected faces:\\n\\n${document.getElementById('command').textContent}`);
            } else {
                alert('Please select at least one face');
            }
        }
        
        // Start
        initThree();
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ Created 3D interactive visualization: {output_file}")
    print(f"\nOpen in your web browser:")
    print(f"  file://{output_file.absolute()}")
    return output_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize faces in 3D HTML")
    parser.add_argument("--faces-dir", type=str, default="stl/faces", 
                       help="Path to faces directory (default: stl/faces)")
    
    args = parser.parse_args()
    visualize_faces_3d_html(args.faces_dir)
