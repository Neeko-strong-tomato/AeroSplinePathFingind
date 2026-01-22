import os
from mesh.segmentation import segmentation, export_segments_for_rl
from mesh.mesh_loader import load_mesh
from env.mesh_env import MeshEnv
from robot.robot import Robot
from rl.train import train_on_segment
from rl.evaluate import evaluate_model
from coverage.zigzag import zigzag_coverage
from visualization.plotter import MeshPlotter
import trimesh

import sys


# Ajoute le dossier racine du projet dans PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    # 1️⃣ Mesh industriel
    mesh_file = "data/meshes/fan.stl"  # change selon ton test
    print(f"[MAIN] Chargement du mesh: {mesh_file}")
    mesh = load_mesh(mesh_file)

    # 2️⃣ Segmentation
    print("[MAIN] Segmentation du mesh...")
    segmented_meshes_dir = "segments"
    os.makedirs(segmented_meshes_dir, exist_ok=True)

    segment_files = export_segments_for_rl(mesh_file, segmented_meshes_dir, angle_deg=30)
    print(f"[MAIN] {len(segment_files)} segments générés.")

    # 3️⃣ Entraînement RL sur chaque segment
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    rl_models = []
    for seg_file in segment_files:
        model_name = os.path.basename(seg_file).replace(".stl", "_ppo")
        model_path = os.path.join(models_dir, model_name)
        print(f"[MAIN] Entraînement RL sur {seg_file}...")
        train_on_segment(seg_file, model_path, timesteps=500)  # tu peux augmenter plus tard
        rl_models.append(model_path)

    # 4️⃣ Évaluation RL et Zigzag
    for seg_file, model_path in zip(segment_files, rl_models):
        print(f"[MAIN] Évaluation RL sur {seg_file}...")
        metrics_rl = evaluate_model(seg_file, model_path)
        print(f"RL Metrics: {metrics_rl}")

        print(f"[MAIN] Calcul trajectoire Zigzag sur {seg_file}...")
        submesh = load_mesh(seg_file)
        path_zigzag = zigzag_coverage(submesh)
        coverage_zigzag = len(set(path_zigzag)) / len(submesh.faces)
        print(f"Zigzag coverage: {coverage_zigzag:.2f}")

        # 5️⃣ Visualisation
        print("[MAIN] Visualisation...")
        robot = Robot(submesh)
        # simulation simple : suivre RL path
        obs_env = MeshEnv(submesh)
        obs_env.reset()
        robot.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            # action RL
            import random
            action = random.randint(0, 4)  # remplace par model.predict pour vrai RL
            robot.step(action)
            steps += 1
            if len(robot.path) >= len(submesh.faces):
                done = True

        plotter = MeshPlotter(submesh)
        plotter.plot(robot)

    print("[MAIN] Projet terminé.")


if __name__ == "__main__":
    main()
