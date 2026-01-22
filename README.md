# AeroSplinePathFingind
Ce code est une tentative de réponse au use case de AeroSpline dans le cadre de la convention ai4industry.

J'applique une méthode de reinforcement learning.
Les fichiers importants sont : mesh_env.py, main.py, pathfinding/rl_agent.py, pathfinding/rl_env.py
mesh_env.py gère la logique du mesh.
rl_agent.py implémente la class QAgent qui utilise une methode de qlearning.
rl_env.py implémente la logique qui régit l'agent avec les reward.
Executer le fichier main.py.
