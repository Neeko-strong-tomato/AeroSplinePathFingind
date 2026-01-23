# AeroSplinePathFingind
Ce code est une tentative de réponse au use case de AeroSpline dans le cadre de la convention ai4industry.


Voici l’ensemble des branches de développement mises en place par l’équipe IA pour le use case, ainsi qu’une branche qui a exploré l’option algorithmique :

- Main : Branche principale.
- Auto-en-full-vibecoding : Algorithme de recouvrement de faces (travail de l’équipe algo).
- Cam2.0 : Essai d’une architecture RL, basée sur la segmentation mise en place par l’équipe. Direction qui pourrait marcher avec plus de capacité d’entraînement et de temps.
- Maillage : Tentataive d'utiliser la segmentation pour effectué le dépliage du mesh et appliqué les points UV avant de reconstruire le mesh
- RL2D :
- cam : Échec de la mise en place de RL (basé sur l’architecture du main initiale, pas forcément adaptée).
- noLLM : Première version algorithmique très simple. Le but de cette branche était de faire un test sans utilisation de LLM pour découvrir trimesh.
- rluv : Tentative de mise en place de RL, problème majeur le robot traverse le mesh 3D. Bonne couverture cependant après beaucoup d'entrainement.
- uv_mapping : Tentative de méthode algorithmique par cartographie UV puis conversion en nuage de point pour une création de chemin par algorithme glouton

De manière générale, chaque branche dérive d'une base de code fourni dans le main contenant des modules pour :
   - emmuler un environement 3D pour simuler le tracé
   - des blueprints pour les différentes méthodes de pathFinding
   - un gestionnaire de la tête robot
   - des implementation naïves pour le RL & l'algorithmie
   - l'algorithme de segmentation

De plus une tentative d'approche RL prosposant des trajets suivant les arêtes.
