# AeroSplinePathFingind
Ce code est une tentative de réponse au use case de AeroSpline dans le cadre de la convention ai4industry.


Voici l’ensemble des branches de développement mises en place par l’équipe IA pour le use case, ainsi qu’une branche qui a exploré l’option algorithmique :

- Main : Branche principale.
- Auto-en-full-vibecoding : Algorithme de recouvrement de faces (travail de l’équipe algo).
- Cam2.0 : Essai d’une architecture RL, basée sur la segmentation mise en place par l’équipe. Direction qui pourrait marcher avec plus de capacité d’entraînement et de temps.
- Maillage : Tentataive d'utiliser la segmentation pour effectué le dépliage du mesh et appliqué les points UV avant de reconstruire le mesh
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

De plus la branche contient une tentative d'approche de reinforcement learning :

Ce code propose une approche de Coverage Path Planning par apprentissage par renforcement appliquée à un graphe d’arêtes extrait d’un mesh 3D.
Chaque arête est considérée comme un nœud, et l’agent apprend à se déplacer d’arête en arête afin de maximiser la couverture globale.
L’environnement RL est basé sur :
une observation composée de la position 3D normalisée de l’arête courante et du taux de progression,
des actions correspondant aux déplacements vers des arêtes adjacentes,
une fonction de récompense favorisant la visite d’arêtes non explorées et pénalisant les boucles.
L’algorithme PPO permet d’apprendre une politique de navigation efficace sur des géométries simples.
La trajectoire générée est ensuite visualisée sous forme de tube continu le long des arêtes, offrant une bonne lisibilité et une impression de glissement de l’outil.

Points forts :
   - Trajectoire continue et lisible
   - Graphe simple et robuste
   - RL pertinent pour la planification globale
     
Limites :
   - La couverture est topologique (arêtes) et non surfacique
     => L’outil ne balaye pas réellement l’intérieur des faces
   - Absence d’orientation outil et de bande de peinture réelle

Conclusion :
Cette approche constitue un prototype convaincant de planification par RL sur mesh, adapté à l’exploration ou au guidage.
De plus la stratégie souvent trouvé comme étant optimal par le modèle est souvent le zigzag, ce qui semble tendre à prouver que c'est une méthode optimal.
Cependant sur les bords arrondis, le RL prend tout son sens, en trouvant des paternes de chemins plus adaté sur ces types de faces.
Pour une application industrielle de peinture, elle doit être complétée par une génération de trajectoires continues sur la surface (bandes, zigzag, projection tangentielle).
