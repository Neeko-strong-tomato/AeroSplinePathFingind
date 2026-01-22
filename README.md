# AeroSplinePathFingind

Ce code est une tentative de réponse au use case de AeroSpline dans le cadre de la convention ai4industry.

Sur cette branche j'ai tester la méthode suivante :
Le but global, c’était de prendre des pièces 3D (des meshes), de les découper en segments pour que chaque segment ait des faces qui sont à peu près dans la même direction, et ensuite d’entraîner un agent RL pour générer des trajectoires de traitement de surface sur ces segments.

- Chargement du mesh : J’ai commencé par charger un mesh 3D avec trimesh, donc par exemple le “fan.stl” qu’on a dans les fichiers de test. C’est juste pour avoir les faces, les sommets, et pouvoir travailler dessus.

- Segmentation: Ensuite j’ai fait une segmentation avec le code dans mesh/segmentation.py (repris de l'équipe segmentation). L’idée c’est de prendre le mesh et de créer des sous-meshes où toutes les faces d’un segment ont des normales à peu près similaires (angle < 30° par exemple). Ça permet de découper les surfaces “courbes” naturellement. Après chaque segment est exporté dans un fichier STL séparé pour pouvoir l’utiliser facilement avec RL.

- Environnement RL: J’ai créé un environnement MeshEnv qui suit l’API de Gym (ou Gymnasium maintenant). L’état de l’agent, c’est la position sur la face, la normale, la couverture actuelle, et quelques infos sur le dernier mouvement. Les actions sont discrètes : aller vers un voisin, tourner gauche, tourner droite, rester, ou stop. La fonction de récompense encourage à couvrir de nouvelles faces, éviter les recouvrements, et limiter les changements brusques d’orientation.

- Entraînement RL: Pour tester, j’ai utilisé stable-baselines3 et le PPO comme algo RL. Pour un premier test rapide, j’ai mis peu de timesteps (500 par segment) pour juste vérifier que ça marche. Chaque segment a son propre modèle RL, stocké dans models/.

- Évaluation et comparaison avec Zigzag : Après entraînement, je fais une évaluation sur chaque segment pour voir la couverture atteinte par l’agent RL. En parallèle, je génère un chemin classique “zigzag” pour comparer (coverage et nombre de croisements).

- Simulation robot + visualisation: J’ai simulé le robot qui suit les actions RL sur chaque segment. J’ai limité le nombre d’étapes pour que ce soit rapide.Et j’ai affiché le tout avec mon plotter pour voir si le robot couvre correctement la surface.

**Problème** :
L’entraînement RL, si on veut qu’il soit efficace, est extrêmement long. Pour l’instant, les résultats ne sont pas au rendez-vous. Avec la base que j’ai générée (architecture + code) et les capacités limitées de mon ordinateur, je n’ai pu faire que des tests très rapides.

Les principaux soucis qui font que ça ne marche pas encore correctement sont :

- Complexité du mesh et du nombre de faces : Les meshes industriels ont beaucoup de faces et des géométries très complexes. Chaque étape RL doit calculer les voisins, les normales, la distance… et ça devient vite lourd pour l’ordinateur.

- Segmentation : Même si j’ai segmenté le mesh, certaines faces isolées ou zones très fines posent problème. L’agent peut se retrouver bloqué ou avoir des actions limitées, ce qui ralentit l’entraînement et réduit la couverture.

- Fonction de récompense : Elle est encore assez simple et ne prend pas en compte tous les critères industriels (croisements, régularité, changements d’orientation…). Du coup, l’agent apprend lentement et parfois se contente de faire des mouvements aléatoires.

- Compatibilité et dépendances : Gym (maintenant Gymnasium), Stable-Baselines3, et certaines versions de NumPy posent parfois des conflits ou ralentissent l’exécution si elles ne sont pas parfaitement configurées.

En résumé : le code et l’architecture sont là, mais pour produire des trajectoires vraiment exploitables, il faudrait soit un ordinateur beaucoup plus puissant, soit un entraînement sur serveur, soit une simplification des meshes pour l’expérimentation.
