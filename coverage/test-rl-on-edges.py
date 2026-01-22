import numpy as np
import trimesh
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
import pyvista as pv
from stable_baselines3 import PPO

# =============================================================================
# 1. ENVIRONNEMENT RL : NAVIGATION SUR ARÊTES
# =============================================================================

class AeroSplineEdgeEnv(gym.Env):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.edge_centers = mesh.vertices[mesh.edges_unique].mean(axis=1)
        
        # Construction du graphe d'adjacence des arêtes
        self.graph = nx.Graph()
        # On connecte les arêtes qui partagent un sommet commun
        vertex_to_edges = [[] for _ in range(len(mesh.vertices))]
        for i, edge in enumerate(mesh.edges_unique):
            vertex_to_edges[edge[0]].append(i)
            vertex_to_edges[edge[1]].append(i)
        
        for e_list in vertex_to_edges:
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    self.graph.add_edge(e_list[i], e_list[j])

        self.n_edges = len(mesh.edges_unique)
        self.action_space = spaces.Discrete(6) # Plus de voisins possibles sur les sommets
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_edge = 0
        self.visited = np.zeros(self.n_edges)
        self.visited[self.current_edge] = 1
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.edge_centers[self.current_edge]
        # On normalise la position relative au centre du mesh
        norm_pos = (pos - self.mesh.center_mass) / (self.mesh.scale + 1e-6)
        progress = np.sum(self.visited) / self.n_edges
        return np.append(norm_pos, [progress]).astype(np.float32)

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_edge))
        reward = -0.01
        
        if neighbors:
            self.current_edge = neighbors[action % len(neighbors)]
            if self.visited[self.current_edge] == 0:
                reward += 10.0
                self.visited[self.current_edge] = 1
            else:
                reward -= 1.0
        
        self.steps += 1
        done = np.all(self.visited) or self.steps >= self.n_edges * 1.5
        return self._get_obs(), reward, done, False, {}

# =============================================================================
# 2. GENERATION ET VISUALISATION
# =============================================================================

def run_edge_painting(mesh_path):
    # Chargement
    raw_mesh = trimesh.load(mesh_path, force='mesh')
    
    # Création environnement
    env = AeroSplineEdgeEnv(raw_mesh)
    
    # Apprentissage express
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=35000)
    
    # Inférence avec heuristique de "peinture"
    obs, _ = env.reset()
    trajectory = [env.edge_centers[env.current_edge]]
    
    for _ in range(env.n_edges * 2):
        action, _ = model.predict(obs)
        # Heuristique : Priorité aux arêtes proches non visitées
        neighbors = list(env.graph.neighbors(env.current_edge))
        unvisited = [n for n in neighbors if env.visited[n] == 0]
        
        if unvisited:
            env.current_edge = unvisited[0]
            env.visited[env.current_edge] = 1
        else:
            obs, _, done, _, _ = env.step(action)
            
        trajectory.append(env.edge_centers[env.current_edge])
        if np.all(env.visited): break

    # --- VISUALISATION PYVISTA ---
    pl = pv.Plotter(title="AeroSpline - Edge Painting Mode")
    
    # Le mesh en filaire pour bien voir les arêtes
    pv_mesh = pv.wrap(raw_mesh)
    pl.add_mesh(pv_mesh, color="white", opacity=0.3, show_edges=True, edge_color="black")
    
    # La trajectoire de peinture sur les arêtes
    path = np.array(trajectory)
    if len(path) > 1:
        # On crée un tube pour simuler le dépôt de peinture sur l'arête
        tube = pv.MultipleLines(points=path).tube(radius=0.05)
        pl.add_mesh(tube, color="blue", label="Trajectoire Outil (Arêtes)")
    
    pl.add_legend()
    pl.show()

if __name__ == "__main__":
    # Test sur un cube pour bien voir le suivi d'arêtes
    cube = trimesh.creation.box(extents=[2, 2, 2])
    cube.export("temp_cube.stl")
    run_edge_painting("fan.stl")