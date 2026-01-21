import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path

from mesh_rl_env import MeshCoverageRLEnv
from dqn_agent import DQNAgent
from mesh_generator import ProceduralMeshGenerator


class RLTrainer:
    """Entraineur pour agent DQN sur couverture de mesh"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = None
        self.training_history = {
            'episode_rewards': [],
            'episode_coverage': [],
            'episode_path_length': [],
            'episode_steps': []
        }
    
    @staticmethod
    def _default_config():
        return {
            'state_dim': 8,              # position(3) + orientation(3) + coverage(1) + steps(1)
            'action_dim': 8,             # 8 directions
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.9999,
            'buffer_size': 50000,
            'batch_size': 32,
            'episodes': 100,
            'target_update_frequency': 10,
            'max_steps_per_episode': 5000,
            'coverage_radius': 0.1,
            'save_frequency': 10,
            'save_dir': './models'
        }
    
    def initialize_agent(self):
        """Initialise l'agent DQN"""
        self.agent = DQNAgent(
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            learning_rate=self.config['learning_rate'],
            gamma=self.config['gamma'],
            epsilon=self.config['epsilon_start'],
            epsilon_min=self.config['epsilon_min'],
            epsilon_decay=self.config['epsilon_decay'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            device=self.device
        )
        print(f"Agent DQN initialisé sur {self.device}")
    
    def train(self, n_episodes=None):
        """
        Boucle d'entraînement principale
        
        Args:
            n_episodes: nombre d'épisodes (ou utilise config)
        """
        if self.agent is None:
            self.initialize_agent()
        
        n_episodes = n_episodes or self.config['episodes']
        
        # Créer dossier de sauvegarde
        Path(self.config['save_dir']).mkdir(exist_ok=True)
        
        print(f"\n🚀 Démarrage entraînement DQN sur {self.device}")
        print(f"Episodes: {n_episodes}, Max steps: {self.config['max_steps_per_episode']}\n")
        
        # Boucle d'entraînement
        pbar = tqdm(range(n_episodes), desc="Entraînement")
        
        for episode in pbar:
            # Générer un mesh aléatoire
            mesh = self._get_training_mesh(episode, n_episodes)
            
            # Créer environnement
            env = MeshCoverageRLEnv(
                mesh=mesh,
                coverage_radius=self.config['coverage_radius'],
                max_steps=self.config['max_steps_per_episode']
            )
            
            # Reset
            state = env.reset()
            episode_reward = 0.0
            done = False
            
            # Épisode
            while not done:
                # Choix action
                action = self.agent.choose_action(state, training=True)
                
                # Étape env
                next_state, reward, done, info = env.step(action)
                
                # Mémoriser
                self.agent.remember(state, action, reward, next_state, done)
                
                # Entraîner
                loss = self.agent.replay()
                
                # Update
                state = next_state
                episode_reward += reward
            
            # Fin d'épisode
            coverage = env.get_coverage_percentage()
            path_length = env.get_path_length()
            n_steps = env.step_count
            
            # Enregistrer statistiques
            self.agent.episode_rewards.append(episode_reward)
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_coverage'].append(coverage)
            self.training_history['episode_path_length'].append(path_length)
            self.training_history['episode_steps'].append(n_steps)
            
            # Update target network
            if (episode + 1) % self.config['target_update_frequency'] == 0:
                self.agent.update_target_network()
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Sauvegarde
            if (episode + 1) % self.config['save_frequency'] == 0:
                self.save_checkpoint(episode + 1)
            
            # Update barre
            stats = self.agent.get_statistics()
            pbar.set_postfix({
                'reward': episode_reward,
                'coverage': f'{coverage:.1f}%',
                'epsilon': f'{stats["epsilon"]:.3f}',
                'avg_loss': f'{stats["avg_loss"]:.4f}'
            })
        
        print("\n✅ Entraînement terminé!")
        self.save_checkpoint(n_episodes, final=True)
        return self.training_history
    
    def _get_training_mesh(self, episode, total_episodes):
        """Obtient un mesh adapté au stade d'entraînement"""
        ratio = episode / total_episodes
        
        if ratio < 0.25:
            return ProceduralMeshGenerator.random_mesh('simple')
        elif ratio < 0.5:
            return ProceduralMeshGenerator.random_mesh('medium')
        elif ratio < 0.75:
            return ProceduralMeshGenerator.random_mesh('hard')
        else:
            return ProceduralMeshGenerator.random_mesh('expert')
    
    def evaluate(self, mesh=None, n_runs=5, max_steps=5000):
        """
        Évalue l'agent sur un mesh donné.
        
        Args:
            mesh: trimesh.Mesh (ou génère aléatoire)
            n_runs: nombre d'évaluations
            max_steps: max étapes par eval
        
        Returns:
            dict avec statistiques
        """
        if self.agent is None:
            raise RuntimeError("Agent non initialisé. Appelez train() d'abord.")
        
        results = {
            'coverage': [],
            'path_length': [],
            'steps': [],
            'reward': []
        }
        
        for run in range(n_runs):
            # Mesh
            if mesh is None:
                test_mesh = ProceduralMeshGenerator.random_mesh('hard')
            else:
                test_mesh = mesh
            
            # Env
            env = MeshCoverageRLEnv(test_mesh, max_steps=max_steps)
            state = env.reset()
            done = False
            episode_reward = 0.0
            
            # Boucle greedy (pas d'exploration)
            while not done:
                action = self.agent.choose_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                episode_reward += reward
            
            # Enregistrer résultats
            results['coverage'].append(env.get_coverage_percentage())
            results['path_length'].append(env.get_path_length())
            results['steps'].append(env.step_count)
            results['reward'].append(episode_reward)
        
        # Statistiques
        stats = {
            'coverage_mean': np.mean(results['coverage']),
            'coverage_std': np.std(results['coverage']),
            'path_length_mean': np.mean(results['path_length']),
            'path_length_std': np.std(results['path_length']),
            'steps_mean': np.mean(results['steps']),
            'reward_mean': np.mean(results['reward']),
            'raw_results': results
        }
        
        return stats
    
    def save_checkpoint(self, episode, final=False):
        """Sauvegarde checkpoint"""
        filename = f"agent_final.pt" if final else f"agent_ep{episode}.pt"
        filepath = os.path.join(self.config['save_dir'], filename)
        self.agent.save(filepath)
        
        if final:
            # Sauvegarder aussi l'historique
            history_path = os.path.join(self.config['save_dir'], 'training_history.npy')
            np.save(history_path, self.training_history)
            print(f"✅ Modèle final sauvegardé: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Charge checkpoint"""
        if self.agent is None:
            self.initialize_agent()
        self.agent.load(filepath)
        print(f"✅ Modèle chargé: {filepath}")
    
    def plot_training_curves(self):
        """Affiche les courbes d'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Récompense par épisode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid()
        
        # Coverage
        axes[0, 1].plot(self.training_history['episode_coverage'])
        axes[0, 1].set_title('Couverture par épisode (%)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Coverage %')
        axes[0, 1].grid()
        axes[0, 1].axhline(y=95, color='r', linestyle='--', label='Objectif 95%')
        axes[0, 1].legend()
        
        # Path length
        axes[1, 0].plot(self.training_history['episode_path_length'])
        axes[1, 0].set_title('Longueur du trajet')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Path Length')
        axes[1, 0].grid()
        
        # Steps
        axes[1, 1].plot(self.training_history['episode_steps'])
        axes[1, 1].set_title('Nombre d\'étapes par épisode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'training_curves.png'), dpi=100)
        print("✅ Courbes d'entraînement sauvegardées")
        plt.show()


def main():
    """Script de démonstration"""
    
    # Configuration
    config = RLTrainer._default_config()
    config['episodes'] = 50  # Commencer petit
    config['save_directory'] = './coverage_models'
    
    # Créer entraîneur
    trainer = RLTrainer(config)
    trainer.initialize_agent()
    
    # Entraîner
    history = trainer.train(n_episodes=config['episodes'])
    
    # Évaluer
    print("\n📊 Évaluation finale...")
    eval_stats = trainer.evaluate()
    print(f"Coverage moyen: {eval_stats['coverage_mean']:.1f}% ± {eval_stats['coverage_std']:.1f}%")
    print(f"Path length moyen: {eval_stats['path_length_mean']:.2f}")
    
    # Plots
    trainer.plot_training_curves()


if __name__ == '__main__':
    main()
