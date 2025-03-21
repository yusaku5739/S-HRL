import yaml
import torch
import numpy as np
from pathlib import Path
import wandb
from datetime import datetime
import time
from collections import deque

from env import GridWorldEnv, RLWrapper
from ppo import GRUPPOAgentWithTarget

class MultiAgentTrainer:
    def __init__(self, config_path: str, num_agents: int = 2,
                 project_name: str = "marl-gridworld",
                 entity: str = None,
                 total_episodes: int = 10000, 
                 eval_interval: int = 100,
                 save_interval: int = 1000, 
                 log_interval: int = 10):
        
        # 設定の読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 環境の設定を更新
        self.config['env']['num_players'] = num_agents
        
        # WandBの初期化
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config={
                "environment": self.config,
                "num_agents": num_agents,
                "total_episodes": total_episodes,
                "eval_interval": eval_interval,
                "save_interval": save_interval,
                "log_interval": log_interval
            }
        )
        
        # トレーニングパラメータ
        self.num_agents = num_agents
        self.total_episodes = total_episodes
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # 環境の初期化
        self.env = RLWrapper(GridWorldEnv(config=self.config, render_mode=None))
        self.eval_env = RLWrapper(GridWorldEnv(config=self.config, render_mode="human"))
        
        # エージェントの初期化
        self.agents = [
            GRUPPOAgentWithTarget(
                obs_dim={k: v.shape[0] for k, v in self.env.observation_spaces[0].spaces.items()},
                action_dim=self.env.action_space.n
            ) for _ in range(num_agents)
        ]
        
        # モデル保存用ディレクトリの設定
        self.save_dir = Path(wandb.run.dir) / "models"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 移動平均の計算用
        self.reward_windows = [deque(maxlen=100) for _ in range(num_agents)]
        
        # WandBのArtifactとしてconfig保存
        artifact = wandb.Artifact('config', type='config')
        with artifact.new_file('config.yaml') as f:
            yaml.dump(self.config, f)
        wandb.log_artifact(artifact)
    
    def save_models(self, episode: int):
        """モデルの保存とWandBへのアップロード"""
        model_dir = self.save_dir / f'models_ep{episode}'
        model_dir.mkdir(exist_ok=True)
        
        # モデルの保存
        for i, agent in enumerate(self.agents):
            model_path = model_dir / f'agent_{i}.pt'
            torch.save({
                'actor_critic_state_dict': agent.actor_critic.state_dict(),
                'target_critic_state_dict': agent.target_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, model_path)
        
        # WandBのArtifactとしてモデルを保存
        artifact = wandb.Artifact(
            f'model-episode-{episode}', 
            type='model',
            description=f'Model checkpoint at episode {episode}'
        )
        artifact.add_dir(str(model_dir))
        wandb.log_artifact(artifact)
    
    def evaluate(self, num_episodes: int = 5):
        """エージェントの評価"""
        eval_rewards = [[] for _ in range(self.num_agents)]
        eval_steps = []
        
        for ep in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_rewards = [0] * self.num_agents
            steps = 0
            
            # 隠れ状態のリセット
            for agent in self.agents:
                agent.hidden_state = None
            
            while not done:
                steps += 1
                actions = []
                for i, agent in enumerate(self.agents):
                    obs_tensor = {k: torch.FloatTensor(v[i]) for k, v in obs.items()}
                    action, _, _ = agent.select_action(obs_tensor)
                    actions.append(action)
                
                obs, rewards, dones, _, _ = self.eval_env.step(actions)
                
                for i in range(self.num_agents):
                    episode_rewards[i] += rewards[i]
                
                done = any(dones)
            
            eval_steps.append(steps)
            for i in range(self.num_agents):
                eval_rewards[i].append(episode_rewards[i])
        
        # 評価結果をWandBにログ
        eval_metrics = {
            f'eval/agent_{i}/mean_reward': np.mean(rewards)
            for i, rewards in enumerate(eval_rewards)
        }
        eval_metrics['eval/mean_steps'] = np.mean(eval_steps)
        wandb.log(eval_metrics)
        
        return [np.mean(rewards) for rewards in eval_rewards]
    
    def log_episode_metrics(self, episode: int, episode_rewards: list, 
                          episode_steps: int, loss_info: dict = None):
        """エピソードごとのメトリクスをWandBに記録"""
        metrics = {}
        
        # 報酬のログ
        for i in range(self.num_agents):
            metrics[f'train/agent_{i}/episode_reward'] = episode_rewards[i]
            metrics[f'train/agent_{i}/reward_moving_avg'] = np.mean(self.reward_windows[i])
        
        # エピソード情報のログ
        metrics['train/episode_steps'] = episode_steps
        metrics['train/episode'] = episode
        
        # 損失情報のログ（存在する場合）
        if loss_info:
            for i in range(self.num_agents):
                metrics[f'train/agent_{i}/actor_loss'] = loss_info[i]['actor_loss']
                metrics[f'train/agent_{i}/value_loss'] = loss_info[i]['value_loss']
                metrics[f'train/agent_{i}/entropy'] = loss_info[i]['entropy']
        
        wandb.log(metrics)
    
    def train(self):
        """マルチエージェント学習のメインループ"""
        start_time = time.time()
        wandb.run.summary['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for episode in range(self.total_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_rewards = [0] * self.num_agents
            episode_steps = 0
            
            # 隠れ状態のリセット
            for agent in self.agents:
                agent.hidden_state = None
            
            while not done:
                episode_steps += 1
                actions = []
                
                # 各エージェントの行動選択
                for i, agent in enumerate(self.agents):
                    obs_tensor = {k: torch.FloatTensor(v[i]) for k, v in obs.items()}
                    action, log_prob, value = agent.select_action(obs_tensor)
                    actions.append(action)
                
                # 環境ステップ
                next_obs, rewards, dones, _, _ = self.env.step(actions)
                
                # 経験の保存
                for i, agent in enumerate(self.agents):
                    agent.buffer.add_episode([{
                        'obs': {k: torch.FloatTensor(v[i]) for k, v in obs.items()},
                        'next_obs': {k: torch.FloatTensor(v[i]) for k, v in next_obs.items()},
                        'action': actions[i],
                        'reward': rewards[i],
                        'log_prob': log_prob,
                        'value': value,
                        'done': dones[i]
                    }])
                    episode_rewards[i] += rewards[i]
                
                obs = next_obs
                done = any(dones)
            
            # 報酬の移動平均を更新
            for i in range(self.num_agents):
                self.reward_windows[i].append(episode_rewards[i])
            
            # モデルの更新と損失の記録
            loss_info = None
            if episode % self.log_interval == 0:
                loss_info = []
                for i, agent in enumerate(self.agents):
                    agent_loss_info = agent.update()
                    if agent_loss_info:
                        loss_info.append(agent_loss_info)
            
            # メトリクスのログ
            if episode % self.log_interval == 0:
                self.log_episode_metrics(episode, episode_rewards, episode_steps, loss_info)
            
            # 評価
            if episode % self.eval_interval == 0:
                self.evaluate()
            
            # モデルの保存
            if episode % self.save_interval == 0:
                self.save_models(episode)
        
        # 訓練終了処理
        training_time = time.time() - start_time
        wandb.run.summary['training_time'] = training_time
        wandb.run.summary['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 最終モデルの保存
        self.save_models(self.total_episodes)
        
        # WandBのセッションを終了
        wandb.finish()

if __name__ == "__main__":
    trainer = MultiAgentTrainer(
        config_path="config.yaml",
        project_name="marl-gridworld",
        entity="your-wandb-entity",  # WandBのユーザー名またはチーム名
        num_agents=2,
        total_episodes=10000,
        eval_interval=100,
        save_interval=1000,
        log_interval=10
    )
    trainer.train()