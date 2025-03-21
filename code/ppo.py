from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Dict, List, Tuple

class EpisodeBuffer:
    def __init__(self, capacity: int = 100, sequence_length: int = 32):
        """
        Args:
            capacity: バッファに保存するエピソード数
            sequence_length: 学習時に使用するシーケンスの長さ
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.episodes = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add_episode(self, episode: List[Dict], priority: float = None):
        """エピソードをバッファに追加"""
        self.episodes.append(episode)
        if priority is None:
            priority = max(self.priorities, default=1.0)  # デフォルトプライオリティ
        self.priorities.append(priority)
    
    def sample_sequences(self, batch_size: int) -> List[List[Dict]]:
        """バッチサイズ分のシーケンスをサンプリング"""
        # エピソードの重み付きサンプリング
        probs = np.array(self.priorities) / sum(self.priorities)
        sampled_episodes = random.choices(self.episodes, weights=probs, k=batch_size)
        
        sequences = []
        for episode in sampled_episodes:
            if len(episode) <= self.sequence_length:
                sequences.append(episode)
            else:
                # ランダムな開始位置からシーケンス長分を抽出
                start_idx = random.randint(0, len(episode) - self.sequence_length)
                sequences.append(episode[start_idx:start_idx + self.sequence_length])
        
        return sequences
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """優先度の更新"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.episodes)

class GRUActorCritic(nn.Module):
    def __init__(self, obs_dim: Dict[str, int], action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 観測エンコーダー
        self.terrain_encoder = nn.Sequential(
            nn.Linear(11 * 11, 128),
            nn.ReLU()
        )
        self.objects_encoder = nn.Sequential(
            nn.Linear(11 * 11, 128),
            nn.ReLU()
        )
        self.entities_encoder = nn.Sequential(
            nn.Linear(11 * 11 * 3, 128),
            nn.ReLU()
        )
        
        # 状態情報エンコーダー
        self.state_encoder = nn.Sequential(
            nn.Linear(3 + 2 + 2 + 4, 64),  # stats + inventory + position + direction
            nn.ReLU()
        )
        
        # 特徴量統合
        self.feature_combine = nn.Sequential(
            nn.Linear(128 * 3 + 64, hidden_dim),
            nn.ReLU()
        )
        
        # GRU層
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Actor (方策)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic (価値関数)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs: Dict[str, torch.Tensor], hidden_state: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs['terrain'].size(0)
        
        # 各観測をエンコード
        terrain_feat = self.terrain_encoder(obs['terrain'])
        objects_feat = self.objects_encoder(obs['objects'])
        entities_feat = self.entities_encoder(obs['entities'])
        
        # 状態情報を結合してエンコード
        state = torch.cat([
            obs['stats'],
            obs['inventory'],
            obs['position'],
            obs['direction']
        ], dim=-1)
        state_feat = self.state_encoder(state)
        
        # 全特徴量を統合
        combined_feat = self.feature_combine(
            torch.cat([terrain_feat, objects_feat, entities_feat, state_feat], dim=-1)
        )
        
        # GRU処理のためにシーケンス次元を追加
        if len(combined_feat.shape) == 2:
            combined_feat = combined_feat.unsqueeze(1)
        
        # 初期隠れ状態がない場合は0で初期化
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim, 
                                     device=combined_feat.device)
        
        # GRU処理
        gru_out, new_hidden = self.gru(combined_feat, hidden_state)
        gru_feat = gru_out[:, -1]  # 最後の時刻の出力を使用
        
        # 方策と価値を出力
        action_probs = F.softmax(self.actor(gru_feat), dim=-1)
        value = self.critic(gru_feat)
        
        return action_probs, value, new_hidden

def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def compute_advantages(returns: List[float], values: List[float]) -> List[float]:
    return [r - v for r, v in zip(returns, values)]

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple

class GRUPPOAgentWithTarget:
    def __init__(self, obs_dim: Dict[str, int], action_dim: int,
                 buffer_capacity: int = 100,
                 sequence_length: int = 32,
                 batch_size: int = 8,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 c1: float = 1.0,
                 c2: float = 0.01,
                 target_update_freq: int = 10,
                 tau: float = 0.005):
        # メインネットワーク
        self.actor_critic = GRUActorCritic(obs_dim, action_dim)
        
        # ターゲットネットワーク（Criticのみ）
        self.target_critic = GRUActorCritic(obs_dim, action_dim)
        # ターゲットネットワークの重みを同期
        self.target_critic.load_state_dict(self.actor_critic.state_dict())
        # 勾配計算を無効化
        for param in self.target_critic.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.buffer = EpisodeBuffer(buffer_capacity, sequence_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.hidden_state = None
        self.target_hidden_state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.update_counter = 0
        
        self.actor_critic.to(self.device)
        self.target_critic.to(self.device)
    
    def update_target_network(self, soft_update: bool = True):
        """ターゲットネットワークの更新"""
        if soft_update:
            # Soft update (DDPG style)
            for target_param, param in zip(self.target_critic.parameters(), 
                                         self.actor_critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + 
                    param.data * self.tau
                )
        else:
            # Hard update
            self.target_critic.load_state_dict(self.actor_critic.state_dict())
    
    def compute_target_values(self, next_obs_batch: Dict[str, torch.Tensor], 
                            rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """ターゲットネットワークを使用して目標値を計算"""
        with torch.no_grad():
            # 次の状態の価値をターゲットネットワークで計算
            _, next_values, self.target_hidden_state = self.target_critic(
                next_obs_batch, self.target_hidden_state
            )
            next_values = next_values.squeeze()
            
            # TD(λ)ターゲットの計算
            target_values = rewards + (1 - dones) * self.gamma * next_values
            
        return target_values
    
    def compute_sequence_loss(self, sequence: List[Dict]) -> Tuple[torch.Tensor, tuple]:
        """シーケンスに対する損失を計算（ターゲットネットワークを使用）"""
        # シーケンスデータの準備
        obs_batch = {
            k: torch.stack([torch.FloatTensor(t['obs'][k]) for t in sequence]).to(self.device)
            for k in sequence[0]['obs'].keys()
        }
        next_obs_batch = {
            k: torch.stack([torch.FloatTensor(t['next_obs'][k]) for t in sequence]).to(self.device)
            for k in sequence[0]['next_obs'].keys()
        }
        actions = torch.tensor([t['action'] for t in sequence]).to(self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in sequence]).to(self.device)
        rewards = torch.tensor([t['reward'] for t in sequence]).to(self.device)
        dones = torch.tensor([t['done'] for t in sequence], dtype=torch.float).to(self.device)
        
        # GRUの隠れ状態をリセット
        hidden = None
        self.target_hidden_state = None
        
        # 現在の方策での評価
        action_probs, values, _ = self.actor_critic(obs_batch, hidden)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # ターゲットネットワークを使用して目標値を計算
        target_values = self.compute_target_values(next_obs_batch, rewards, dones)
        
        # Advantages の計算
        advantages = (target_values - values.squeeze()).detach()
        
        # PPOクリップ付き目的関数
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with target network
        value_loss = F.mse_loss(values.squeeze(), target_values)
        
        # 総損失
        total_loss = actor_loss + self.c1 * value_loss - self.c2 * entropy
        
        return total_loss, (actor_loss.item(), value_loss.item(), entropy.item())
    
    def update(self, num_updates: int = 10):
        """経験再生とターゲットネットワークを使用したモデルの更新"""
        if len(self.buffer) < self.batch_size:
            return
        
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(num_updates):
            # バッファからシーケンスをサンプリング
            sequences = self.buffer.sample_sequences(self.batch_size)
            
            # 各シーケンスに対して更新
            for sequence in sequences:
                self.optimizer.zero_grad()
                loss, (actor_loss, value_loss, entropy) = self.compute_sequence_loss(sequence)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss
                total_value_loss += value_loss
                total_entropy += entropy
                
                # ターゲットネットワークの更新
                self.update_counter += 1
                if self.update_counter % self.target_update_freq == 0:
                    self.update_target_network(soft_update=True)
        
        # 平均損失を計算
        num_sequences = num_updates * self.batch_size
        return {
            'actor_loss': total_actor_loss / num_sequences,
            'value_loss': total_value_loss / num_sequences,
            'entropy': total_entropy / num_sequences
        }

def train_agents_with_target(env, num_episodes: int = 1000):
    """ターゲットネットワークを使用した学習ループ"""
    agents = [
        GRUPPOAgentWithTarget(
            obs_dim={k: v.shape[0] for k, v in env.observation_spaces[0].spaces.items()},
            action_dim=env.action_space.n
        ) for _ in range(2)
    ]
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_data = [[] for _ in range(2)]
        
        # エピソード開始時に隠れ状態をリセット
        for agent in agents:
            agent.reset_hidden_state()
        
        while not done:
            actions = []
            for i in range(2):
                obs_tensor = {k: torch.FloatTensor(v[i]) for k, v in obs.items()}
                action, log_prob, value = agents[i].select_action(obs_tensor)
                actions.append(action)
            
            next_obs, rewards, dones, _, _ = env.step(actions)
            
            # 各エージェントの経験を保存
            for i in range(2):
                episode_data[i].append({
                    'obs': {k: torch.FloatTensor(v[i]) for k, v in obs.items()},
                    'next_obs': {k: torch.FloatTensor(v[i]) for k, v in next_obs.items()},
                    'action': actions[i],
                    'reward': rewards[i],
                    'log_prob': log_prob,
                    'value': value,
                    'done': dones[i]
                })
            
            obs = next_obs
            done = any(dones)
        
        # エピソードをバッファに保存
        for i in range(2):
            cumulative_reward = sum(t['reward'] for t in episode_data[i])
            agents[i].store_episode(episode_data[i], priority=abs(cumulative_reward))
        
        # 定期的な更新
        if episode % 5 == 0:
            for i in range(2):
                loss_info = agents[i].update()
                if loss_info and episode % 50 == 0:
                    print(f"Agent {i} Episode {episode} - Losses:", loss_info)