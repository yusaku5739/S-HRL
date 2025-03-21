import gymnasium as gym
import yaml
import numpy as np
import pygame
from enum import Enum
import random
from typing import Dict, List, Tuple, Optional
import math
from collections import deque
from noise import pnoise2
import matplotlib.pyplot as plt

from object import Player, Animal
from each_class import Terrain, Action, GridObject, Direction

class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config, render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config
        self.grid_size = config["env"]["grid_size"]
        self.num_players = config["env"]["num_players"]
        self.num_animals = config["env"]["num_animals"]
        self.scale = config["env"]["scale"]
        self.lava_threshold = config["env"]["lava_threshold"]
        self.land_threshold = config["env"]["land_threshold"]
        self.damage_during_hunger_or_thirst = config["env"]["damage_during_hunger_or_thirst"]
        self.decrement_of_hunger = config["env"]["decrement_of_hunger"]
        self.decrement_of_thirst = config["env"]["decrement_of_thirst"]
        self.recovery_speed = config["env"]["recovery_speed"]
        self.recovery_threshold = config["env"]["recovery_threshold"]
        self.render_mode = render_mode
        self.under_player_objects = {i: GridObject.EMPTY.value for i in range(self.num_players)}
        
        # Observation space is a tuple of:
        # - terrain_map (grid_size x grid_size)
        # - object_map (grid_size x grid_size)
        # - player stats (health, hunger, thirst, inventory)
        # - animal positions and stats
        self.observation_space = gym.spaces.Dict({
            # 各プレイヤーの局所的な観測
            'local_view': gym.spaces.Dict({
                'terrain': gym.spaces.Box(0, 2, shape=(self.num_players, 11, 11), dtype=np.int8),
                'objects': gym.spaces.Box(0, 4, shape=(self.num_players, 11, 11), dtype=np.int8),
                'entities': gym.spaces.Box(0, 1, shape=(self.num_players, 11, 11, 3), dtype=np.float32)  # presence, health, level
            }),
            
            # プレイヤー自身の状態
            'self_state': gym.spaces.Dict({
                'stats': gym.spaces.Box(0, 100, shape=(self.num_players, 3), dtype=np.float32),  # health, hunger, thirst
                'inventory': gym.spaces.Box(0, float('inf'), shape=(self.num_players, 2), dtype=np.int32),
                'position': gym.spaces.Box(0, self.grid_size-1, shape=(self.num_players, 2), dtype=np.int32),
                'direction': gym.spaces.Discrete(4)
            })
        })
        
        self.action_space = gym.spaces.Discrete(len(Action))
        
        # Initialize rendering
        if render_mode == 'human':
            pygame.init()
            self.window_size = 800
            self.cell_size = self.window_size // self.grid_size
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self, seed=0):
        super().reset(seed=seed)
        
        # Initialize terrain
        self.terrain_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._generate_terrain(scale=self.scale, seed=seed, land_threshold=self.land_threshold, lava_threshold=self.lava_threshold)
        """grid = np.zeros_like(self.terrain_map, dtype=np.uint8)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if self.terrain_map[x][y] == Terrain.LAND:
                    grid[x][y] = 1
                elif self.terrain_map[x][y] == Terrain.LAVA:
                    grid[x][y] = 2
                else:
                    grid[x][y] = 0
                
        plt.imshow(grid, vmax=2, vmin=0)
        plt.colorbar()
        plt.show()"""
        # Initialize object map
        self.object_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Initialize players
        self.players = []
        for _ in range(self.num_players):
            pos = self._get_random_land_position()
            player = Player(pos[0], pos[1], config=self.config["player"])
            self.players.append(player)
            self.object_map[pos[0], pos[1]] = GridObject.PLAYER.value
            
        # Initialize animals
        self.animals = []
        for _ in range(self.num_animals):
            pos = self._get_random_land_position()
            animal = Animal(pos[0], pos[1], config=self.config["animal"])
            self.animals.append(animal)
            self.object_map[pos[0], pos[1]] = GridObject.ANIMAL.value
            
        # Initialize blocks (adding 10 random blocks)
        for _ in range(10):
            pos = self._get_random_land_position()
            self.object_map[pos[0], pos[1]] = GridObject.BLOCK.value
        
        observation = self._get_observation()
        info = {}
    
        return observation, info

    def step(self, actions):
        if not isinstance(actions, list):
            actions = [actions]
                
        rewards = np.zeros(self.num_players)
        dones = np.zeros(self.num_players, dtype=bool)
        
        # Process player actions
        for player_idx, action in enumerate(actions):
            player = self.players[player_idx]
            is_done = self._process_player_action(player_idx, Action(action))
            dones[player_idx] = is_done
                
            # Update player status
            player.hunger = max(0, player.hunger - self.decrement_of_hunger)
            player.thirst = max(0, player.thirst - self.decrement_of_thirst)
                
            if player.hunger == 0 or player.thirst == 0:
                player.health -= self.damage_during_hunger_or_thirst
                    
            if player.hunger > self.recovery_threshold and player.thirst > self.recovery_threshold:
                player.health = min(100, player.health + self.recovery_speed)
                    
            if player.health <= 0:
                dones[player_idx] = True
            
            xtt_health, xtt_hunger, xtt_thirst = player.calculate_introception(dones[player_idx])
            reward = player.calculate_reward(xtt_health, xtt_hunger, xtt_thirst)
            rewards[player_idx] = reward
        
        # Update animals
        self._update_animals()
        
        # Spawn new animals if needed
        self._maintain_animal_population()
        
        observation = self._get_observation()
        info = {}
        
        return observation, rewards, dones, False, info

    def _get_observation(self):
        observations = {
            'local_view': {
                'terrain': np.zeros((self.num_players, 11, 11), dtype=np.int8),
                'objects': np.zeros((self.num_players, 11, 11), dtype=np.int8),
                'entities': np.zeros((self.num_players, 11, 11, 3), dtype=np.float32)
            },
            'self_state': {
                'stats': np.zeros((self.num_players, 3), dtype=np.float32),
                'inventory': np.zeros((self.num_players, 2), dtype=np.int32),
                'position': np.zeros((self.num_players, 2), dtype=np.int32),
                'direction': np.zeros(self.num_players, dtype=np.int32)
            }
        }
        
        # 各プレイヤーの観測を生成
        for player_idx, player in enumerate(self.players):
            # 局所的な観測を生成
            view_area = player.get_view_area()
            local_view = self._get_local_view(player, view_area)
            observations['local_view']['terrain'][player_idx] = local_view['terrain']
            observations['local_view']['objects'][player_idx] = local_view['objects']
            observations['local_view']['entities'][player_idx] = local_view['entities']
            
            # プレイヤー自身の状態
            observations['self_state']['stats'][player_idx] = [
                player.health,
                player.hunger,
                player.thirst
            ]
            observations['self_state']['inventory'][player_idx] = [
                player.inventory['block'],
                player.inventory['meat']
            ]
            observations['self_state']['position'][player_idx] = [player.x, player.y]
            observations['self_state']['direction'][player_idx] = player.direction.value
        
        return observations

    def _get_entity_info(self, x: int, y: int) -> Optional[np.ndarray]:
        """指定座標のエンティティ情報を取得"""
        # プレイヤーの確認
        for player in self.players:
            if player.x == x and player.y == y:
                return np.array([1.0, player.health / 100, 0.0])
        
        # 動物の確認
        for animal in self.animals:
            if animal.x == x and animal.y == y:
                return np.array([1.0, animal.health / 100, animal.level / 5])
        
        return None
    
    def _get_local_view(self, player: Player, view_area: List[Tuple[int, int]]) -> Dict:
        """プレイヤーの局所的な観測を生成"""
        local_view = {
            'terrain': np.zeros((11, 11), dtype=np.int8),
            'objects': np.zeros((11, 11), dtype=np.int8),
            'entities': np.zeros((11, 11, 3), dtype=np.float32)
        }
        
        center_x, center_y = 5, 5
        for vx, vy in view_area:
            if 0 <= vx < self.grid_size and 0 <= vy < self.grid_size:
                # 相対座標に変換
                rel_x = vx - player.x + center_x
                rel_y = vy - player.y + center_y
                
                if 0 <= rel_x < 11 and 0 <= rel_y < 11:
                    local_view['terrain'][rel_x, rel_y] = self.terrain_map[vx, vy]
                    local_view['objects'][rel_x, rel_y] = self.object_map[vx, vy]
                    
                    # エンティティ情報の追加
                    entity_info = self._get_entity_info(vx, vy)
                    if entity_info is not None:
                        local_view['entities'][rel_x, rel_y] = entity_info
        
        return local_view

    """def _generate_terrain(self):
        # Simple terrain generation using noise
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                r = random.random()
                if r < 0.2:
                    self.terrain_map[x, y] = Terrain.WATER.value
                elif r < 0.95:
                    self.terrain_map[x, y] = Terrain.LAND.value
                else:
                    self.terrain_map[x, y] = Terrain.LAVA.value"""
    
    def _generate_terrain(self, 
                             scale, 
                             seed, 
                             land_threshold,
                             lava_threshold,
                             min_water_size=5, 
                             water_prob=0.7,):
        """
        Perlinノイズを使用して「海」「陸」「溶岩」を含むマップを生成。
        """
        random.seed(seed)
        base_map = [[pnoise2(x / scale, y / scale, octaves=4, base=seed)
                    for x in range(self.grid_size)] for y in range(self.grid_size)]

        # Perlinノイズ値を「海」「陸」「溶岩」に変換
        map_grid = []
        for row in base_map:
            new_row = []
            for value in row:
                if value < land_threshold:
                    new_row.append(Terrain.LAND.value)  
                else:
                    new_row.append(Terrain.WATER.value)   
            map_grid.append(new_row)
        
        self.make_lava(map_grid, water_prob)
        
        self.terrain_map = np.array(map_grid)
        
    def make_lava(self, map_grid, water_prob):
        if not map_grid or not map_grid[0]:
            return 0
        W = self.grid_size
        H = self.grid_size
        visited = [[False] * W for _ in range(H)]
        
        # 8方向の移動ベクトル
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        def is_valid(x, y):
            return 0 <= x < H and 0 <= y < W
        
        def bfs(start_x, start_y, is_lava):
            queue = deque([(start_x, start_y)])
            visited[start_x][start_y] = True
            
            while queue:
                curr_x, curr_y = queue.popleft()
                
                # 8方向を探索
                for dx, dy in directions:
                    next_x = curr_x + dx
                    next_y = curr_y + dy
                    
                    # 範囲内で、未訪問の水たまりマスを探索
                    if (is_valid(next_x, next_y) and 
                        not visited[next_x][next_y] and 
                        map_grid[next_x][next_y] == Terrain.WATER.value):

                        if is_lava: map_grid[next_x][next_y] = Terrain.LAVA.value
                        visited[next_x][next_y] = True
                        queue.append((next_x, next_y))
    
        # 全マスを探索
        for i in range(H):
            for j in range(W):
                if not visited[i][j] and map_grid[i][j] == Terrain.WATER.value:
                    if random.random() < water_prob:
                        bfs(i, j, is_lava=False)
                    else:
                        bfs(i, j, is_lava=True)

    def _get_random_land_position(self) -> Tuple[int, int]:
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (self.terrain_map[x, y] == Terrain.LAND.value and 
                self.object_map[x, y] == GridObject.EMPTY.value):
                return x, y

    def _process_player_action(self, player_idx: int, action: Action) -> Tuple[float, bool]:
        player = self.players[player_idx]
        
        if action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            # 移動処理
            dx, dy = 0, 0
            if action == Action.UP:
                dx = -1
                player.direction = Direction.UP
            elif action == Action.DOWN:
                dx = 1
                player.direction = Direction.DOWN
            elif action == Action.LEFT:
                dy = -1
                player.direction = Direction.LEFT
            elif action == Action.RIGHT:
                dy = 1
                player.direction = Direction.RIGHT
                
            new_x = player.x + dx
            new_y = player.y + dy
            
            # 移動先の有効性チェック
            if player.can_move_to(new_x, new_y, self.terrain_map, self.object_map):
                # 移動前の位置のオブジェクトを更新
                self.object_map[player.x, player.y] = GridObject.EMPTY.value
                
                # プレイヤーを移動
                player.x = new_x
                player.y = new_y
                
                # 移動先のオブジェクトを更新
                if self.object_map[new_x, new_y] == GridObject.MEAT.value:
                    player.inventory['meat'] += 1  # 肉の上を通過すると自動で拾う
                self.object_map[new_x, new_y] = GridObject.PLAYER.value
                
            # 溶岩による即死判定
            elif (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and 
                self.terrain_map[new_x, new_y] == Terrain.LAVA.value):
                player.health = 0
                return True
        
        elif action == Action.EAT:
            if player.inventory['meat'] > 0:
                player.inventory['meat'] -= 1
                player.hunger = min(100, player.hunger + 30)
            
        elif action == Action.DRINK:
            # Check if adjacent to water
            front_x, front_y = player.get_front_position()
            if (0 <= front_x < self.grid_size and 
                0 <= front_y < self.grid_size and 
                self.terrain_map[front_x, front_y] == Terrain.WATER.value):
                player.thirst = min(100, player.thirst + 30)
                    
        elif action == Action.ATTACK:
            # Check for animals in front
            front_x, front_y = player.get_front_position()
            if (0 <= front_x < self.grid_size and 
                0 <= front_y < self.grid_size):
                for animal in self.animals:
                    if animal.x == front_x and animal.y == front_y:
                        animal.health -= player.attack_power
                        animal.is_aggressive = True
                        animal.target_player = player
                        if animal.health <= 0:
                            self.animals.remove(animal)
                            self.object_map[front_x, front_y] = GridObject.EMPTY.value
                            player.inventory['meat'] += animal.meat_amount
                        break
        
        elif action == Action.PLACE_MEAT:
            if player.inventory['meat'] > 0:
                front_x, front_y = player.get_front_position()
                if player.can_place_object(front_x, front_y, self.terrain_map, self.object_map, GridObject.MEAT):
                    self.object_map[front_x, front_y] = GridObject.MEAT.value
                    player.inventory['meat'] -= 1
        
        elif action == Action.PLACE_BLOCK:
            if player.inventory['block'] > 0:
                front_x, front_y = player.get_front_position()
                if player.can_place_object(front_x, front_y, self.terrain_map, self.object_map, GridObject.BLOCK):
                    self.object_map[front_x, front_y] = GridObject.BLOCK.value
                    player.inventory['block'] -= 1
        
        elif action == Action.PICKUP:
            front_x, front_y = player.get_front_position()
            if player.can_pickup_object(front_x, front_y, self.object_map):
                obj = self.object_map[front_x, front_y]
                if obj == GridObject.BLOCK.value:
                    player.inventory['block'] += 1
                elif obj == GridObject.MEAT.value:
                    player.inventory['meat'] += 1
                self.object_map[front_x, front_y] = GridObject.EMPTY.value
        
        return player.health <= 0

    def _update_animals(self):
        for animal in self.animals:
            old_x, old_y = animal.x, animal.y
            animal.update(self.players, self.terrain_map, self.object_map)
            
            # Update object map
            if (old_x != animal.x or old_y != animal.y):
                self.object_map[old_x, old_y] = GridObject.EMPTY.value
                self.object_map[animal.x, animal.y] = GridObject.ANIMAL.value
            
            # Check for attacks on players
            for player in self.players:
                if (abs(animal.x - player.x) <= 1 and 
                    abs(animal.y - player.y) <= 1 and 
                    animal.is_aggressive):
                    player.health -= animal.attack_power

    def _maintain_animal_population(self):
        while len(self.animals) < self.num_animals // 2:
            pos = self._get_random_land_position()
            animal = Animal(pos[0], pos[1], config=self.config["animal"])
            self.animals.append(animal)
            self.object_map[pos[0], pos[1]] = GridObject.ANIMAL.value

    def render(self):
        if self.render_mode == "human":
            self.window.fill((255, 255, 255))
            
            # Draw terrain
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = (
                        y * self.cell_size,  # left
                        x * self.cell_size,  # top
                        self.cell_size,      # width
                        self.cell_size       # height
                    )
                    
                    color = (200, 200, 200)  # Default gray for land
                    if self.terrain_map[x, y] == Terrain.WATER.value:
                        color = (0, 0, 255)   # Blue for water
                    elif self.terrain_map[x, y] == Terrain.LAVA.value:
                        color = (255, 69, 0)  # Red-orange for lava
                        
                    pygame.draw.rect(self.window, color, rect)
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 1)
            
            # Draw objects and players
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    rect = (
                        y * self.cell_size,  # left
                        x * self.cell_size,  # top
                        self.cell_size,      # width
                        self.cell_size       # height
                    )
                    
                    if self.object_map[x, y] == GridObject.PLAYER.value:
                        # プレイヤーの円を描画
                        center_x = y * self.cell_size + self.cell_size // 2
                        center_y = x * self.cell_size + self.cell_size // 2
                        radius = self.cell_size // 3
                        pygame.draw.circle(
                            self.window,
                            (0, 255, 0),  # Green for player
                            (center_x, center_y),
                            radius
                        )
                        
                        # プレイヤーの向きを示す三角形を描画
                        for player in self.players:
                            if player.x == x and player.y == y:
                                direction_points = []
                                if player.direction == Direction.UP:
                                    direction_points = [
                                        (center_x, center_y - radius),
                                        (center_x - radius//2, center_y + radius//2),
                                        (center_x + radius//2, center_y + radius//2)
                                    ]
                                elif player.direction == Direction.DOWN:
                                    direction_points = [
                                        (center_x, center_y + radius),
                                        (center_x - radius//2, center_y - radius//2),
                                        (center_x + radius//2, center_y - radius//2)
                                    ]
                                elif player.direction == Direction.LEFT:
                                    direction_points = [
                                        (center_x - radius, center_y),
                                        (center_x + radius//2, center_y - radius//2),
                                        (center_x + radius//2, center_y + radius//2)
                                    ]
                                elif player.direction == Direction.RIGHT:
                                    direction_points = [
                                        (center_x + radius, center_y),
                                        (center_x - radius//2, center_y - radius//2),
                                        (center_x - radius//2, center_y + radius//2)
                                    ]
                                pygame.draw.polygon(self.window, (0, 100, 0), direction_points)
                                
                    elif self.object_map[x, y] == GridObject.ANIMAL.value:
                        pygame.draw.circle(
                            self.window,
                            (255, 0, 0),  # Red for animal
                            (y * self.cell_size + self.cell_size // 2,
                            x * self.cell_size + self.cell_size // 2),
                            self.cell_size // 3
                        )
                    elif self.object_map[x, y] == GridObject.BLOCK.value:
                        pygame.draw.rect(
                            self.window,
                            (139, 69, 19),  # Brown for block
                            rect
                        )
                    elif self.object_map[x, y] == GridObject.MEAT.value:
                        pygame.draw.rect(
                            self.window,
                            (255, 192, 203),  # Pink for meat
                            rect
                        )
            
            # Draw player stats
            font = pygame.font.Font(None, 24)
            for i, player in enumerate(self.players):
                stats_text = f"Player {i + 1}: HP:{player.health:.1f} Hunger:{player.hunger:.1f} Thirst:{player.thirst:.1f}"
                inventory_text = f"Block:{player.inventory['block']} Meat:{player.inventory['meat']}"
                text_surface = font.render(stats_text, True, (0, 0, 0))
                inv_surface = font.render(inventory_text, True, (0, 0, 0))
                self.window.blit(text_surface, (10, 10 + i * 50))
                self.window.blit(inv_surface, (10, 30 + i * 50))
            
            # 視界範囲の可視化
            for player in self.players:
                view_area = player.get_view_area()
                for vx, vy in view_area:
                    if 0 <= vx < self.grid_size and 0 <= vy < self.grid_size:
                        # 視界範囲を半透明の黄色で表示
                        s = pygame.Surface((self.cell_size, self.cell_size))
                        s.set_alpha(64)
                        s.fill((255, 255, 0))
                        self.window.blit(s, (vy * self.cell_size, vx * self.cell_size))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict

class RLWrapper(gym.Wrapper):
    """Wrapper for the GridWorld environment to make it more suitable for RL."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Flatten and normalize the observation space
        self.observation_spaces = {
            i: gym.spaces.Dict({
                'terrain': gym.spaces.Box(0, 1, shape=(11 * 11,), dtype=np.float32),
                'objects': gym.spaces.Box(0, 1, shape=(11 * 11,), dtype=np.float32),
                'entities': gym.spaces.Box(0, 1, shape=(11 * 11 * 3,), dtype=np.float32),
                'stats': gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32),
                'inventory': gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32),
                'position': gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32),
                'direction': gym.spaces.Box(0, 1, shape=(4,), dtype=np.float32)
            }) for i in range(self.num_players)
        }

    def _normalize_observation(self, obs: Dict, agent_id: int) -> Dict:
        """Normalize and flatten the observation for a specific agent."""
        # Flatten and normalize terrain
        terrain = obs['local_view']['terrain'][agent_id].ravel() / 2.0
        
        # Flatten and normalize objects
        objects = obs['local_view']['objects'][agent_id].ravel() / 4.0
        
        # Flatten and normalize entities
        entities = obs['local_view']['entities'][agent_id].reshape(-1)
        
        # Normalize stats
        stats = obs['self_state']['stats'][agent_id] / 100.0
        
        # Normalize inventory (assuming max capacity of 10)
        inventory = np.array(obs['self_state']['inventory'][agent_id], dtype=np.float32) / 10.0
        
        # Normalize position
        position = obs['self_state']['position'][agent_id] / float(self.env.grid_size)
        
        # One-hot encode direction
        direction = np.zeros(4, dtype=np.float32)
        direction[obs['self_state']['direction'][agent_id]] = 1.0
        
        return {
            'terrain': terrain,
            'objects': objects,
            'entities': entities,
            'stats': stats,
            'inventory': inventory,
            'position': position,
            'direction': direction
        }

    def step(self, actions):
        """Override step to handle single action and normalize observation."""
        obs, reward, done, truncated, info = self.env.step(actions)

        processed_obs = {}
        for agent_id in range(self.num_players): 
            processed_obs[agent_id] = self._normalize_observation(obs, agent_id)

        return processed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Override reset to normalize the initial observation."""
        obs, info = self.env.reset(**kwargs)
        processed_obs = {}
        for agent_id in range(self.num_players): 
            processed_obs[agent_id] = self._normalize_observation(obs, agent_id)
        return processed_obs, info

class KeyboardPlayerInterface:
    def __init__(self, env: GridWorldEnv):
        self.env = env
        self.running = True
        
    def run(self):
        """
        Run the environment with keyboard controls for the first player
        Keys:
        - Arrow keys: Movement
        - E: Eat meat
        - D: Drink water
        - A: Attack
        - M: Place meat
        - B: Place block
        - P: Pickup
        - Q: Quit
        """
        self.env.reset()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                
                if event.type == pygame.KEYDOWN:
                    action = None
                    
                    if event.key == pygame.K_UP:
                        action = Action.UP
                    elif event.key == pygame.K_DOWN:
                        action = Action.DOWN
                    elif event.key == pygame.K_LEFT:
                        action = Action.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Action.RIGHT
                    elif event.key == pygame.K_e:
                        action = Action.EAT
                    elif event.key == pygame.K_d:
                        action = Action.DRINK
                    elif event.key == pygame.K_a:
                        action = Action.ATTACK
                    elif event.key == pygame.K_m:
                        action = Action.PLACE_MEAT
                    elif event.key == pygame.K_b:
                        action = Action.PLACE_BLOCK
                    elif event.key == pygame.K_p:
                        action = Action.PICKUP
                    elif event.key == pygame.K_q:
                        self.running = False
                        break
                    
                    if action is not None:
                        # For multiplayer, create a list of actions where only the first player
                        # is controlled by keyboard, and others perform NOOP
                        actions = [action.value] + [Action.NOOP.value] * (self.env.num_players - 1)
                        obs, rewards, dones, truncated, info = self.env.step(actions)
                        self.env.render()
                        
                        if dones[0]:  # If first player is done
                            print("Game Over!")
                            self.running = False
                            break
        
        self.env.close()

# Example usage:
if __name__ == "__main__":
    with open("config.yaml", "rb") as f:
        config = yaml.safe_load(f)
        
    env = GridWorldEnv(config)
    obs, _ = env.reset()

    # 観測の構造を確認
    print("Local view shape (terrain):", obs['local_view']['terrain'].shape)  # (2, 11, 11)
    print("Local view shape (entities):", obs['local_view']['entities'].shape)  # (2, 11, 11, 3)
    print("Self state shape (stats):", obs['self_state']['stats'].shape)  # (2, 3)
    print(obs)