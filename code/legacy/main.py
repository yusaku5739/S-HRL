import numpy as np
from enum import Enum, auto
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from noise import pnoise2  # Perlinノイズ生成用
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from collections import deque
import pygame
import sys



class TerrainType(Enum):
    WATER = 0
    LAND = 1
    LAVA = 2

@dataclass(frozen=True)  # frozen=True でイミュータブルにする
class Position:
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
        
    def is_adjacent(self, other) -> bool:
        """別の位置が隣接しているかどうかを確認（上下左右）"""
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
@dataclass
class Item:
    """アイテムを表すクラス"""
    name: str
    amount: int
    healing_value: float  # 回復量
    
class BlockType(Enum):
    BLOCK = auto()  # シンプルな単一ブロック

@dataclass
class Block:
    """ブロックの情報を保持するクラス"""
    position: Position
    is_bridge: bool 

class Inventory:
    """インベントリを管理するクラス"""
    def __init__(self):
        self.items: Dict[str, Item] = {}
    
    def add_item(self, name: str, amount: int, healing_value: float):
        if name in self.items:
            self.items[name].amount += amount
        else:
            self.items[name] = Item(name, amount, healing_value)
    
    def consume_item(self, name: str, amount: int = 1) -> Optional[Item]:
        if name in self.items and self.items[name].amount >= amount:
            item = self.items[name]
            item.amount -= amount
            if item.amount <= 0:
                del self.items[name]
            return Item(name, amount, item.healing_value)
        return None

class Animal:
    def __init__(self, position: Position, health: float = 50.0, level:int = 1):
        self.position = position
        self.level = level
        
        # レベルに応じたステータス計算
        self.max_health = 50.0 + (level - 1) * 20.0
        self.health = self.max_health
        self.attack_power = 5.0 + (level - 1) * 2.0
        
        # ドロップアイテムの設定
        self.meat_drop = {
            "amount": level * 2,  # レベルに応じて増加
            "healing_value": 10.0 + (level - 1) * 5.0  # レベルに応じて回復量増加
        }
        
        self.color = self._calculate_color()
    
    def _calculate_color(self) -> str:
        """レベルに応じた色を計算（HTML色コード）"""
        # レベルが上がるほど赤色が濃くなる
        red = min(255, 100 + self.level * 30)
        return f"#{red:02x}4040"
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def take_damage(self, damage: float):
        """ダメージを受ける"""
        self.health = max(0.0, self.health - damage)
    
    def random_move(self, valid_positions: List[Position]) -> None:
        """隣接する有効な位置にランダムに移動"""
        if valid_positions:
            self.position = np.random.choice(valid_positions)
    
    @property
    def stats(self) -> Dict:
        """現在のステータスを取得"""
        return {
            'level': self.level,
            'health': self.health,
            'max_health': self.max_health,
            'attack_power': self.attack_power,
            'color': self.color
        }

class Agent:
    def __init__(self, position: Position, health: float = 100.0, 
                 hunger: float = 100.0, thirst: float = 100.0):
        self.position = position
        self.health = health
        self.max_health = health
        self.hunger = hunger
        self.thirst = thirst
        self.attack_damage = 10.0  # 攻撃力を追加
        self.block_count = 0
        
        self.inventory = Inventory()
        
        # 自然回復のしきい値
        self.healing_threshold = 70.0  # この値以上のhungerとthirstで回復
        self.healing_rate = 0.5  # 1ステップあたりの回復量
        
    def is_alive(self) -> bool:
        return self.health > 0
    
    def attack(self, animal: 'Animal') -> None:
        """動物を攻撃"""
        animal.take_damage(self.attack_damage)
    
    def take_damage(self, damage: float):
        """動物からダメージを受ける"""
        self.health = max(0.0, min(self.max_health, self.health - damage))
        
    def update_status(self):
        if self.hunger <= 0 or self.thirst <= 0:
            self.health -= 1.0
        elif self.hunger >= self.healing_threshold and self.thirst >= self.healing_threshold:
            # 体力が最大値未満の場合のみ回復
            if self.health < self.max_health:
                self.health = min(self.max_health, self.health + self.healing_rate)
            
        self.health = max(0.0, min(100.0, self.health))
        self.hunger = max(0.0, min(100.0, self.hunger))
        self.thirst = max(0.0, min(100.0, self.thirst))
    
    def eat_meat(self) -> bool:
        """肉を食べて空腹度を回復"""
        meat = self.inventory.consume_item("meat")
        if meat:
            self.hunger = min(100.0, self.hunger + meat.healing_value)
            return True
        return False
    
    def add_blocks(self, amount: int = 1):
        """ブロックを追加"""
        self.block_count += amount
    
    def use_block(self) -> bool:
        """ブロックを使用"""
        if self.block_count > 0:
            self.block_count -= 1
            return True
        return False
    
    def get_inventory_state(self) -> Dict:
        """インベントリの状態を取得"""
        return {
            'blocks': self.block_count,
            **{name: {"amount": item.amount, "healing_value": item.healing_value}
               for name, item in self.inventory.items.items()}
        }

class GridWorld:
    def __init__(self, width: int, height: int, target_animal_count=5, max_animal_level=5):
        self.width = width
        self.height = height
        #self.grid = np.full((height, width), TerrainType.LAND)
        self.agents: List[Agent] = []
        self.animals: List[Animal] = []
        self.target_animal_count = target_animal_count
        self.max_animal_level = max_animal_level
        self.blocks: Dict[Position, Block] = {}

        
    def generate_terrain_map(self, 
                             scale, 
                             seed, 
                             land_threshold,
                             lava_threshold,
                             min_water_size=5, 
                             water_prob=0.7,
                             block_probability: float = 0.01,
                            initial_animals: int = None,):
        """k
        Perlinノイズを使用して「海」「陸」「溶岩」を含むマップを生成。
        """
        random.seed(seed)
        base_map = [[pnoise2(x / scale, y / scale, octaves=4, base=seed)
                    for x in range(self.width)] for y in range(self.height)]

        # Perlinノイズ値を「海」「陸」「溶岩」に変換
        map_grid = []
        for row in base_map:
            new_row = []
            for value in row:
                if value < land_threshold:
                    new_row.append(TerrainType.LAND)  
                else:
                    new_row.append(TerrainType.WATER)   
            map_grid.append(new_row)
        
        self.make_lava(map_grid, water_prob)
        
        self.grid = np.array(map_grid, dtype=TerrainType)
        
        self._generate_random_blocks(block_probability)

        # 動物の生成
        initial_count = initial_animals if initial_animals is not None else self.target_animal_count
        self._generate_random_animals(initial_count)
    
    def make_lava(self, map_grid, water_prob):
        if not map_grid or not map_grid[0]:
            return 0
        W = self.width
        H = self.height   
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
                        map_grid[next_x][next_y] == TerrainType.WATER):

                        if is_lava: map_grid[next_x][next_y] = TerrainType.LAVA
                        visited[next_x][next_y] = True
                        queue.append((next_x, next_y))
    
        # 全マスを探索
        for i in range(H):
            for j in range(W):
                if not visited[i][j] and map_grid[i][j] == TerrainType.WATER:
                    if random.random() < water_prob:
                        bfs(i, j, is_lava=False)
                    else:
                        bfs(i, j, is_lava=True)

    def _generate_random_blocks(self, probability: float):
        """陸地にランダムにブロックを配置"""
        for y in range(self.height):
            for x in range(self.width):
                if (self.grid[y][x] == TerrainType.LAND and  # 陸地のみ
                    np.random.random() < probability):        # 確率判定
                    position = Position(x, y)
                    # エージェントや動物がいない場所にのみ配置
                    if not any(agent.position == position for agent in self.agents) and \
                    not any(animal.position == position for animal in self.animals):
                        self.place_block(position)
                        
    def _generate_random_animals(self, count: int):
        """指定された数の動物をランダムな位置に生成"""
        valid_positions = self._get_valid_spawn_positions_for_animals()
        
        # 有効な位置が十分にある場合のみ動物を生成
        if len(valid_positions) >= count:
            # ランダムに位置を選択（重複なし）
            spawn_positions = np.random.choice(
                len(valid_positions), 
                size=count, 
                replace=False
            )
            
            for idx in spawn_positions:
                position = valid_positions[idx]
                # レベルをランダムに決定
                level = np.random.randint(1, self.max_animal_level + 1)
                animal = Animal(position, level)
                self.animals.append(animal)

    def _get_valid_spawn_positions_for_animals(self) -> List[Position]:
        """動物のスポーンに有効な位置のリストを取得"""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                pos = Position(x, y)
                # 以下の条件をすべて満たす位置のみを有効とする：
                # 1. 陸地である
                # 2. ブロックが置かれていない
                # 3. 他のエージェントがいない
                # 4. 他の動物がいない
                if (self.grid[y][x] == TerrainType.LAND and
                    pos not in self.blocks and
                    not any(agent.position == pos for agent in self.agents) and
                    not any(animal.position == pos for animal in self.animals)):
                    valid_positions.append(pos)
        return valid_positions

    def get_state(self) -> Dict:
        """環境の現在の状態を取得"""
        return {
            'grid': self.grid.copy(),
            'agents': [(agent.position, agent.health, agent.hunger, agent.thirst, 
                    agent.get_inventory_state()) 
                    for agent in self.agents],
            'animals': [(animal.position, animal.level, animal.health, 
                        animal.max_health, animal.attack_power) 
                    for animal in self.animals],
            'blocks': [(pos, block.is_bridge) 
                    for pos, block in self.blocks.items()]
        }
    
    def get_valid_spawn_positions(self) -> List[Position]:
        """エージェントのスポーンに適した位置（陸地）のリストを取得"""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == TerrainType.LAND:
                    valid_positions.append(Position(x, y))
        return valid_positions

    def add_agent(self, position: Position = None) -> Agent:
        """新しいエージェントを環境に追加。位置が指定されない場合は適切な場所にランダム配置"""
        if position is None:
            valid_positions = self.get_valid_spawn_positions()
            if not valid_positions:
                raise ValueError("No valid spawn positions available")
            position = np.random.choice(valid_positions)
        
        agent = Agent(position)
        self.agents.append(agent)
        return agent
        
    def set_terrain(self, position: Position, terrain: TerrainType):
        if 0 <= position.x < self.width and 0 <= position.y < self.height:
            self.grid[position.y, position.x] = terrain
            
    def get_terrain(self, position: Position) -> TerrainType:
        if 0 <= position.x < self.width and 0 <= position.y < self.height:
            return self.grid[position.y, position.x]
        return None
    
    def get_adjacent_positions(self, position: Position) -> List[Position]:
        """指定された位置に隣接する有効な位置のリストを取得"""
        adjacent = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 上下左右
            new_pos = Position(position.x + dx, position.y + dy)
            if self.is_valid_position(new_pos) and \
               self.get_terrain(new_pos) != TerrainType.LAVA:  # 溶岩は避ける
                adjacent.append(new_pos)
        return adjacent

    def spawn_animal(self) -> None:
        """新しい動物をランダムな位置にスポーン（衝突判定付き）"""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                pos = Position(x, y)
                if (self.get_terrain(pos) == TerrainType.LAND and 
                    not self.is_occupied(pos)):
                    valid_positions.append(pos)
        
        if valid_positions:
            position = np.random.choice(valid_positions)
            level = np.random.randint(1, self.max_animal_level + 1)
            animal = Animal(position, level)
            self.animals.append(animal)
    
    def get_animal_distribution(self) -> Dict[int, int]:
        """現在の動物のレベル分布を取得"""
        distribution = {}
        for animal in self.animals:
            distribution[animal.level] = distribution.get(animal.level, 0) + 1
        return distribution

    def maintain_animal_population(self) -> None:
        """動物の数が目標値を下回っている場合、新しい動物をスポーン"""
        while len(self.animals) < self.target_animal_count:
            self.spawn_animal()

    def get_animal_at_position(self, position: Position) -> Optional[Animal]:
        """指定された位置にいる動物を取得"""
        for animal in self.animals:
            if animal.position == position:
                return animal
        return None

    def attack_animal(self, agent: Agent, position: Position) -> bool:
        """エージェントが指定された位置にいる動物を攻撃"""
        animal = self.get_animal_at_position(position)
        if animal and agent.position.is_adjacent(position):
            agent.attack(animal)
            
            # 動物が倒された場合、ドロップアイテムを処理
            if not animal.is_alive():
                agent.inventory.add_item(
                    "meat",
                    animal.meat_drop["amount"],
                    animal.meat_drop["healing_value"]
                )
            return True
        return False
    
    def is_occupied(self, position: Position) -> bool:
        """指定された位置が何かのエンティティによって占有されているかチェック"""
        # 他のエージェントがいるかチェック
        for agent in self.agents:
            if agent.position == position:
                return True
                
        # 動物がいるかチェック
        for animal in self.animals:
            if animal.position == position:
                return True
                
        # ブロックがあるかチェック（ただし、水や溶岩上のブロックは通過可能）
        if position in self.blocks:
            terrain = self.get_terrain(position)
            if terrain == TerrainType.LAND:  # 陸地上のブロックは通過不可
                return True
                
        return False
    
    def place_block(self, position: Position) -> bool:
        """ブロックの配置（衝突判定付き）"""
        # 既に他のエンティティが存在する場合は配置不可
        if self.is_occupied(position):
            return False
            
        # 有効な位置かチェック
        if not self.is_valid_position(position):
            return False
        
        # 地形に応じてブロックの種類を決定
        terrain = self.get_terrain(position)
        is_bridge = terrain in [TerrainType.WATER, TerrainType.LAVA]
        
        self.blocks[position] = Block(position, is_bridge)
        return True

    def remove_block(self, position: Position) -> bool:
        """指定位置のブロックを除去"""
        if position in self.blocks:
            del self.blocks[position]
            return True
        return False
    
    def is_passable(self, position: Position) -> bool:
        """指定位置が通行可能かどうかを判定"""
        if not self.is_valid_position(position):
            return False
            
        terrain = self.get_terrain(position)
        
        # 水や溶岩は、ブロックが置かれていない限り通行不可
        if terrain in [TerrainType.WATER, TerrainType.LAVA]:
            return position in self.blocks
        
        # 陸地の場合、他のエンティティがいなければ通行可能
        return not self.is_occupied(position)
    
    def move_animal(self, animal: Animal) -> None:
        """動物の移動（衝突判定付き）"""
        possible_moves = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 上下左右
            new_pos = Position(animal.position.x + dx, animal.position.y + dy)
            if self.is_passable(new_pos) and self.get_terrain(new_pos) == TerrainType.LAND:
                possible_moves.append(new_pos)
        
        if possible_moves:
            animal.position = np.random.choice(possible_moves)
        

    def step(self):
        """環境を1ステップ進める"""
        # エージェントの更新
        for agent in self.agents:
            agent.hunger -= 0.5
            agent.thirst -= 1.0
            
            terrain = self.get_terrain(agent.position)
            if terrain == TerrainType.WATER:
                agent.thirst = 100.0
            elif terrain == TerrainType.LAVA:
                agent.health -= 10.0
                
            agent.update_status()
        
        # 動物の更新
        for animal in self.animals:
            # ランダムな移動
            valid_moves = self.get_adjacent_positions(animal.position)
            animal.random_move(valid_moves)
        
        # 死亡した個体の除去
        self.agents = [agent for agent in self.agents if agent.is_alive()]
        self.animals = [animal for animal in self.animals if animal.is_alive()]
        
        # 動物の数を維持
        self.maintain_animal_population()

    def get_state(self) -> Dict:
        """環境の現在の状態を取得"""
        return {
            'grid': self.grid.copy(),
            'agents': [(agent.position, agent.health, agent.hunger, agent.thirst, 
                       agent.get_inventory_state()) 
                      for agent in self.agents],
            'animals': [(animal.position, animal.stats) 
                       for animal in self.animals],
            'blocks': [(pos, block.is_bridge) 
                      for pos, block in self.blocks.items()]
        }

    def move_agent(self, agent: Agent, dx: int, dy: int) -> bool:
        """エージェントの移動（衝突判定付き）"""
        new_pos = Position(agent.position.x + dx, agent.position.y + dy)
        if self.is_passable(new_pos):
            agent.position = new_pos
            return True
        return False


    def is_valid_position(self, position: Position) -> bool:
        return (0 <= position.x < self.width and 
                0 <= position.y < self.height)

    def render(self, save_path: str = None, show: bool = True):
        """
        環境をmatplotlibを使用して描画
        
        Args:
            save_path: 画像を保存するパス（オプション）
            show: プロットを表示するかどうか
        """
        # プロットのサイズ設定
        plt.figure(figsize=(10, 10))
        
        # 地形の色マップ作成
        terrain_colors = ['#4A90E2', '#90EE90', '#FF6B6B']  # 水、陸地、溶岩の色
        terrain_cmap = ListedColormap(terrain_colors)
        
        # 地形の描画
        terrain_data = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                terrain_data[y, x] = self.grid[y, x].value
        
        plt.imshow(terrain_data, cmap=terrain_cmap)
        
        # グリッド線の描画
        plt.grid(True, color='black', alpha=0.2)
        plt.xticks(np.arange(-0.5, self.width, 1), [])
        plt.yticks(np.arange(-0.5, self.height, 1), [])
        
        # ブロックの描画
        for pos, block in self.blocks.items():
            if block.is_bridge:
                color = '#FFFFFF80'  # 半透明の白（足場）
                marker = 's'
            else:
                color = '#8B4513'  # 茶色（壁）
                marker = 's'
            plt.plot(pos.x, pos.y, marker=marker, color=color, 
                    markersize=10, markeredgecolor='black')
        
        # 動物の描画（レベルに応じた大きさ）
        for animal in self.animals:
            marker_size = 100 + animal.level * 50  # レベルが高いほど大きく
            plt.plot(animal.position.x, animal.position.y, 'D', 
                    color='yellow', markersize=np.sqrt(marker_size), 
                    markeredgecolor='black')
            # レベル数字の表示
            plt.text(animal.position.x, animal.position.y, str(animal.level),
                    ha='center', va='center', color='black', fontweight='bold')
        
        # エージェントの描画
        for agent in self.agents:
            # 体力に応じた色（体力が低いほど赤く）
            health_ratio = agent.health / agent.max_health
            agent_color = (1.0, health_ratio, health_ratio)  # RGB
            
            plt.plot(agent.position.x, agent.position.y, 'o',
                    color=agent_color, markersize=15, markeredgecolor='black')
        
        # 凡例の追加
        legend_elements = [
            patches.Patch(facecolor=terrain_colors[0], label='Water'),
            patches.Patch(facecolor=terrain_colors[1], label='Land'),
            patches.Patch(facecolor=terrain_colors[2], label='Lava'),
            patches.Patch(facecolor='#8B4513', label='Wall Block'),
            patches.Patch(facecolor='#FFFFFF80', label='Bridge Block'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                    markersize=10, label='Agent'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow',
                    markersize=10, label='Animal')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                loc='upper left')
        
        plt.title('Grid World Environment')
        plt.tight_layout()
        
        # 画像の保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # プロットの表示
        if show:
            plt.show()
        else:
            plt.close()

    def render_with_info(self, save_path: str = None, show: bool = True):
        """エージェントと動物の詳細情報を含めて描画"""
        # 図全体の作成
        fig, (map_ax, info_ax) = plt.subplots(1, 2, figsize=(15, 8), 
                                            gridspec_kw={'width_ratios': [1.5, 1]})
        
        # 左側: マップの描画
        # 地形の色マップ作成
        terrain_colors = ['#4A90E2', '#90EE90', '#FF6B6B']  # 水、陸地、溶岩の色
        terrain_cmap = ListedColormap(terrain_colors)
        
        # 地形の描画
        terrain_data = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                terrain_data[y, x] = self.grid[y, x].value
        
        map_ax.imshow(terrain_data, cmap=terrain_cmap)
        
        # グリッド線の描画
        map_ax.grid(True, color='black', alpha=0.2)
        map_ax.set_xticks(np.arange(-0.5, self.width, 1), [])
        map_ax.set_yticks(np.arange(-0.5, self.height, 1), [])
        
        # ブロックの描画
        for pos, block in self.blocks.items():
            if block.is_bridge:
                color = '#FFFFFF80'  # 半透明の白（足場）
                marker = 's'
            else:
                color = '#8B4513'  # 茶色（壁）
                marker = 's'
            map_ax.plot(pos.x, pos.y, marker=marker, color=color, 
                    markersize=10, markeredgecolor='black')
        
        # 動物の描画
        for animal in self.animals:
            marker_size = 100 + animal.level * 50
            map_ax.plot(animal.position.x, animal.position.y, 'D', 
                    color='yellow', markersize=np.sqrt(marker_size), 
                    markeredgecolor='black')
            map_ax.text(animal.position.x, animal.position.y, str(animal.level),
                    ha='center', va='center', color='black', fontweight='bold')
        
        # エージェントの描画
        for agent in self.agents:
            health_ratio = agent.health / agent.max_health
            agent_color = (1.0, health_ratio, health_ratio)
            map_ax.plot(agent.position.x, agent.position.y, 'o',
                    color=agent_color, markersize=15, markeredgecolor='black')
        
        # 凡例の追加
        legend_elements = [
            patches.Patch(facecolor=terrain_colors[0], label='Water'),
            patches.Patch(facecolor=terrain_colors[1], label='Land'),
            patches.Patch(facecolor=terrain_colors[2], label='Lava'),
            patches.Patch(facecolor='#8B4513', label='Wall Block'),
            patches.Patch(facecolor='#FFFFFF80', label='Bridge Block'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                    markersize=10, label='Agent'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow',
                    markersize=10, label='Animal')
        ]
        map_ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                    loc='upper left')
        
        map_ax.set_title('Grid World Map')
        
        # 右側: 情報の表示
        info_ax.axis('off')
        
        # エージェント情報
        info_text = "Agents Status:\n"
        for i, agent in enumerate(self.agents):
            info_text += f"\nAgent {i+1}:\n"
            info_text += f"Position: ({agent.position.x}, {agent.position.y})\n"
            info_text += f"Health: {agent.health:.1f}/{agent.max_health}\n"
            info_text += f"Hunger: {agent.hunger:.1f}\n"
            info_text += f"Thirst: {agent.thirst:.1f}\n"
            info_text += f"Blocks: {agent.block_count}\n"
        
        # 動物情報
        info_text += "\nAnimals Status:\n"
        for i, animal in enumerate(self.animals):
            info_text += f"\nAnimal {i+1}:\n"
            info_text += f"Position: ({animal.position.x}, {animal.position.y})\n"
            info_text += f"Level: {animal.level}\n"
            info_text += f"Health: {animal.health:.1f}/{animal.max_health}\n"
        
        info_ax.text(0.05, 0.95, info_text, fontsize=10, 
                    verticalalignment='top', 
                    transform=info_ax.transAxes)
        
        info_ax.set_title('Status Information')
        
        # レイアウトの調整
        plt.tight_layout()
        
        # 画像の保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # プロットの表示
        if show:
            plt.show()
        else:
            plt.close()

class Action(Enum):
    """プレイヤーの行動を定義"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PLACE_BLOCK = 4  # スペースキーでブロック設置
    ATTACK = 5       # Enterキーで攻撃

class GameController:
    def __init__(self, world, cell_size=30, vision_range=10):
        self.world = world
        self.cell_size = cell_size
        self.vision_range = vision_range
        
        # ステータスバーの設定
        self.status_height = 40  # ステータスバー領域の高さ
        self.bar_height = 10     # 各ゲージの高さ
        self.bar_width = 200     # ゲージの幅
        self.bar_margin = 10     # ゲージ間の余白

        # Pygameの初期化
        pygame.init()
        pygame.display.set_caption('Grid World Game')
        
        # 表示サイズの設定
        viewport_size = (vision_range * 2 + 1)
        self.width = viewport_size * cell_size
        self.height = viewport_size * cell_size + self.status_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        # 色の定義（基本色を先に定義）
        self.colors = {
            'WATER': (74, 144, 226),
            'LAND': (144, 238, 144),
            'LAVA': (255, 107, 107),
            'BLOCK': (139, 69, 19),
            'BRIDGE': (255, 255, 255),
            'AGENT': (255, 0, 0),
            'ANIMAL': (255, 255, 0),
            'TEXT': (0, 0, 0),
            'FOG': (100, 100, 100),
        }
        
        # ステータスバー用の色を追加
        self.colors.update({
            'HEALTH_BAR': (255, 0, 0),      # 赤
            'HUNGER_BAR': (0, 255, 0),      # 緑
            'THIRST_BAR': (0, 191, 255),    # 青
            'BAR_BG': (64, 64, 64),         # 暗灰色（バーの背景）
        })
        
        self.font = pygame.font.Font(None, 24)
        self.last_key_time = 0
        self.key_delay = 100  # ミリ秒
    
    def handle_input(self):
        """キー入力の処理"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_key_time < self.key_delay:
            return

        if not self.world.agents:
            return

        agent = self.world.agents[0]
        keys = pygame.key.get_pressed()
        moved = False

        # 移動
        if keys[pygame.K_UP]:
            moved = self.world.move_agent(agent, 0, -1)
        elif keys[pygame.K_DOWN]:
            moved = self.world.move_agent(agent, 0, 1)
        elif keys[pygame.K_LEFT]:
            moved = self.world.move_agent(agent, -1, 0)
        elif keys[pygame.K_RIGHT]:
            moved = self.world.move_agent(agent, 1, 0)
        
        # ブロック設置
        elif keys[pygame.K_SPACE]:
            if agent.block_count > 0:
                target_pos = Position(agent.position.x, agent.position.y + 1)
                if self.world.place_block(target_pos):
                    agent.use_block()
                    moved = True
        
        # 攻撃
        elif keys[pygame.K_RETURN]:
            for animal in self.world.animals:
                if agent.position.is_adjacent(animal.position):
                    self.world.attack_animal(agent, animal.position)
                    moved = True

        # 肉を食べる (Q キー)
        elif keys[pygame.K_q]:
            if agent.eat_meat():
                moved = True
                print("Ate meat! Hunger restored.")
            else:
                print("No meat available!")

        # 水を飲む (E キー)
        elif keys[pygame.K_e]:
            # 水源に隣接しているかチェック
            can_drink = False
            current_terrain = self.world.get_terrain(agent.position)
            
            # 現在地が水の場合
            if current_terrain == TerrainType.WATER:
                can_drink = True
            else:
                # 隣接するマスをチェック
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor_pos = Position(agent.position.x + dx, agent.position.y + dy)
                    if (self.world.is_valid_position(neighbor_pos) and 
                        self.world.get_terrain(neighbor_pos) == TerrainType.WATER):
                        can_drink = True
                        break
            
            if can_drink:
                agent.thirst = 100.0  # 水分を最大値まで回復
                moved = True
                print("Drank water! Thirst restored.")
            else:
                print("Must be on or next to water to drink!")

        if moved:
            self.last_key_time = current_time
            self.world.step()
        
    def run(self):
        """ゲームループの実行"""
        clock = pygame.time.Clock()
        running = True
        
        try:
            while running:
                # イベント処理
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                # 入力処理
                self.handle_input()
                
                # 画面の更新
                self.render()
                
                # フレームレートの制御
                clock.tick(30)
                
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            pygame.quit()
            sys.exit()
        
    def render_status_bars(self, agent):
        """ステータスバーの描画（修正版）"""
        # バーの開始位置と設定
        margin_left = 10
        start_y = 5
        label_width = 50
        spacing = 5
        value_width = 60

        # 各ステータスのデータ
        stats = [
            ("HP", agent.health / agent.max_health, self.colors['HEALTH_BAR']),
            ("Food", agent.hunger / 100.0, self.colors['HUNGER_BAR']),
            ("Water", agent.thirst / 100.0, self.colors['THIRST_BAR'])
        ]

        for i, (label, ratio, color) in enumerate(stats):
            y = start_y + i * (self.bar_height + self.bar_margin)
            x = margin_left

            # ラベルの描画
            text = self.font.render(label, True, self.colors['TEXT'])
            text_rect = text.get_rect(left=x, centery=y + self.bar_height/2)
            self.screen.blit(text, text_rect)

            # バーの背景
            bar_x = x + label_width
            bar_bg_rect = pygame.Rect(bar_x, y, self.bar_width, self.bar_height)
            pygame.draw.rect(self.screen, self.colors['BAR_BG'], bar_bg_rect)

            # バーの前景（現在値）
            bar_width = int(self.bar_width * max(0, min(1, ratio)))  # 0から1の間に制限
            bar_rect = pygame.Rect(bar_x, y, bar_width, self.bar_height)
            pygame.draw.rect(self.screen, color, bar_rect)

            # 数値表示
            value_text = self.font.render(f"{ratio*100:.1f}%", True, self.colors['TEXT'])
            value_rect = value_text.get_rect(
                left=bar_x + self.bar_width + spacing,
                centery=y + self.bar_height/2
            )
            self.screen.blit(value_text, value_rect)
        
        # キー操作の説明を追加
        help_text = [
            "Controls:",
            "Arrow keys: Move",
            "SPACE: Place block",
            "ENTER: Attack",
            "Q: Eat meat",
            "E: Drink water"
        ]
        
        # 画面右側に説明を表示
        text_x = self.width - 150
        text_y = 5
        for line in help_text:
            text = self.font.render(line, True, self.colors['TEXT'])
            self.screen.blit(text, (text_x, text_y))
            text_y += 20
    
    def render_inventory(self, agent, x, y):
        """インベントリの描画"""
        # インベントリの背景
        inventory_width = 200
        inventory_height = 100
        padding = 10
        
        # 背景の描画（半透明の黒）
        inventory_surface = pygame.Surface((inventory_width, inventory_height))
        inventory_surface.fill((50, 50, 50))
        inventory_surface.set_alpha(200)
        self.screen.blit(inventory_surface, (x, y))
        
        # タイトル
        title = self.font.render("Inventory", True, self.colors['TEXT'])
        self.screen.blit(title, (x + padding, y + padding))
        
        # アイテムリストの描画
        item_y = y + 40
        
        # ブロックの表示
        block_text = self.font.render(f"Blocks: {agent.block_count}", True, self.colors['TEXT'])
        self.screen.blit(block_text, (x + padding, item_y))
        
        # インベントリ内のアイテムを表示
        for name, item in agent.inventory.items.items():
            item_y += 20
            item_text = self.font.render(
                f"{name}: {item.amount} (Heal: {item.healing_value})", 
                True, 
                self.colors['TEXT']
            )
            self.screen.blit(item_text, (x + padding, item_y))
    
    def render(self):
        """画面の描画（ステータスバー付き）"""
        try:
            self.screen.fill(self.colors['FOG'])
            
            if not self.world.agents:
                return
            
            agent = self.world.agents[0]
            viewport_x = agent.position.x - self.vision_range
            viewport_y = agent.position.y - self.vision_range
            
            # ゲーム画面の描画用サーフェス
            game_surface = pygame.Surface((self.width, self.height - self.status_height))
            game_surface.fill(self.colors['FOG'])
            
            # 地形の描画
            for y in range(max(0, viewport_y), min(self.world.height, viewport_y + 2 * self.vision_range + 1)):
                for x in range(max(0, viewport_x), min(self.world.width, viewport_x + 2 * self.vision_range + 1)):
                    screen_x = (x - viewport_x) * self.cell_size
                    screen_y = (y - viewport_y) * self.cell_size
                    
                    if 0 <= screen_x < self.width and 0 <= screen_y < game_surface.get_height():
                        rect = pygame.Rect(screen_x, screen_y, self.cell_size, self.cell_size)
                        terrain = self.world.grid[y][x]
                        
                        if terrain == TerrainType.WATER:
                            color = self.colors['WATER']
                        elif terrain == TerrainType.LAVA:
                            color = self.colors['LAVA']
                        else:
                            color = self.colors['LAND']
                        
                        pygame.draw.rect(game_surface, color, rect)
                        pygame.draw.rect(game_surface, (0, 0, 0), rect, 1)
            
            # ブロックの描画
            for pos, block in self.world.blocks.items():
                screen_x = (pos.x - viewport_x) * self.cell_size
                screen_y = (pos.y - viewport_y) * self.cell_size
                
                if (0 <= screen_x < self.width and 0 <= screen_y < game_surface.get_height() and
                    abs(pos.x - agent.position.x) <= self.vision_range and 
                    abs(pos.y - agent.position.y) <= self.vision_range):
                    
                    rect = pygame.Rect(screen_x, screen_y, self.cell_size, self.cell_size)
                    color = self.colors['BRIDGE'] if block.is_bridge else self.colors['BLOCK']
                    pygame.draw.rect(game_surface, color, rect)
            
            # 動物の描画
            for animal in self.world.animals:
                screen_x = (animal.position.x - viewport_x) * self.cell_size
                screen_y = (animal.position.y - viewport_y) * self.cell_size
                
                if (0 <= screen_x < self.width and 
                    0 <= screen_y < game_surface.get_height() and
                    abs(animal.position.x - agent.position.x) <= self.vision_range and 
                    abs(animal.position.y - agent.position.y) <= self.vision_range):
                    
                    center = (screen_x + self.cell_size // 2, screen_y + self.cell_size // 2)
                    pygame.draw.circle(game_surface, self.colors['ANIMAL'], 
                                    center, self.cell_size // 2 - 2)
                    
                    # レベルの表示
                    text = self.font.render(str(animal.level), True, self.colors['TEXT'])
                    text_rect = text.get_rect(center=center)
                    game_surface.blit(text, text_rect)
            
            # プレイヤー（エージェント）の描画
            player_screen_x = self.vision_range * self.cell_size
            player_screen_y = self.vision_range * self.cell_size
            player_center = (player_screen_x + self.cell_size // 2, 
                            player_screen_y + self.cell_size // 2)
            
            pygame.draw.circle(game_surface, self.colors['AGENT'], 
                            player_center, self.cell_size // 2 - 2)
            
            # ゲーム画面をメイン画面に転送
            self.screen.blit(game_surface, (0, self.status_height))
            
            # ステータスバーの描画
            self.render_status_bars(agent)
            
            # インベントリの描画（画面左下）
            self.render_inventory(agent, 10, self.height - 120)
            
            # キー操作ガイドの描画（画面右上）
            help_text = [
                "Controls:",
                "Arrow keys: Move",
                "SPACE: Place block",
                "ENTER: Attack",
                "Q: Eat meat",
                "E: Drink water"
            ]
            
            text_x = self.width - 150
            text_y = 5
            for line in help_text:
                text = self.font.render(line, True, self.colors['TEXT'])
                self.screen.blit(text, (text_x, text_y))
                text_y += 20
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Render error: {e}")
            traceback.print_exc()

if __name__=="__main__":
    
    world = GridWorld(width=64, height=64)

    # Perlinノイズで地形を生成
    world.generate_terrain_map(
        scale=40.0,           # 地形の大きさを調整
        lava_threshold=0.0,
        land_threshold=0.1,   
        seed=0             # 再現性のために乱数シードを設定
    )
    
    import matplotlib.pyplot as plt
    grid = np.zeros_like(world.grid, dtype=np.uint8)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if world.grid[x][y] == TerrainType.LAND:
                grid[x][y] = 1
            elif world.grid[x][y] == TerrainType.LAVA:
                grid[x][y] = 2
            else:
                grid[x][y] = 0
            
    plt.imshow(grid, vmax=2, vmin=0)
    plt.colorbar()
    plt.show()
    # エージェントをランダムな適切な位置に配置
    agent = world.add_agent()
    
    """world.step()
    state = world.get_state()

    # 現在の動物のレベル分布を確認
    level_distribution = world.get_animal_distribution()
    print("Animal Level Distribution:", level_distribution)

    # 特定の動物の詳細情報を確認
    for animal_pos, animal_stats in state['animals']:
        print(f"Animal at {animal_pos}:")
        print(f"  Level: {animal_stats['level']}")
        print(f"  Health: {animal_stats['health']}/{animal_stats['max_health']}")
        print(f"  Attack Power: {animal_stats['attack_power']}")"""
    
    agent.add_blocks(10)  # 10個のブロックを所持
    
    controller = GameController(world)
    controller.run()

    """# ブロックの設置例
    target_pos = Position(agent.position.x + 1, agent.position.y)
    if agent.use_block():  # ブロックを消費
        world.place_block(target_pos)  # ブロックを設置

    # 移動の例（ブロックの有無で通行可能性が変化）
    world.move_agent(agent, 1, 0)
    
    # シミュレーションの実行
    for _ in range(100):
        # エージェントの行動例
        world.move_agent(agent, 1, 0)  # 移動
        
        # 隣接する動物を攻撃
        for animal in world.animals:
            if agent.position.is_adjacent(animal.position):
                world.attack_animal(agent, animal.position)
        
        # 空腹度が低い場合は肉を食べる
        if agent.hunger < 50.0:
            agent.eat_meat()
        
        # 環境を1ステップ進める
        world.step()
        
        # 状態の確認
        state = world.get_state()
        
        # エージェントの状態を表示
        agent_state = state['agents'][0]
        print(f"Health: {agent_state[1]}, Hunger: {agent_state[2]}, Thirst: {agent_state[3]}")
        print(f"Inventory: {agent_state[4]}")
        
    # 基本的な描画
    world.render()

    # 詳細情報付きの描画
    world.render_with_info()

    # 画像として保存
    world.render(save_path="grid_world.png", show=False)"""