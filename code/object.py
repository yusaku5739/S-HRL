import numpy as np
from enum import Enum
from collections import deque
import random
from typing import List, Tuple
import math

from each_class import Terrain, GridObject, Direction
    
    
class Player:
    def __init__(self, x: int, y: int, config: dict):
        self.x = x
        self.y = y
        self.config = config
        self.direction = Direction.RIGHT
        self.health = config["health"]
        self.hunger = config["hunger"]
        self.thirst = config["thirst"]
        self.ut = {"health":config["health"], "hunger":config["hunger"], "thirst":config["hunger"]}
        self.dt = self.config["drive_weight_health"] * (1 - self.config["sustain_param_health"]) + \
                  self.config["drive_weight_hunger"] * (1 - self.config["sustain_param_hunger"]) + \
                  self.config["drive_weight_thirst"] * (1 - self.config["sustain_param_thirst"])
        self.control_param = ["health", "hunger", "thirst"]
        self.inventory = {'block': 0, 'meat': 0}
        self.attack_power = config["attack_power"]
        self.view_range = config["view_range"]  # プレイヤーの視界範囲
        self.view_angle = config["view_angle"]  # 視野角（度）
        self.last_known_positions = {}  # 他エージェントの最後に観測された位置
        self.interaction_range = 1 # プレイヤーの相互作用範囲
        
        self.reward_scaling = config["reward_scaling"]
        self.reward_history = deque(maxlen=config["history_size"])
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.epsilon = 1e-8 
        
    def can_move_to(self, x: int, y: int, terrain_map, object_map) -> bool:
        """
        移動可能かどうかをより厳密にチェック
        """
        # マップ境界チェック
        if not (0 <= x < terrain_map.shape[0] and 0 <= y < terrain_map.shape[1]):
            return False
            
        terrain = terrain_map[x, y]
        obj = object_map[x, y]
        
        # 溶岩は即死なので移動不可
        if terrain == Terrain.LAVA.value:
            return False
        
        # 水上移動の条件チェック
        if terrain == Terrain.WATER.value:
            # 水上はブロックが設置されている場合のみ移動可能
            return obj == GridObject.BLOCK.value
            
        # 陸地での移動条件チェック
        if terrain == Terrain.LAND.value:
            # 空のマスのみ移動可能に変更
            return obj == GridObject.EMPTY.value
            
        return False

    def get_front_position(self) -> Tuple[int, int]:
        """
        プレイヤーの向いている方向の座標を取得
        """
        dx, dy = 0, 0
        if self.direction == Direction.UP:
            dx = -1
        elif self.direction == Direction.DOWN:
            dx = 1
        elif self.direction == Direction.LEFT:
            dy = -1
        elif self.direction == Direction.RIGHT:
            dy = 1
        return (self.x + dx, self.y + dy)

    def is_within_interaction_range(self, target_x: int, target_y: int) -> bool:
        """
        指定座標が相互作用範囲内かどうかをチェック
        """
        dx = abs(self.x - target_x)
        dy = abs(self.y - target_y)
        return dx <= self.interaction_range and dy <= self.interaction_range

    def can_place_object(self, x: int, y: int, terrain_map, object_map, object_type: GridObject) -> bool:
        """
        オブジェクトを設置可能かどうかをチェック
        """
        # マップ境界チェック
        if not (0 <= x < terrain_map.shape[0] and 0 <= y < terrain_map.shape[1]):
            return False

        # 設置対象の位置が相互作用範囲内かチェック
        if not self.is_within_interaction_range(x, y):
            return False

        terrain = terrain_map[x, y]
        obj = object_map[x, y]

        # ブロックの設置条件
        if object_type == GridObject.BLOCK:
            # 水上または陸地にのみ設置可能
            if terrain not in [Terrain.WATER.value, Terrain.LAND.value]:
                return False
        # 肉の設置条件
        elif object_type == GridObject.MEAT:
            # 陸地にのみ設置可能
            if terrain != Terrain.LAND.value:
                return False

        # 設置先が空いているかチェック
        return obj == GridObject.EMPTY.value

    def can_pickup_object(self, x: int, y: int, object_map) -> bool:
        """
        オブジェクトを拾えるかどうかをチェック
        """
        # マップ境界チェック
        if not (0 <= x < object_map.shape[0] and 0 <= y < object_map.shape[1]):
            return False

        # 相互作用範囲内かチェック
        if not self.is_within_interaction_range(x, y):
            return False

        obj = object_map[x, y]
        return obj in [GridObject.BLOCK.value, GridObject.MEAT.value]
    
    def get_view_area(self) -> List[Tuple[int, int]]:
        """プレイヤーの視界範囲内の座標リストを取得"""
        view_coords = []
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                # 視界範囲内の距離チェック
                if dx*dx + dy*dy <= self.view_range*self.view_range:
                    # 視野角のチェック
                    if self._is_in_view_angle(dx, dy):
                        view_coords.append((self.x + dx, self.y + dy))
        return view_coords
        
    def _is_in_view_angle(self, dx: int, dy: int) -> bool:
        """指定された相対座標が視野角内かどうかをチェック"""
        if dx == 0 and dy == 0:
            return True
            
        # グリッドの座標系に合わせて角度を計算
        # GridWorldでは上がマイナス、下がプラス
        target_angle = math.degrees(math.atan2(-dx, dy))  # dxの符号を反転
        
        # 方向に応じた基準角度を設定
        base_angle = {
            Direction.RIGHT: 0,
            Direction.DOWN: 270,
            Direction.LEFT: 180,
            Direction.UP: 90
        }[self.direction]
        
        # 角度の差分を計算（-180から180の範囲に正規化）
        angle_diff = ((target_angle - base_angle + 180) % 360) - 180
        
        # 視野角の半分を基準に判定
        return abs(angle_diff) <= self.view_angle / 2
    
    def _update_reward_stats(self, reward):
        """Update running statistics for reward scaling"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            self.reward_running_mean = np.mean(self.reward_history)
            self.reward_running_std = np.std(self.reward_history) + self.epsilon

    def _scale_reward(self, reward):
        """Scale reward to approximately [-1, 1] range"""
        if not self.reward_scaling:
            return reward
            
        self._update_reward_stats(reward)
        scaled_reward = (reward - self.reward_running_mean) / self.reward_running_std
        return np.clip(scaled_reward, -1.0, 1.0)
    
    def calculate_introception(self, player_death_flag):
        if player_death_flag:
            utt_health = 0
        else:
            utt_health = 0.95*self.ut["health"] + 0.05*self.health
        utt_hunger = 0.95*self.ut["hunger"] + 0.05*self.hunger
        utt_thirst = 0.95*self.ut["thirst"] + 0.05*self.thirst
        self.ut["health"] = utt_health
        self.ut["hunger"] = utt_hunger
        self.ut["thirst"] = utt_thirst
        xtt_health = utt_health / self.config["health"]
        xtt_hunger = utt_hunger / self.config["hunger"]
        xtt_thirst = utt_thirst / self.config["thirst"]
        return xtt_health, xtt_hunger, xtt_thirst
    
    def calculate_reward(self, xtt_health, xtt_hunger, xtt_thirst):
        dtt = self.config["drive_weight_health"] * (xtt_health - self.config["sustain_param_health"]) + \
               self.config["drive_weight_hunger"] * (xtt_hunger - self.config["sustain_param_hunger"]) + \
               self.config["drive_weight_thirst"] * (xtt_thirst - self.config["sustain_param_thirst"])
        reward = dtt - self.dt
        self.dt = dtt
        if self.config["reward_scaling"]:
            return self._scale_reward(reward)
        else:
            return reward


class Animal:
    def __init__(self, x: int, y: int, config: dict):
        self.x = x
        self.y = y
        self.max_level = config["max_level"]
        self.level = random.randint(1, self.max_level)
        self.health = config["base_health"] + (self.level - 1) * config["mag_health"]
        self.attack_power = config["base_attack_power"] + (self.level - 1) * config["mag_attack_power"]
        self.meat_amount = config["base_meat_amount"] * self.level * 2
        self.target_player = None
        self.is_aggressive = False

    def can_move_to(self, x: int, y: int, terrain_map, object_map) -> bool:
        """
        動物の移動可能判定
        """
        # マップ境界チェック
        if not (0 <= x < terrain_map.shape[0] and 0 <= y < terrain_map.shape[1]):
            return False
            
        terrain = terrain_map[x, y]
        obj = object_map[x, y]
        
        # 陸地でかつ何もないマスのみ移動可能
        return (terrain == Terrain.LAND.value and 
                obj == GridObject.EMPTY.value)

    def update(self, players, terrain_map, object_map):
        # 現在位置を保存
        old_x, old_y = self.x, self.y
        
        if self.is_aggressive and self.target_player:
            # 追跡行動
            dx = np.sign(self.target_player.x - self.x)
            dy = np.sign(self.target_player.y - self.y)
            
            # 移動先の候補を設定
            new_positions = [
                (self.x + dx, self.y + dy),  # 斜め移動
                (self.x + dx, self.y),       # 水平移動
                (self.x, self.y + dy),       # 垂直移動
            ]
            
            # 移動可能な位置を探す
            for new_x, new_y in new_positions:
                if self.can_move_to(new_x, new_y, terrain_map, object_map):
                    self.x = new_x
                    self.y = new_y
                    break
        else:
            # ランダム移動
            directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)  # 方向をランダムに選択
            
            # 移動可能な方向を探す
            if directions[0][0]== 0 and directions[0][1] == 1:
                pass
            else:
                for dx, dy in directions:
                    new_x = self.x + dx
                    new_y = self.y + dy
                    if self.can_move_to(new_x, new_y, terrain_map, object_map):
                        self.x = new_x
                        self.y = new_y
                        break
        
        # 移動が成功した場合、オブジェクトマップを更新
        if (old_x != self.x or old_y != self.y):
            object_map[old_x, old_y] = GridObject.EMPTY.value
            object_map[self.x, self.y] = GridObject.ANIMAL.value