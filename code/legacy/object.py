import random
import numpy as np

ENEMY_HEALTH = 5
ENEMY_DAMAGE = 1

COW_HEALTH = 5

MAX_HEALTH = 10
MAX_THIRSTY = 10

WATER_INTAKE = 1
MEAN_GET_MEAT = 4

class Enemy:
    def __init__(self):
        self.health = ENEMY_HEALTH
        self.is_alive = True

    def move_towards(self, player, game):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右
        dx = player.x - self.x
        dy = player.y - self.y

        if abs(dx) > abs(dy):
            move_x = 1 if dx > 0 else -1
            move_y = 0
        else:
            move_x = 0
            move_y = 1 if dy > 0 else -1

        new_x = self.x + move_x
        new_y = self.y + move_y

        if game.is_position_available(new_x, new_y):
            game.remove_object(self)
            game.add_object(self, new_x, new_y)

    def attack(self, player):
        if self.is_alive:
            player.take_damage(ENEMY_DAMAGE)
            print(f"Enemy attacked Player {player.id} and dealt {ENEMY_DAMAGE} damage!")


class Cow:
    def __init__(self):
        """
        牛オブジェクトの初期化
        
        Parameters:
            x (int): 牛の初期x座標
            y (int): 牛の初期y座標
        """
        self.health = COW_HEALTH  # 体力は正規分布に従う
        self.is_alive = True

    def move(self, game):
        """
        牛をランダムに隣接する陸地に移動させる
        
        Parameters:
            game
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右
        move_x, move_y = random.choice(directions)
        new_x = self.x + move_x
        new_y = self.y + move_y
        
        if game.is_position_available(new_x, new_y):
            game.remove_object(self)
            game.add_object(self, new_x, new_y)

    def take_damage(self, damage):
        """
        牛の体力を減少させ、体力が0以下になれば死亡
        
        Parameters:
            damage (int): 与えるダメージ量
        """
        self.health -= damage
        if self.health <= 0:
            self.is_alive = False
            return True  # 牛が死んだことを返す
        return False

class Tree:
    def __init__(self):
        self.type = "wood"

class Player:
    def __init__(self, id, x, y):
        """
        プレイヤークラスの初期化
        
        Parameters:
            id (int): プレイヤーID
            x (int): 初期位置のx座標
            y (int): 初期位置のy座標
        """
        self.id = id
        self.x = x
        self.y = y
        self.health = 10
        self.water = 10
        self.thirsty = False
        self.hungry = False
        self.inventory = {
            "meat":0,
            "tree":0
            }  # プレイヤーのインベントリ（所持品）

    def move(self, action, game):
        """
        プレイヤーを指定された行動に基づいて移動させる
        
        Parameters:
            action (str): "up", "down", "left", "right" のいずれか
            map_grid (list of list): マップデータ
        """
        new_x, new_y = self.x, self.y

        if action == "up":
            new_y -= 1
        elif action == "down":
            new_y += 1
        elif action == "left":
            new_x -= 1
        elif action == "right":
            new_x += 1

        if game.is_position_available(new_x, new_y):
            game.remove_object(self)
            game.add_object(self, new_x, new_y)
    
    def observe(self, map_grid, range_size=1):
        """
        プレイヤーが周囲range_size x range_sizeブロックを観測する機能
        
        Parameters:
            map_grid (list of list): マップデータ
            range_size (int): 観測範囲の半径（デフォルトは1）
        
        Returns:
            observed_area (list of list): プレイヤーの周囲の観測データ
        """
        observed_area = []

        # 観測範囲を決定（周囲range_sizeブロック範囲）
        for dy in range(-range_size, range_size + 1):  # -range_size から range_size まで
            row = []
            for dx in range(-range_size, range_size + 1):  # -range_size から range_size まで
                nx, ny = self.x + dx, self.y + dy
                # 範囲外を避ける
                if 0 <= nx < len(map_grid[0]) and 0 <= ny < len(map_grid):
                    row.append(map_grid[ny][nx])
                else:
                    row.append("out_of_bounds")  # 範囲外
            observed_area.append(row)

        return observed_area

    def attack(self, map_grid, cows):
        """
        プレイヤーが攻撃アクションを取る
        
        プレイヤーが隣接する牛を攻撃し、体力が0になった場合、肉を獲得
        
        Parameters:
            map_grid (list of list): マップデータ
            cows (list): 現在マップに存在する牛オブジェクトのリスト
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < len(map_grid[0]) and 0 <= ny < len(map_grid):
                for cow in cows:
                    if cow.is_alive and cow.x == nx and cow.y == ny:
                        if cow.take_damage(10):  # 体力が0になった場合
                            self.inventory["meat"] += MEAN_GET_MEAT  # 肉を獲得
                            print(f"Player {self.id} killed a cow at ({nx}, {ny}) and got meat!")
                        return
                
                for enemy in self.enemies:
                    if enemy.is_alive and enemy.x == nx and enemy.y == ny:
                        enemy.attack(self)

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            print(f"Player {self.id} has died!")
    
    def interact(self, map_grid):
        """
        プレイヤーが周囲の「木」を収集する。
        プレイヤーの周囲1ブロック内に「木」がある場合、その「木」を収集。
        
        Parameters:
            map_grid (list of list): マップデータ。
        """
        # 周囲の4方向に「木」があるかチェック
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < len(map_grid[0]) and 0 <= ny < len(map_grid):
                if map_grid[ny][nx] == "tree":  # 「木」があれば収集
                    self.inventory["tree"] += 1
                    map_grid[ny][nx] = "land"  # 木を収集したので「陸地」に戻す
                    print(f"Player {self.id} collected a tree at ({nx}, {ny})")
                    return
                elif map_grid[ny][nx] == "water":
                    if self.water < MAX_THIRSTY:
                        self.water += WATER_INTAKE
    
    def eat(self):
        if self.inventory["meat"] > 0 and self.health < MAX_HEALTH:
            self.health += 1
            self.inventory["meat"] -= 1
    
            

    def __repr__(self):
        """
        プレイヤーの現在の状態を表示するための文字列
        """
        return f"Player {self.id} at ({self.x}, {self.y}), Inventory: {self.inventory}"

