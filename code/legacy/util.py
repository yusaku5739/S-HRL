from noise import pnoise2
import random
import matplotlib.pyplot as plt
import numpy as np

import object

def generate_perlin_map(width, height, scale, seed, land_threshold, lava_threshold, min_water_size=5):
    """
    Perlinノイズを使用して「海」「陸」「溶岩」を含むマップを生成。
    """
    random.seed(seed)
    base_map = [[pnoise2(x / scale, y / scale, octaves=4, base=seed)
                 for x in range(width)] for y in range(height)]
    
    # Perlinノイズ値を「海」「陸」「溶岩」に変換
    map_grid = []
    for row in base_map:
        new_row = []
        for value in row:
            if value < land_threshold:
                new_row.append("water")  # 海
            elif value < lava_threshold:
                new_row.append("land")   # 陸
            else:
                new_row.append("lava")   # 溶岩
        map_grid.append(new_row)

    # 「海」と「溶岩」が隣接しないように調整
    ensure_no_water_lava_contact(map_grid, width, height)

    # 小さな「海」を陸地で埋める
    fill_small_water_areas(map_grid, width, height, min_water_size)

    return map_grid

def ensure_no_water_lava_contact(map_grid, width, height):
    """
    海と溶岩が直接隣接しないように調整。
    隣接する「水」または「溶岩」は「陸」に変換。
    """
    for y in range(height):
        for x in range(width):
            if map_grid[y][x] == "water":
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and map_grid[ny][nx] == "lava":
                        map_grid[ny][nx] = "land"  # 隣接する溶岩を陸地に変換

def fill_small_water_areas(map_grid, width, height, min_size):
    """
    サイズが小さい海を陸地で埋める。
    """
    visited = [[False for _ in range(width)] for _ in range(height)]

    def flood_fill(x, y):
        # 海の領域を探索して座標を収集
        stack = [(x, y)]
        area = []
        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < width and 0 <= cy < height) or visited[cy][cx] or map_grid[cy][cx] != "water":
                continue
            visited[cy][cx] = True
            area.append((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((cx + dx, cy + dy))
        return area

    # 小さな海を陸地で埋める
    for y in range(height):
        for x in range(width):
            if map_grid[y][x] == "water" and not visited[y][x]:
                water_area = flood_fill(x, y)
                if len(water_area) < min_size:
                    for wx, wy in water_area:
                        map_grid[wy][wx] = "land"

def spawn_players_with_distance(map_grid, num_players, min_distance, max_distance):
    """
    プレイヤーを陸地にスポーンさせる関数（最小距離と最大距離を考慮）。
    
    Parameters:
        map_grid (list of list): マップデータ。
        num_players (int): スポーンさせるプレイヤーの人数。
        min_distance (int): プレイヤー間の最小距離（ブロック数）。
        max_distance (int): プレイヤー間の最大距離（ブロック数）。
    
    Returns:
        list of tuple: プレイヤーの座標リスト。
    """
    height = len(map_grid)
    width = len(map_grid[0])
    land_positions = [(x, y) for y in range(height) for x in range(width) if map_grid[y][x] == "land"]

    if len(land_positions) < num_players:
        raise ValueError("陸地が足りないため、すべてのプレイヤーをスポーンさせることができません。")

    player_positions = []

    while len(player_positions) < num_players:
        # ランダムに陸地を選択
        x, y = random.choice(land_positions)

        # 他のプレイヤーとの距離をチェック
        if all(
            min_distance <= ((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= max_distance
            for px, py in player_positions
        ):
            player_positions.append((x, y))

    return player_positions

def place_trees(map_grid, num_trees):
    """
    マップにランダムに木を配置する関数。
    
    Parameters:
        map_grid (list of list): マップデータ。
        num_trees (int): 木の数。
    """
    height = len(map_grid)
    width = len(map_grid[0])
    land_positions = [(x, y) for y in range(height) for x in range(width) if map_grid[y][x] == "land"]
    
    for _ in range(num_trees):
        # ランダムに陸地を選択して木を配置
        x, y = random.choice(land_positions)
        map_grid[y][x] = "tree"  # 木を配置
        print(f"Tree placed at ({x}, {y})")
        
def spawn_cows(map_grid, num_cows):
    """
    牛をマップにランダムにスポーンさせる関数
    
    Parameters:
        map_grid (list of list): マップデータ
        num_cows (int): スポーンさせる牛の数
    """
    height = len(map_grid)
    width = len(map_grid[0])
    land_positions = [(x, y) for y in range(height) for x in range(width) if map_grid[y][x] == "land"]
    
    cows = []
    for _ in range(num_cows):
        x, y = random.choice(land_positions)
        cows.append(object.Cow(x, y))
        map_grid[y][x] = "cow"  # 牛を配置
        print(f"Cow spawned at ({x}, {y})")
    
    return cows


def ensure_minimum_cows(map_grid, cows, min_cows):
    """
    牛の数が一定数以下の場合、新しい牛をランダムにスポーンさせる関数
    
    Parameters:
        map_grid (list of list): マップデータ
        cows (list): 現在の牛オブジェクトリスト
        min_cows (int): 必要な最小数
    """
    if len(cows) < min_cows:
        new_cows = spawn_cows(map_grid, min_cows - len(cows))
        cows.extend(new_cows)

def spawn_enemies(map_grid, num_enemies):
    height = len(map_grid)
    width = len(map_grid[0])
    land_positions = [(x, y) for y in range(height) for x in range(width) if map_grid[y][x] == "land"]
    
    enemies = []
    for _ in range(num_enemies):
        x, y = random.choice(land_positions)
        enemies.append(object.Enemy(x, y))
        map_grid[y][x] = "enemy"
        print(f"Enemy spawned at ({x}, {y})")
    
    return enemies

def ensure_minimum_enemies(map_grid, enemies, min_enemies):
    if len(enemies) < min_enemies:
        new_enemies = spawn_enemies(map_grid, min_enemies - len(enemies))
        enemies.extend(new_enemies)

def visualize_map_with_objects(map_grid, players):
    """
    プレイヤーと木が配置されたマップをmatplotlibで可視化。
    """
    # マップを数値データに変換
    color_map = {"water": 0, "land": 1, "lava": 2, "tree": 3}
    numeric_map = np.array([[color_map[cell] for cell in row] for row in map_grid])
    print(numeric_map)
    # カラーマップを設定
    cmap = plt.cm.get_cmap("terrain", 4)  # 4色（海、陸、溶岩、木）
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.imshow(numeric_map, cmap=cmap, origin="upper")

    # プレイヤーをプロット
    for player in players:
        plt.scatter(player.x, player.y, color="red", label=f"Player {player.id}", edgecolors="black", s=100, zorder=5)
    
    # カラーバー
    cbar = plt.colorbar(ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Water (~)", "Land (#)", "Lava (^)","Tree (T)"])
    plt.title("Map with Players and Trees")
    plt.axis("off")
    plt.show()