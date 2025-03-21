from enum import Enum

class Terrain(Enum):
    WATER = 0
    LAND = 1
    LAVA = 2

class Action(Enum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    EAT = 5
    DRINK = 6
    ATTACK = 7
    PLACE_MEAT = 8
    PLACE_BLOCK = 9
    PICKUP = 10

class GridObject(Enum):
    EMPTY = 0
    PLAYER = 1
    ANIMAL = 2
    BLOCK = 3
    MEAT = 4
    
class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3