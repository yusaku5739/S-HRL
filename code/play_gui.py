import pygame
import yaml

from env import GridWorldEnv, RLWrapper
from each_class import Action

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
        
    env = RLWrapper(GridWorldEnv(config=config,  render_mode="human"))
    interface = KeyboardPlayerInterface(env)
    interface.run()
    env = RLWrapper(GridWorldEnv(config=config,  render_mode="human"))
    obs, _ = env.reset()
    print(obs)