import gymnasium as gym
from stable_baselines3 import PPO
import time
import os
import random

def evaluate():
    model_path = "models/PPO/50000.zip" # Load the last model
    
    print(f"🔎 Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except:
        print("❌ Model not found! Run training first.")
        return

    # Create env
    env = gym.make("CartPole-v1")
    
    obs, info = env.reset()
    score = 0
    
    print("🚗 Starting Simulation Loop...")
    
    # Traffic state
    obstacles = [] # List of [row, col]
    road_width = 20
    road_height = 10
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # --- Update Visuals ---
        
        # 1. Update Obstacles (Fake Traffic for visuals)
        # Move existing down
        new_obstacles = []
        for r, c in obstacles:
            if r < road_height - 1:
                new_obstacles.append([r + 1, c])
        obstacles = new_obstacles
        
        # Spawn new interaction
        if random.random() < 0.3: # 30% chance per frame
            obstacles.append([0, random.randint(0, road_width-1)])
            
        # 2. Player Position (Map CartPole -2.4 to 2.4 -> 0 to 20)
        cart_pos = obs[0]
        player_col = int((cart_pos + 2.4) / 4.8 * road_width)
        player_col = max(0, min(road_width-1, player_col))
        
        # 3. Render 2D Grid
        os.system('clear') # Clear screen for animation
        print(f"Score: {score:.1f}")
        print("=" * (road_width + 2))
        
        for r in range(road_height):
            line = [" "] * road_width
            
            # Draw Traffic
            for ob_r, ob_c in obstacles:
                if ob_r == r:
                    line[ob_c] = "🚙"
            
            # Draw Player (only on bottom row)
            if r == road_height - 1:
                 # Check Collision
                if [r, player_col] in obstacles:
                     line[player_col] = "💥"
                     print(f"|{''.join(line)}|")
                     print(f"======================")
                     print(f"\n❌ CRASHED into a Blue Car! Score: {score:.1f}")
                     time.sleep(2)
                     
                     # Reset
                     obs, info = env.reset()
                     score = 0
                     obstacles = []
                     break # Stop rendering this frame
                
                line[player_col] = "🚗"

            print(f"|{''.join(line)}|")
            
        print("=" * (road_width + 2))
        
        time.sleep(0.1)
        
        if terminated or truncated:
            obs, info = env.reset()
            score = 0
            obstacles = []
            print("\n❌ Pole Fell! Resetting...")
            time.sleep(1)

if __name__ == "__main__":
    evaluate()
