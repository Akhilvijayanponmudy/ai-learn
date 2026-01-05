import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create log dir
models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def train():
    print("🚗 Starting Training on CartPole-v1...")
    
    # Create Environment
    env = gym.make("CartPole-v1")
    
    # Initialize Agent (PPO is a standard, robust algorithm)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # Train
    TIMESTEPS = 10000
    iters = 0
    while iters < 5: # Train for 5 iterations (50k steps total)
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{models_dir}/{TIMESTEPS*iters}")
        print(f"💾 Model saved: {models_dir}/{TIMESTEPS*iters}")
        
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()
