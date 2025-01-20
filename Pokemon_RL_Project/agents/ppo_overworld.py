from stable_baselines3 import PPO
from environments.overworld_env import OverworldEnv

try:
    # Initialize environment and model
    env = OverworldEnv("data/roms/PokemonRed.gb")
    model = PPO("CnnPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000)
    model.save("data/models/ppo_overworld_final.zip")
    print("Training completed and model saved.")

except Exception as e:
    print(f"An error occurred: {e}")
