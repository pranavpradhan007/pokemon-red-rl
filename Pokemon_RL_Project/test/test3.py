from environments.overworld_env import OverworldEnv

env = OverworldEnv("data/roms/PokemonRed.gb")
obs = env.reset()

for _ in range(10):  # Test 10 steps
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

env.close()
