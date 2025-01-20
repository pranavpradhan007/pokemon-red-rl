from environments.overworld_env import OverworldEnv
env = OverworldEnv("data/roms/PokemonRed.gb")
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Replace with meaningful actions if possible
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    if done:
        break
env.close()
