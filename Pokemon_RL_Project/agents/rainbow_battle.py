from stable_baselines3 import DQN
from environments.battle_env import BattleEnv

env = BattleEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("data/models/rainbow_battle.zip")
