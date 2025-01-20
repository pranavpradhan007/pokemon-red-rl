import gym
from gym import spaces
import numpy as np

class BattleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)  # 4 moves
        self.observation_space = spaces.Box(low=0, high=255, shape=(10,), dtype=np.float32)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        obs = np.random.random(10)  # Example
        reward = 1.0 if action == 2 else -0.1  # Example logic
        done = False
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
