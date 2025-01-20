import gym
import numpy as np
from gym import spaces
import logging
from pyboy import PyBoy, WindowEvent


class OverworldEnv(gym.Env):
    def __init__(self, rom_path, emulation_speed=2):
        super().__init__()
        self.rom_path = rom_path  # Store ROM path
        self.emulation_speed = emulation_speed  # Store emulation speed
        self.pyboy = PyBoy(rom_path, window_type="SDL2")
        self.pyboy.set_emulation_speed(emulation_speed)

        # Define available actions
        self.actions = {
            0: WindowEvent.PRESS_ARROW_UP,
            1: WindowEvent.PRESS_ARROW_DOWN,
            2: WindowEvent.PRESS_ARROW_LEFT,
            3: WindowEvent.PRESS_ARROW_RIGHT,
            4: WindowEvent.PRESS_BUTTON_A,
            5: WindowEvent.PRESS_BUTTON_B,
        }

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(160, 144, 3), dtype=np.uint8)

        # Internal state
        self.steps_taken = 0
        self.max_steps = 1000  # End episode after a fixed number of steps (example)
        self.visited_areas = set()  # Track visited areas (e.g., locations or screen hashes)

    # In overworld_env.py
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        logging.info("Initializing PyBoy emulator...")
        self.pyboy = PyBoy(self.rom_path, window_type="SDL2")
        self.pyboy.set_emulation_speed(self.emulation_speed)
        self.steps_taken = 0
        self.visited_areas.clear()
        return self._get_observation()


    def step(self, action):
        """
        Perform the specified action and update the environment state.
        """
        try:
            # Send the action to the emulator
            pyboy_action = self.actions[action]
            self.pyboy.send_input(pyboy_action)
            
            # Advance the emulator by one frame
            self.pyboy.tick()

            # Get the current screen as the observation
            obs = self._get_observation()

            # Calculate the reward
            reward = self._calculate_reward()

            # Determine if the episode is done
            done = self.steps_taken >= self.max_steps

            # Check for additional info (e.g., battle triggers)
            info = {
                "battle_trigger": self._check_battle_trigger(),
                "location": "overworld"
            }

            # Update steps taken
            self.steps_taken += 1

            logging.info(f"Step: {self.steps_taken}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

            return obs, reward, done, info

        except Exception as e:
            logging.error("Error during step execution:", exc_info=True)
            raise





    def _get_observation(self):
        """
        Extract the current game screen as the observation and ensure it matches
        the expected (width, height, channels) format.
        """
        screen = self.pyboy.botsupport_manager().screen().screen_ndarray()
        return np.transpose(screen, (1, 0, 2))  # Transpose to (width, height, channels)

    def _calculate_reward(self):
        """
        Calculate the reward based on the current state.
        """
        # Check if the agent enters a new area
        if self.agent_enters_new_area():
            return 1.0  # Higher reward for exploration
        return -0.1  # Small penalty for "idle" actions

    def agent_enters_new_area(self):
        """
        Determine if the agent has entered a new area.
        """
        # Example: Use the hash of the current screen as a proxy for location
        current_screen = self._get_observation()
        screen_hash = hash(current_screen.tobytes())

        if screen_hash not in self.visited_areas:
            self.visited_areas.add(screen_hash)  # Mark the area as visited
            return True
        return False

    def render(self, mode="human"):
        """
        Render the environment for debugging or visualization.
        """
        if mode == "human":
            # Display the game screen
            self.pyboy.set_window_type("SDL2")

    def close(self):
        """
        Properly shut down the emulator.
        """
        if hasattr(self, "pyboy") and self.pyboy:
            self.pyboy.stop()
    def _check_battle_trigger(self):
        """
        Determine if a battle is triggered based on the game state.
        """
        # Customize this logic as per your game
        return np.random.rand() < 0.05  # Example: 5% chance of triggering a battle
