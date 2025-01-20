import logging
from environments.overworld_env import OverworldEnv
from environments.battle_env import BattleEnv

class HierarchicalAgent:
    def __init__(self):
        self.overworld_env = OverworldEnv("data/roms/PokemonRed.gb")
        self.battle_env = BattleEnv()
        self.battle_model = None  # Placeholder for a trained battle model

    def run(self):
        """
        Main execution loop for hierarchical agent, managing transitions between overworld and battle phases.
        """
        try:
            max_steps = 5000  # Total steps for the full game
            step_count = 0

            # Overworld phase
            logging.info("Starting overworld exploration...")
            obs = self.overworld_env.reset()
            done = False

            while step_count < max_steps:
                action = self.overworld_env.action_space.sample()  # Random action
                logging.info(f"Overworld action: {action}")
                obs, reward, done, info = self.overworld_env.step(action)
                logging.info(f"Overworld Reward: {reward}, Done: {done}, Info: {info}")
                step_count += 1

                if done or info.get("battle_trigger", False):
                    logging.info("Switching to battle phase...")
                    self.handle_battle()
                    logging.info("Returning to overworld phase...")
                    obs = self.overworld_env.reset()
                    done = False  # Reset `done` for overworld phase

            logging.info("Game execution completed.")

        except Exception as e:
            logging.error("Error during execution:", exc_info=True)
            raise
        finally:
            self.overworld_env.close()
            self.battle_env.close()


    def handle_battle(self):
        """
        Handles the battle phase using the battle environment and a pre-trained model.
        """
        logging.info("Starting battle phase...")
        obs = self.battle_env.reset()
        done = False
        steps = 0
        max_battle_steps = 100  # Set a limit for battle steps

        while not done and steps < max_battle_steps:
            if self.battle_model:
                action, _ = self.battle_model.predict(obs)  # Use trained model
            else:
                action = self.battle_env.action_space.sample()  # Fallback to random action

            logging.info(f"Battle Step: {steps}, Action: {action}")
            obs, reward, done, info = self.battle_env.step(action)
            logging.info(f"Battle Reward: {reward}, Done: {done}, Info: {info}")
            steps += 1

        if steps >= max_battle_steps:
            logging.warning("Battle phase ended due to step limit.")
        else:
            logging.info("Battle phase completed successfully.")

    def _is_battle_triggered(self, info):
        """
        Determines if a battle should be triggered based on overworld info.
        """
        logging.info(f"Checking battle trigger condition. Info: {info}")
        return info.get("battle_trigger", False)  # Adjust as per your overworld logic
