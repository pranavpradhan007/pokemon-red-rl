from environments.battle_env import BattleEnv

# Initialize the environment
env = BattleEnv()

# Reset the environment and start testing
obs = env.reset()
done = False
step_count = 0

try:
    while not done:
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)

        # Debugging information
        step_count += 1
        print(f"Step: {step_count}, Action: {action}, Reward: {reward}, Obs Shape: {obs.shape}")

        # Optional: Stop after a certain number of steps for testing
        if step_count > 100:
            break

except KeyboardInterrupt:
    print("Testing interrupted manually.")

finally:
    env.close()  # Ensure emulator shuts down properly
    print("Environment closed.")
