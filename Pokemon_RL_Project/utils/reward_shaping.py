def calculate_overworld_reward(state):
    return 1.0 if state["new_area"] else -0.1
