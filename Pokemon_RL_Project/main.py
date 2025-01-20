from agents.hierarchical_agent import HierarchicalAgent

if __name__ == "__main__":
    agent = HierarchicalAgent("data/models/ppo_overworld.zip", "data/models/rainbow_battle.zip")
    agent.run()
