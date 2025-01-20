import logging
from agents.hierarchical_agent import HierarchicalAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    try:
        logging.info("Starting Full Game Execution...")
        agent = HierarchicalAgent()
        agent.run()
    except Exception as e:
        logging.error("An error occurred during full game execution:", exc_info=True)
    finally:
        logging.info("Script execution completed.")
