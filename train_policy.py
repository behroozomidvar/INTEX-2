# This is the main Python script to learn text-based exploration policies.

from data.db_interface import DBInterface
# This library provides the definition for the environment class (state transitions and rewards).
from rl.environment import Environment
# This library provides definitions for the agent class (policy learner).
from rl.agent import Agent
import numpy as np
import json

with DBInterface("imdb") as db_interface:
    env = Environment(db_interface)
    # In the current version of the code, we simply consider the target to be a single item (e.g., "862088").
    target_item_id = db_interface.get_random_item_id()
    env.set_target(target_item_id)
    # The variable "explorer" is an instantiation of the agent class.
    explorer = Agent(env, db_interface.get_random_item())
    policy = explorer.learn_policy()
    # This will create a file called "policy.npy" in the current directory.
    np.save("model/policy", policy)

    training_data = {
        "target_item_id": target_item_id
    }

    with open('model/training_data.json', 'w') as outfile:
        json.dump(training_data, outfile, indent=4)
