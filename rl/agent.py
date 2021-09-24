# This library provides definitions for the agent class (policy learner).

# This library contains all configurable parameters.
import configuration
# This library consists of functions related to text-based exploration actions.
import exploration_actions
import utilities            # This library consists of helper functions.
from rl.environment import Environment
import numpy as np
import random
from tqdm import tqdm


class Agent:

    def __init__(self, environment: Environment, start_element):
        # The agent keeps track of the current input element and one
        # ... in the previous iteration.
        self.current_input_element = start_element
        self.previous_input_element = self.current_input_element

        # The agent keeps track of the current output element and ones
        # ... in the previous iteration.
        self.current_output_elements = []
        self.previous_output_elements = []

        # The agent keeps track of the exploration action chosen in
        # ... the previous iteration.
        self.previous_action = None

        # In current version of the code, Q-learning is used to learn
        # ... policies. Hence a simple lookup table initialized here.
        self.q_table = None

        # The agent stores an instance of the environment.
        self.environment = environment

        # All configuration parameters for learning policies
        self.config = configuration.learning_configurations

        # Number of exploration actions
        self.nb_exploration_actions = environment.exploration.nb_actions

    # In this version of the code, we lean text-based exploration policies
    # ... using the Q-learning algorithm. Hence the output of the "lean_policy"
    # ... function is a lookup table called Q-table.
    def learn_policy(self):

        self.q_table = self.initialize_q_table()

        epsilon = self.config["epsilon"]
        print("Running training episodes")
        for episode in tqdm(range(self.config["nb_episodes"])):

            # A random state is generated to kick off this episode.
            current_state = self.environment.random_state()

            episode_reward = 0

            for iteration_count in range(self.config["episode_length"]):

                # Step 1: Rolling dice for the epsilon-greedy method
                action_chance = round(utilities.roll_dice(), 2)

                # Step 2: The exploration is either the one with optimized
                # ... value from the Q-table, or one at random, depending
                # ... on the value of the "action_chance" variable.
                exploration_action = np.argmax(self.pick_from_q_table(
                    current_state)) if action_chance > self.config["epsilon"] else utilities.roll_dice(0, self.nb_exploration_actions)

                # Step 3: Apply the exploration action
                # The exploration action is performed in the "get_state" function
                new_state, new_input_element, new_output_elements = self.get_state(
                    exploration_action)

                # Compute the reward given the new input element.
                reward, done = self.environment.get_reward(new_input_element)

                # Step 4: Accumulate reward by adding the recent reward
                # ... obtained after applying the exploration action.
                episode_reward += reward

                # Step 5: Update the agent
                self.set(new_input_element, new_output_elements,
                         exploration_action)

                # Step 6: Update the Q-table
                max_future_q = np.max(self.pick_from_q_table(new_state))
                current_q = self.pick_from_q_table(
                    current_state, exploration_action)
                new_q = (1 - self.config["alpha"]) * current_q + self.config["alpha"] * (
                    reward + self.config["gamma"] * max_future_q)
                self.q_table[current_state[0], current_state[1],
                             current_state[2], current_state[3], exploration_action] = new_q

                if done:
                    break

            # Epsilon decay
            if self.config["end_epsilon_decaying"] >= episode >= self.config["start_epsilon_decaying"]:
                epsilon -= self.config["epsilon_decay_value"]

            # Show learning progress
            if episode % self.config["show_every"] == 0:
                print("** cumulative reward for episode", episode,
                      ":", episode_reward, " epsilon: ", epsilon)

        return self.q_table

    # Initialize the Q-table given dimensions of number of exploration actions and
    # ... the number of state features.
    def initialize_q_table(self):

        # Step 1: Define Q-table size
        q_table_size = []
        for dimension in self.environment.config["state_features"]:
            q_table_size.append(
                len(self.environment.config["state_features"][dimension]))
        q_table_size.append(self.nb_exploration_actions)

        # Step 2: Initialization
        initialized_q_table = np.random.uniform(
            low=0, high=1, size=(q_table_size))

        return initialized_q_table

    # Transition to a new state by considering the current elements as
    # ... previous elements, and new elements as current ones.
    def set_new_state(self, new_input_element, new_output_elements):
        self.previous_input_element = self.current_input_element
        self.previous_output_elements = self.current_output_elements
        self.current_input_element = new_input_element
        self.current_output_elements = new_output_elements

    # Given a new state, this function updates the status of the agent.

    def set(self, new_input_element, new_output_elements, exploration_action):
        self.previous_input_element = self.current_input_element
        self.previous_output_elements = self.current_output_elements
        self.current_input_element = new_input_element
        self.current_output_elements = new_output_elements
        self.previous_action = exploration_action

    # Pick a random element from the output elements.
    # Note: This function can be implemented with more sophisticated
    # ... semantics than random.
    def pick_from_output_elements(self, output_elements):
        nb_choices = len(output_elements)
        picked = utilities.roll_dice(0, nb_choices)
        return output_elements.iloc[picked]

    # Given the exploration action (received as a mystified code),
    # ... this function determines the state.
    def get_state(self, exploration_action):

        # Step 1: Demystify the exploration action
        quality_function, relevance_function = self.environment.demystify_action(
            exploration_action)

        # Step 2: Collect the exploration parameters
        input_element = self.previous_input_element
        k_prime = self.environment.exploration.config["k_prime"]
        k = self.environment.exploration.config["k"]
        time_limit = self.environment.exploration.config["time_limit"]
        optimization_direction = self.environment.exploration.config["optimization_direction"]

        # Step 3: Perform the exploration to get the new set of output elements
        new_output_elements = self.environment.exploration.parametrized_explore(input_element,
                                                                                relevance_function, k_prime, k, time_limit, quality_function, optimization_direction)

        # If for any reason, the exploration function didn't return any result,
        # ... the previous output elements will be considered as the new elements.
        if len(new_output_elements) == 0:
            new_output_elements = self.previous_output_elements

        # Step 4: Pick one element our of k elements in the output elements
        new_input_element = self.pick_from_output_elements(new_output_elements)

        # Step 5: A new state is defined based on the new output elements,
        # ... the input element, and the exploration action.
        new_state = self.environment.get_bucketized_state(
            new_output_elements, new_input_element, exploration_action)

        return new_state, new_input_element, new_output_elements

    def pick_from_q_table(self, given_state, action=-1):
        if action == -1:
            return self.q_table[given_state[0], given_state[1], given_state[2], given_state[3]]
        return self.q_table[given_state[0], given_state[1], given_state[2], given_state[3], action]
