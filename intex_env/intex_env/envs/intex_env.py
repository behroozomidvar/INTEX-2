# This library provides the definition for the environment class (state transitions and rewards).
# This is an instantiation of the gym class.
# To install this environment, execute "pip install -e .".

import random
import math
import configuration
import exploration_actions
import gym
import numpy as np
import pandas as pd
import utilities  # This library consists of helper functions.
import moo  # Multi-objective optimization library

# This library consists of several functions for interacting with the data at online time.
# This library consists of functions related to text-based exploration actions.
from data import data_utilities
from data.db_interface import DBInterface

from rl import boolean_state_features as state_features # boolean features
# from rl import state_features # non-boolean features


class intex_env(gym.Env):

    # Metadata required to match with gym
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # All environment parameters inherited from the "configuration" library
        self.config = configuration.environment_configurations

        # Current state employed in the environment
        self.state = np.zeros(self.config["nb_state_features"])

        # Reward provided by the environment in the current state
        self.reward = 0

        # The input element employed by the environment. By default, it is set
        # ... to -1, i.e., no input element yet.
        self.input_element = -1

        # The output element received from the exploration action
        self.output_elements = []

        # In the current version of the code, we simply consider the target to
        # ... be a single item (e.g., "862088"). That target is first initialized
        # ... as empty, but later gets a value, once the environment class is
        # ... instantiated.
        self.targets = []

        # This dictionary maps each relevance function code to its name.
        self.relevance_function_dic = {0: "sim", 1: "summary_sim",
                                       2: "sentiment_sim", 3: "tag_sim", 4: "topic_sim", 5: "attribute_sim"}
        # This dictionary maps each quality function code to its name.
        self.quality_function_dic = {
            0: "none", 1: "diverse_numerical", 2: "diverse_review", 3: "coverage_review"}
        # This dictionary maps the element types
        # self.element_type_dic = {0: "item", 1: "review"}
        self.done = False

        self.discarded_steps = 0

    # This function determines the starting point of exploration and the target
    def initialize(self, k, target_element_ids, reward_variant, db_interface: DBInterface, input_element_selection_strategy='random', reward_power=1, eval_mode=False):
        self.db_interface = db_interface
        self.eval_mode = eval_mode
        self.input_element_selection_strategy = input_element_selection_strategy
        self.state_encoder = state_features.StateEncoder(db_interface)
        # An instance of the "exploration" class to handle exploration actions
        # ... in the environment.
        self.exploration = exploration_actions.exploration(self.db_interface)
        self.output_element_count = k
        self.initial_targets = target_element_ids
        self.reward_power = reward_power

        if self.config["show_targets"]:
            print("Targets:")
            for attributes in self.db_interface.get_elements(target_element_ids).attributes:
                try:
                    print(f"   - {attributes['title']}")
                except:
                    continue

        # Control when the "done" variable should be set to true.
        self.reach_size = int(len(self.initial_targets)
                              * self.config["nb_reach"])

        self.reward_variant = reward_variant

        self.reward_buffer = []

    # The "step" function receives as input as (mystified) exploration action, applies
    # ... it to the current input element stored in the environment, returns the new
    # ... state and the reward, and update the internal state of the environment.
    def step(self, exploration_action):

        # Demystify the input exploration action
        quality_function, relevance_function, input_element_index = self.demystify_action(
            exploration_action)
        self.quality_function_counters[quality_function] += 1
        self.relevance_function_counters[relevance_function] += 1
        self.select_input_element(
            self.output_elements, input_element_index)

        self.output_elements = self.exploration.explore_by_functions(
            self.input_element, relevance_function, quality_function)

        # Build the state of the environment

        target_reach = self.get_found_target_count() / len(self.initial_targets)

        self.state = self.state_encoder.state_feature_representation(
            self.input_element, self.output_elements, target_reach)

        self.reward, reward_info = self.get_reward()

        if reward_info == "neutral":
            self.discarded_steps += 1

        self.cumulated_reward += self.reward
        self.update_seen_counters()

        if len(self.targets) == self.reach_size:
            self.done = True

        return [self.state, self.reward, self.done, {}]

    # The reset function is necessary when a new episode begins.
    def reset(self):
        self.discoveries = []
        self.discarded_steps = 0
        self.review_ids_seen = set()
        self.item_ids_seen = set()
        start_elements = self.db_interface.get_episode_start_elements(
            self.output_element_count)
        self.input_element = start_elements.iloc[0]
        self.output_elements = start_elements.iloc[1:]
        self.state = self.state_encoder.state_feature_representation(
            self.input_element, self.output_elements)
        self.targets = self.initial_targets.copy()
        # self.state = np.zeros((self.config["nb_state_features"],))
        self.reward = 0
        self.cumulated_reward = 0
        self.done = False
        # Action counters for logs
        self.quality_function_counters = {
            "none": 0, "diverse_numerical": 0, "diverse_review": 0, "coverage_review": 0}
        self.relevance_function_counters = {
            "sim": 0, "summary_sim": 0, "sentiment_sim": 0, "tag_sim": 0, "topic_sim": 0, "attribute_sim": 0}
        return self.state

    def update_seen_counters(self):
        item_ids = set(
            self.output_elements[self.output_elements.type == "item"].id.to_list())
        self.item_ids_seen = self.item_ids_seen | item_ids
        review_ids = set(
            self.output_elements[self.output_elements.type == "review"].id.to_list())
        self.review_ids_seen = self.review_ids_seen | review_ids

    def choose_random_action(self):
        return random.randint(0, self.exploration.nb_actions-1)

    def render(self):
        print(self.state)

    def get_action_space_size(self):
        if self.input_element_selection_strategy == 'agent_selection':
            # * len(self.element_type_dic)
            return len(self.relevance_function_dic) * len(self.quality_function_dic) * self.output_element_count
        else:
            # * len(self.element_type_dic)
            return len(self.relevance_function_dic) * len(self.quality_function_dic)

    # Given a mystified exploration action ID, this function demystifies the ID into
    # ... a relevance function and a quality function.
    def demystify_action(self, exploration_action_mystified):
        # Obtain singleton codes associated to the element type and relevance and quality functions
        relevance_function_code = exploration_action_mystified % len(
            self.relevance_function_dic)
        if self.input_element_selection_strategy == 'agent_selection':
            quality_function_code = exploration_action_mystified // len(
                self.relevance_function_dic) % len(self.quality_function_dic)
            input_element_index = exploration_action_mystified // len(
                self.relevance_function_dic) // len(self.quality_function_dic)
        else:
            quality_function_code = exploration_action_mystified // len(
                self.relevance_function_dic)
            input_element_index = -1
        return self.quality_function_dic[quality_function_code], self.relevance_function_dic[relevance_function_code], input_element_index

    # Pick a random element from the output elements.
    # Current strategies are "random" and "rating" (choose the element with top rating)
    def select_input_element(self, output_elements, input_element_index):
        if len(self.output_elements) != 0:
            if self.input_element_selection_strategy == 'agent_selection':
                self.input_element = self.output_elements.iloc[input_element_index]
            elif self.input_element_selection_strategy == "rating":
                self.input_element = output_elements.loc[int(
                    output_elements["rating"].idxmax())]
            else:
                # random strategy
                picked = utilities.roll_dice(0, len(output_elements))
                self.input_element = output_elements.iloc[picked]

    # In policy evalaution, only the non-exponent target reward operates.
    def get_evaluation_reward(self):
        evaluation_reward = 0
        for output_element in self.output_elements.id.tolist():
            if output_element in self.targets:
                evaluation_reward += 1
                self.targets.remove(output_element)
        return evaluation_reward

    # Given a new input element, this function computes the reward associated to
    # ... this new element.
    def get_reward(self):
        
        if self.eval_mode:
            return self.get_evaluation_reward(), "eval"
        
        else: # in training mode
            elements = set(self.output_elements.id.tolist())
            if self.config["reward_for_seen"]:
                elements = list(elements)
            else:
                elements = list(
                    elements - (self.review_ids_seen | self.item_ids_seen))
            if len(elements) == 0:
                return 0, "empty"
            else:
                all_rewards = data_utilities.RewardManager(
                    elements, self.targets, db_interface=self.db_interface)
                if self.reward_variant == "SCL":
                    return math.pow(all_rewards.mean_reward, self.reward_power), "scl"

                # multi-objective optimization of rewards.
                reward_vector = [all_rewards.text_reward, all_rewards.summary_reward,
                                 all_rewards.topic_reward, all_rewards.sentiment_reward, all_rewards.tag_reward]
                
                # in case the reward buffer is not full
                if len(self.reward_buffer) < self.config["reward_buffer_size"]:
                    if moo.buffer_eligible(reward_vector, self.reward_buffer):
                        for d in moo.buffer_dominance(reward_vector, self.reward_buffer):
                            self.reward_buffer.remove(d)
                        self.reward_buffer.append(reward_vector)
                        return math.pow(all_rewards.mean_reward, self.reward_power), "moo"
                    else:
                        # This is a signal that the reward is not accepted and hence the whole action should be reversed.
                        return self.config["neutral_reward"], "neutral"
                else: # in case the reward buffer is full
                    self.reward_buffer = []
                    return math.pow(all_rewards.mean_reward, self.reward_power), "moo"

    def get_found_target_count(self):
        return len(self.initial_targets) - len(self.targets)
