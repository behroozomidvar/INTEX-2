# This library provides the definition for the environment class (state transitions and rewards).

# This library contains all configurable parameters.
import configuration
# This library consists of functions related to text-based exploration actions.
import exploration_actions

import utilities            # This library consists of helper functions.
from data import data_utilities, db_interface


class Environment():

    def __init__(self, db_interface):
        self.db_interface = db_interface
        # All environment parameters inherited from the "configuration" library
        self.config = configuration.environment_configurations

        # An instance of the "exploration" class to handle exploration actions
        # ... in the environment.
        self.exploration = exploration_actions.exploration(db_interface)

        # In the current version of the code, we simply consider the target to
        # ... be a single item (e.g., "862088"). That target is first initialized
        # ... as empty, but later gets a value, once the environment class is
        # ... instantiated.
        self.target = ""

    # Given a new input element, this function computes the reward associated to
    # ... this new element.
    def get_reward(self, new_input_element):

        rewards = data_utilities.RewardManager(
            new_input_element.id, self.target, self.db_interface)
        out_reward = rewards.mean_reward
        return out_reward, rewards.done

    # Exploration actions are a combination of a choice for the relevance function
    # ... and another choice for the quality function. This combination should be
    # ... encoded to obtain one unique "exploration action ID". This is what this
    # ... function "mystify_action" does.
    def mystify_action(self, relevance_function, quality_function):

        # This dictionary maps each relevance function to a code in the range [0, 4].
        relevance_function_inverse_dic = {
            "sim": 0, "summary_sim": 1, "sentiment_sim": 2, "tag_sim": 3, "topic_sim": 4, "attribute_sim": 5}

        # This dictionary maps each quality function to a code in the range [0, 3].
        quality_function_inverse_dic = {
            "none": 0, "diverse_numerical": 1, "diverse_review": 2, "coverage_review": 3}

        # The two codes for the relevance and quality functions will combined to obtain one unique ID.
        exploration_action_mystified = quality_function_inverse_dic[quality_function] * \
            len(relevance_function_inverse_dic) + \
            relevance_function_inverse_dic[relevance_function]

        return exploration_action_mystified

    # Given a mystified exploration action ID, this function demystifies the ID into
    # ... a relevance function and a quality function.
    def demystify_action(self, exploration_action_mystified):

        # This dictionary maps each relevance function codes to its name.
        relevance_function_dic = {0: "sim", 1: "summary_sim",
                                  2: "sentiment_sim", 3: "tag_sim", 4: "topic_sim", "attribute_sim": 5}

        # This dictionary maps each quality function code to its name.
        quality_function_dic = {
            0: "none", 1: "diverse_numerical", 2: "diverse_review", 3: "coverage_review"}

        # Obtain singleton codes associated to the relevance and quality functions
        relevance_function_code = int(
            round(exploration_action_mystified % 4, 0))
        quality_function_code = int(round(exploration_action_mystified / 4, 0))

        # In case for any reason the code for the functions are larger
        # ... than the number of available functions, we cap it to the code
        # ... for the last function.
        if relevance_function_code > 4:
            relevance_function_code = 4
        if quality_function_code > 3:
            quality_function_code = 3

        return quality_function_dic[quality_function_code], relevance_function_dic[relevance_function_code]

    # Given the input element and the output elements as the result of applying
    # ... an exploration action, this function returns the unique which represents
    # ... the state the agent is currently placed in.

    def get_bucketized_state(self, output_elements, input_element, exploration_action):

        # Get the textual content of the reviews for the output elements
        output_elements_texts = output_elements.text.to_list()

        # Get the textual content of the review for the input element
        input_element_texts = input_element.text

        # In the current version of the code, we assume (heuristically) that a state is
        # ... represented using four following features: diversity of output elements (F1),
        # ... number of unique words in output elements (F2), number of unique words in
        # ... the input element (F3), and the chosen exploration action (F4). Note that
        # ... this feature model is very preliminary and should definitely evolve.

        # Compute the diversity among the reviews of output elements (i.e., F1)
        output_elements_diversity = round(
            utilities.collective_jaccard(output_elements_texts) / 0.25, 0) - 1

        # Compute the number of unique words for both the input element and output elements (i.e., F2 and F3)
        max_possible_words = 1000
        output_elements_unique_word_count = round(utilities.unique_word_count(
            output_elements_texts) / (max_possible_words * 0.25), 0)
        input_element_unique_word_count = round(utilities.unique_word_count(
            input_element_texts) / (max_possible_words * 0.25), 0)

        # Normalize the word count for both input elements and output elements.
        if output_elements_unique_word_count > 3:
            output_elements_unique_word_count = 3
        if input_element_unique_word_count > 3:
            input_element_unique_word_count = 3

        # Compute the exploration action as the feature F4
        action_for_state = round((exploration_action) / 5, 0) - 1

        # The state is compiled as a list of F1, F2, F3, and F4.
        state = [int(output_elements_diversity), int(output_elements_unique_word_count), int(
            input_element_unique_word_count), int(action_for_state)]

        return state

    # Generate a random state (specially at the beginning of a session)
    def random_state(self):

        output_state = []
        nb_buckets = self.config["nb_buckets"]

        for _ in range(len(self.config["state_features"])):
            output_state.append(utilities.roll_dice(0, nb_buckets))

        return output_state

    def set_target(self, target):
        self.target = target

    # This function implements the simulated situation where one element is
    # ... chosen among k elements of "output_elements". This normally should
    # ... be implemented as rolling a dice, but currently it always returns
    # ... a fixed value.
    def get_feedback(output_elements):
        return "3333"
