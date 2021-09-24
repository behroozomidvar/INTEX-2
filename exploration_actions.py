# This library file contains all implementations related to text-based exploration actions.
# The exploration function receives an element "input_element" and output k other elements "output_elements".
# Both "input_element" and "output_elements" consist of reviews IDs.
# NOTE: In the current version of the code, only "review exploration" is implemented.

from data.db_interface import DBInterface
import configuration 	# This library contains all configurable parameters.
import utilities		# This library consists of helper functions.
import statistics
import time
import pandas as pd
import random
import intex_experiments


class exploration():

    def __init__(self, db_interface: DBInterface):

        # All exploration parameters inherited from the "configuration" library
        self.config = configuration.exploration_configurations

        # Lazy loading mechanism is in place, i.e., the data will be loaded only when needed.
        self.data = None

        # This is the number if exploration actions, which is equal to the number of relevance
        # ... functions times the number of quality functions.
        self.nb_actions = self.config["nb_relevance_functions"] * \
            self.config["nb_quality_functions"]

        self.db_interface = db_interface

    # This is the main exploration function, given all necessary parameters to execute exploration.
    # input_element: 			The current item/review under investigation
    # relevance_function: 		The type of relevance computation
    # k_prime:					The maximum number of relevant items to retrieve (for efficiency reasons)
    # k: 						The number of elements in the output / The minimum number of relevant items to retrieve
    # timelimit:				The time limit for the optimization process in the quality function.
    # quality_function:			The type of quality computation
    # optimization direction:	It defines if the optimization is maximization or minimization.
    def parametrized_explore(self, input_element, relevance_function, k_prime, k, timelimit, quality_function, optim_loops, optim_meter, optimization_direction):

        # Step 1 of exploration: Obtain a list of candidates which are relevant to the input element
        elements_shortlist = self.db_interface.get_relevant_elements(
            input_element, relevance_function, k_prime)

        nb_items_received = len(
            elements_shortlist[elements_shortlist["type"] == "item"]) / k_prime
        # print(nb_items_received)

        # Step 2 of exploration: Obtain a list of k optimized elements with respect to the quality function
        output_elements = self.compute_quality(
            elements_shortlist, k, timelimit, quality_function, optim_loops, optim_meter, optimization_direction)

        return output_elements

    # Functions outside the "exploration" often call the following simplified function of exploration, which
    # ... runs by default values set in the configuration library.
    def explore(self, input_element):

        # Call the "parametrized_explore" function by filling the parameters from the configuration library
        output_elements = self.parametrized_explore(input_element, self.config["relevance_function"],
                                                    self.config["k_prime"], self.config["k"], self.config["time_limit"],
                                                    self.config["quality_function"], self.config["nb_optimization_loops"],
                                                    self.config["optimization_meter"], self.config["optimization_direction"])

        return output_elements

    # Another variant is "exploration_by_functions" where the exploration is executed by determining the
    # ... relevance and quality functions (and not other paramters.)
    # With "options", the functionality of operators can be limited, for experimental purposes.
    def explore_by_functions(self, input_element, relevance_function, quality_function):

        # apply exploration operator variants TEXT, TSG, ATTRIB, ALL
        relevance_function = intex_experiments.apply_operator_variant(
            relevance_function)

        # Call the "parametrized_explore" function by filling the parameters from the configuration library
        output_elements = self.parametrized_explore(input_element, relevance_function,
                                                    self.config["k_prime"], self.config["k"], self.config["time_limit"],
                                                    quality_function, self.config["nb_optimization_loops"],
                                                    self.config["optimization_meter"], self.config["optimization_direction"])

        return output_elements

    def compute_quality(self, elements_shortlist, k, timelimit, quality_function, optim_loops, optim_meter, optimization_direction):

        # We initialize output elements with top-k most relevant elements.
        output_element_ids = elements_shortlist.iloc[0:k].id.to_list()

        # If the quality function "none" is selected, then it suffices to return the
        # ... top-k most relevant elements.
        if quality_function == "none":
            return elements_shortlist.iloc[0:k]

        # In the quality improvement loops, the "cursor" variable shows which candidate element should be selected next.
        cursor = k

        # We accumulate time in "time_spent" and we stop if we reach the time limit.
        time_spent = 0

        # We count how many quality improvement loops are
        # ... performed within the time limit.
        loop_count = 0

        if quality_function == "diverse_numerical":
            elements_data = elements_shortlist.set_index('id').rating.to_dict()
        else:  # "diverse_review" or "coverage_review"
            elements_data = elements_shortlist.set_index('id').text.to_dict()

        candidate_ids = list(elements_data.keys())

        # Quality score is a value between 0 and 1. Our aim is to maximize this value, in
        # ... case optimization_direction = "max" (and to minimize, otherwise).
        # The function "compute_quality_score" computes this score.
        current_quality_score = self.compute_quality_score(
            quality_function, output_element_ids, elements_data)

        # The quality improvement loop begins here. It will go on until the time limit is not exceeded.
        while ((time_spent < timelimit and optim_meter == "timelimit") or (loop_count < optim_loops and optim_meter == "nb_optimization_loops")):

            start_time = time.time()
            if not candidate_ids[cursor] in output_element_ids:
                # This inner loop checks for possible replacements in the "output_elements" to improve the quality score.
                for i in range(k):
                    # Replacement of the ith output element by the candidate at cursor for quality evaluation
                    candidate_output_element_ids = output_element_ids.copy()
                    candidate_output_element_ids[i] = candidate_ids[cursor]
                    # We obtain the score of the list with the replacement, to compare with the current "output_elements".

                    candidate_quality_score = self.compute_quality_score(
                        quality_function, candidate_output_element_ids, elements_data)

                    # Based on the optimization direction, the "improved quality" returns True if a better
                    # ... quality score is obtained.
                    if self.improved_quality(candidate_quality_score, current_quality_score, optimization_direction) == True:
                        output_element_ids = candidate_output_element_ids
                        current_quality_score = candidate_quality_score
                        break

            # Cursor will now point to the next element in the shortlist.
            cursor += 1

            # If the cursor reaches the end of the shortlist, we simply break the quality improvement loop.
            if cursor == len(elements_shortlist):
                break

            end_time = time.time()
            time_spent += (end_time - start_time)
            loop_count += 1

        return elements_shortlist[elements_shortlist.id.isin(output_element_ids)]

    # The function returns a value between 0 and 1, depending on the semantics of the quality function.

    def compute_quality_score(self, quality_function, element_ids, elements_data):

        quality = 0
        data = list(map(lambda x: elements_data[x], element_ids))
        if quality_function == "diverse_numerical":
            quality = statistics.stdev(data)
        elif quality_function == "diverse_review":
            quality = utilities.collective_jaccard(data)
        elif quality_function == "coverage_review":
            quality = utilities.unique_word_count(data)

        return quality

    # Return True if a better quality score is obtained.

    def improved_quality(self, new_quality_score, old_quality_score, optimization_direction):

        improved = False

        if new_quality_score > old_quality_score and optimization_direction == "max":
            improved = True
        elif new_quality_score < old_quality_score and optimization_direction == "min":
            improved = True

        return improved
