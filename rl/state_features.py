# This library contains functions to obtain the feature-based representation of the state.

from pandas.io.sql import execute
from data.db_interface import DBInterface
import utilities
import configuration
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict


class StateEncoder:
    def __init__(self, db_interface: DBInterface) -> None:
        self.dataset = db_interface.dataset
        self.max_rating = 10 if self.dataset == "imdb" else 5
        self.max_text_size = db_interface.get_max_text_size()
        self.max_tag_count = db_interface.get_max_tag_count()

        self.feature_count = 131

    def bucketize(self, value, nb_buckets, min_value, max_value):
        bucket_output = []
        step_size = (max_value - min_value) / nb_buckets
        current_min = min_value
        current_max = min_value + step_size
        value = value - 0.001  # for stability purposes
        while current_max <= max_value:
            if value >= current_min and value < current_max:
                bucket_output.append(1)
            else:
                bucket_output.append(0)
            current_min = current_max
            current_max = current_min + step_size
        return bucket_output

    def bucket_quality(self, elements, quality_function):
        element_texts = elements.text.to_list()
        element_ratings = elements.rating.to_list()
        quality_score = 0

        if quality_function == "diverse_numerical":
            quality_score = utilities.collective_jaccard(element_texts)
        if quality_function == "diverse_review":
            quality_score = statistics.stdev(element_ratings)
        if quality_function == "coverage_review":
            max_nb_words = configuration.exploration_configurations["max_nb_words"]
            quality_score = utilities.unique_word_count(
                element_texts) / float(max_nb_words)

        bucket_output = self.bucketize(quality_score, 5, 0, 1)

        return bucket_output

    # This function determines the similarity between the element and the target.
    # Note: In the current version of the implementation, it is assumed that both the element
    # ... and the target is of type "review".

    def target_attribute_reach(self, element, target):

        elements_to_consider = []
        if type(element) != list:
            elements_to_consider.append(element)
        else:
            elements_to_consider = element[:]

        sum_of_values = 0
        for element_to_consider in elements_to_consider:

            local_elements = [element_to_consider, target]
            exp = exploration_data.exploration_data(local_elements)
            reviews = exp.get_reviews()

            reviews_text = []

            for review_id in reviews:
                reviews_text.append(reviews[review_id])

            quality_score = 1.0 - utilities.collective_jaccard(reviews_text)
            sum_of_values += quality_score

        average_value = sum_of_values / float(len(elements_to_consider))
        bucket_output = self.bucketize(average_value, 5, 0, 1)

        return bucket_output

    def average_textual_values(self, element, textual_type):
        elements_to_consider = []
        if type(element) != pd.DataFrame:
            elements_to_consider = element.to_frame().T
        else:
            elements_to_consider = element

        if textual_type == "sentiments":
            sumed_values = [sum(item) for item in zip(
                *list(elements_to_consider.sentiments))]
        if textual_type == "topics":
            sumed_values = [sum(item)
                            for item in zip(*list(elements_to_consider.topics))]

        max_val = max(sumed_values)
        result = list(map(lambda x: round(x/max_val, 2), sumed_values))

        return result

    # Encodes an element into a 24 normalized features array
    def encode_element(self, element):
        features = []
        # element type - 1
        features.append(0 if element.type == "item" else 1)
        # rating - 1
        features.append(element.rating / self.max_rating)
        # text length - 1
        features.append(len(element.text)/self.max_text_size)
        # tag count - 1
        features.append(len(element.tags)/self.max_tag_count)
        # sentiments vector - 10
        max_sentiment = max(element.sentiments)
        features += list(map(lambda x: round(x/max_sentiment,
                                             2), element.sentiments))
        # topics vector - 10
        max_topic = max(element.topics)
        features += list(map(lambda x: round(x/max_topic, 2), element.topics))
        return features

    def state_feature_representation(self, input_element, output_elements, target):
        # Assume exploration_state is a tuple where input_element is the input element,
        # ... and output_elements is the output element.

        target = 0

        feature_representation = []

        feature_representation += self.encode_element(input_element)  # 24
        for index, output_element in output_elements.iterrows():
            # 24 * 3 = 72
            feature_representation += self.encode_element(output_element)

        feature_representation.extend(self.bucket_quality(
            output_elements, "diverse_numerical"))  # 5 features
        feature_representation.extend(self.bucket_quality(
            output_elements, "diverse_review"))	 # 5 features
        feature_representation.extend(self.bucket_quality(
            output_elements, "coverage_review"))	 # 5 features

        # Somehow redundant with the output_elements topics and sentiments encoding...
        feature_representation.extend(self.average_textual_values(
            output_elements, "topics")) 	 # 10 features
        feature_representation.extend(self.average_textual_values(
            output_elements, "sentiments"))  # 10 features

        # TARGET RELATED FEATURES
        # feature_representation.extend(target_attribute_reach(
        #     input_element, target)) 	 # 5 features
        # feature_representation.extend(target_attribute_reach(
        #     output_elements, target)) 	 # 5 features
        # # A boolean feature which is equal to "1" if the input element is a target element, otherwise "0"
        # feature_representation[65] = target_touched(input_element)
        # # A boolean feature which is equal to "1" if at least one of the output elements is a target element, otherwise "0"
        # feature_representation[66] = target_touched(output_elements)
        # This feature has four boolean values. It is a one-hot vector representing the coverage of the target with input elements selected since the beginning if the session.
        # If the coverage is lower than 0.25, the first value is "1". If it is between 0.25 and 0.5, the second value is "1", and vice versa.
        # feature_representation[75:79] = discovered_target_attributes_so_far(target)

        # REWARD RELATED FEATURES
        # It is a boolean feature. It is "1" is a reward larger than 0 is received.
        # feature_representation[160] = positive_reward()

        # ACTION RELATED FEATURES
        # It is a one-hot vector specifying which relevance function is employed.
        # feature_representation[161:166] = previous_relevance()
        # It is a one-hot vector specifying which quality function is employed.
        # feature_representation[167:172] = previous_quality()

        # REDUNDANCIES
        # Following features need to be implemented
        # A boolean feature which is equal to "1" if at least one sentiment value of the input element is larger than or equal to 0.8
        # Note: the sentiment values of the input element is already computed in line 131
        # feature_representation[67] = sentiment_target_reach(input_element, 0.8)

        # A boolean feature which is equal to "1" if at least one sentiment value of the input element is larger than or equal to 0.5
        # Note: the sentiment values of the input element is already computed in line 131
        # feature_representation[68] = sentiment_target_reach(input_element, 0.5)

        # A boolean feature which is equal to "1" if at least one topic value of the input element is larger than or equal to 0.8
        # Note: the topic values of the input element is already computed in line 133
        # feature_representation[69] = topic_target_reach(input_element, 0.8)

        # A boolean feature which is equal to "1" if at least one topic value of the input element is larger than or equal to 0.5
        # Note: the topic values of the input element is already computed in line 133
        # feature_representation[70] = topic_target_reach(input_element, 0.5)

        # A boolean feature which is equal to "1" if at least one sentiment value of the output elements is larger than or equal to 0.8
        # Note: the sentiment values of the output elements is already computed in line 132
        # feature_representation[71] = sentiment_target_reach(output_elements, 0.8)

        # A boolean feature which is equal to "1" if at least one sentiment value of the output elements is larger than or equal to 0.5
        # Note: the sentiment values of the output elements is already computed in line 132
        # feature_representation[72] = sentiment_target_reach(output_elements, 0.5)

        # A boolean feature which is equal to "1" if at least one topic value of the output elements is larger than or equal to 0.8
        # Note: the topic values of the output elements is already computed in line 134
        # feature_representation[73] = topic_target_reach(output_elements, 0.8)

        # A boolean feature which is equal to "1" if at least one topic value of the output elements is larger than or equal to 0.5
        # Note: the topic values of the output elements is already computed in line 134
        # feature_representation[74] = topic_target_reach(output_elements, 0.5)

        # It is the binary version of the features obtained in line 131. If a sentiment value is larger than 0.8, it becomes "1", otherwise "0".
        # feature_representation[80:89] = presence_of_high_sentiment(input_element, 0.8)

        # It is the binary version of the features obtained in line 131. If a sentiment value is larger than 0.5, it becomes "1", otherwise "0".
        # feature_representation[90:99] = presence_of_high_sentiment(input_element, 0.5)

        # It is the binary version of the features obtained in line 133. If a topic value is larger than 0.8, it becomes "1", otherwise "0".
        # feature_representation[100:109] = presence_of_high_topic(input_element, 0.8)

        # It is the binary version of the features obtained in line 133. If a topic value is larger than 0.5, it becomes "1", otherwise "0".
        # feature_representation[110:119] = presence_of_high_topic(input_element, 0.5)

        # It is the binary version of the features obtained in line 132. If a sentiment value is larger than 0.8, it becomes "1", otherwise "0".
        # feature_representation[120:129] = presence_of_high_sentiment(output_elements, 0.8)

        # It is the binary version of the features obtained in line 132. If a sentiment value is larger than 0.5, it becomes "1", otherwise "0".
        # feature_representation[130:139] = presence_of_high_sentiment(output_elements, 0.5)

        # It is the binary version of the features obtained in line 134. If a topic value is larger than 0.8, it becomes "1", otherwise "0".
        # feature_representation[140:149] = presence_of_high_topic(output_elements, 0.8)

        # It is the binary version of the features obtained in line 133. If a topic value is larger than 0.5, it becomes "1", otherwise "0".
        # feature_representation[150:159] = presence_of_high_topic(output_elements, 0.5)

        feature_representation = np.array(feature_representation)
        return feature_representation

    # env = environment.environment()
    # env.set_target("862088")
    # explorer = agent.agent(env)
    # e = exploration_actions.exploration()

    # x = 623218
    # X = [1314080, 664722, 64203]
    # state_features = np.array(0)

    # for iteration in range(1,5):
    # 	print("Iteration",iteration,"... ")
    # 	exploration_state = [x, X]

    # 	rep = state_feature_representation(exploration_state)
    # 	state_features = np.array(rep)

    # 	print("input element:", x)
    # 	print("output elements:", X)
    # 	print("state features:", state_features)

    # 	x = explorer.pick_from_output_elements(X)
    # 	X = e.explore(x)

    # 	reward = env.get_reward(x)

    # 	print("reward:", reward)

    # 	print()
