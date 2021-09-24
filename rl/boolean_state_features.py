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
        self.max_text_size = configuration.exploration_configurations["max_text_size"]
        self.max_tag_count = configuration.exploration_configurations["max_tag_count"]
        # self.max_text_size = db_interface.get_max_text_size()
        # self.max_tag_count = db_interface.get_max_tag_count()

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

    
    def encode_element(self, element):
        features = []
        
        # element type (1 bool feature)
        features.append(0 if element.type == "item" else 1)
        
        # rating (5 bool features)
        normalized_rating = element.rating
        if self.max_rating == 10:
            normalized_rating = int(normalized_rating / 2)
        bucketized_rating = self.bucketize(normalized_rating, nb_buckets = 5, min_value = 0, max_value = 5)
        for bucket_value in bucketized_rating:
            features.append(bucket_value)

        # text size (5 bool features)
        text_size = min(len(element.text), self.max_text_size)
        bucketized_text_size = self.bucketize(text_size, nb_buckets = 5, min_value = 0, max_value = self.max_text_size)
        for bucket_value in bucketized_text_size:
            features.append(bucket_value)
        
        # tag count (5 bool features)
        tag_count = min(len(element.tags), self.max_tag_count)
        bucketized_tag_count = self.bucketize(tag_count, nb_buckets = 5, min_value = 0, max_value = self.max_tag_count)
        for bucket_value in bucketized_tag_count:
            features.append(bucket_value)
        
        # sentiments vector (10 bool features)
        for sentiment_value in element.sentiments:
            features.append(1 if sentiment_value > 0 else 0)

        # topics vector (10 boold features)
        for topic_value in element.topics:
            features.append(1 if topic_value > 0 else 0)
        
        return features

    def state_feature_representation(self, input_element, output_elements, target_reach = 0):

        feature_representation = []

        # features of the input element
        feature_representation += self.encode_element(input_element)
        
        # features of the output element
        for index, output_element in output_elements.iterrows():
            feature_representation += self.encode_element(output_element)

        # features of the quality functions
        feature_representation.extend(self.bucket_quality(output_elements, "diverse_numerical"))  # 5 features
        feature_representation.extend(self.bucket_quality(output_elements, "diverse_review"))	  # 5 features
        feature_representation.extend(self.bucket_quality(output_elements, "coverage_review"))	  # 5 features

        # features for target (5)
        feature_representation.extend(self.bucketize(target_reach, nb_buckets = 5, min_value = 0, max_value = 1))

        feature_representation = np.array(feature_representation)
        return feature_representation



