# This library consists of helper functions.


import numpy as np
from numpy import exp
import sys
import configuration 


# ***** PART 1: INTERNAL HELPER FUNCTIONS FOR QUALITY FUNCTIONS *****

# In case quality_function is "diverse_review", we compute the diversity of reviews
# ... using collective Jaccard.
def collective_jaccard(text):

    words = []
    quality_score = 0

    # Add all words in the reviews to the list "words"
    # Each review has its own set of words.
    # Note that obviously the words could be repetitive.
    for review in text:
        review_words = review.split(" ")
        words.append(review_words)

    # Check a pair of reviews at a time, and compute the intersection and the union.
    # The Jaccard diversity score (between each pair of reviews) is obtained by
    # ... dividing the intersection over the union.
    cnt = 0
    for i in range(0, len(words)):
        for j in range(i+1, len(words)):
            intersected_words = [word for word in words[i] if word in words[j]]
            union_words = words[i] + words[j]
            pairwise_jaccard = 1 - (len(intersected_words) / len(union_words))
            quality_score += pairwise_jaccard
            cnt += 1

    # The overall Jaccard score is the average of individual score, i.e, the sum
    # ... divided by the count. For any reason, if the count remained zero, the
    # ... function returns 0 (i.e., zero quality).
    try:
        quality_score /= cnt
        return quality_score
    except:
        return 0

# In case quality_function is "coverage_review", we compute the coverage of reviews
# ... using the unique word count.


def unique_word_count(reviews):

    max_nb_words = configuration.exploration_configurations["max_nb_words"]                  

    unique_words = []

    for review in reviews:
        review_words = review.split(" ")
        unique_words = unique_words + review_words

    quality = min(1, float(len(unique_words) / max_nb_words))
    return quality


# ***** PART 2: FUNCTIONS FOR AN AMELIORATED PRESENTATION OF RESULTS *****

# Review sentiment is a vector of size 10 which shows the sentiment of reviews in
# ... different degrees, from "very negative" to "neutral" to "very positive". In
# ... the database, this vector is recorded as a string where values are separated
# ... by space. The following function presents this vector (admitted as
# ... "sentiment_result") in a human-understandable form.
# In case verbose = True, the labels are also shown for the sentiments.
# An example sentiment string for review "445459": "0 0 0 0 8 18 11 6 0 0 "
def prettify_sentiment(sentiment_list, verbose=True):
    # Step 2: To simplify presentation, we reduce the number of values to 5 (instead of 10).
    small_list = []
    for i in (0, 2, 4, 6, 8):
        small_list.append(sentiment_list[i]+sentiment_list[i+1])

    # Step 3: Fit the 5 sentiment values into a Softmax function to obtain
    # ... a sentiment distribution.
    np_result = np.array(small_list)
    dist = softmax(np_result)

    # Step 4: Change the sentiment distribution values to percentage values
    dist_percent = []
    for s in dist:
        dist_percent.append(round(s * 100, 2))

    if verbose == False:
        return dist_percent

    # Step 5: Associate labels to the sentiment percentage values
    out_dic = {"very negative": dist_percent[0], "negative": dist_percent[1], "neutral": dist_percent[2],
               "positive": dist_percent[3], "very positive": dist_percent[4]}

    # Step 6: Sort the labeled percentage values, from the largest to the lowest
    sorted_out_dic = dict(
        sorted(out_dic.items(), key=lambda item: item[1], reverse=True))

    # Step 7: Make the output as a string which concatenates labels and values
    out_str = ""
    for s in sorted_out_dic:
        out_str += str(sorted_out_dic[s])+"% " + s + ", "

    return out_str[:-2]

# Review topic is a vector of size 10 which represents the topic distribution of the review.
# This function returns a Softmax distribution over these topics.
# The output of the function is a list.
# An example topic string for review "483880": "0.0031263358 0.0031260687 0.9718651 0.0031260373
# ... 0.0031258801 0.0031257705 0.0031261435 0.0031261863 0.003126235 0.0031261854 "


def prettify_topic(topic_result):

    # Step 1: The variable "topic_list" hosts all topic values cast as float.
    topic_list_str = topic_result[0].split()
    topic_list = []
    for s in topic_list_str:
        topic_list.append(float(s))

    # Step 2: Fit the 10 topic values into a Softmax function to obtain
    # ... a topic distribution.
    np_result = np.array(topic_list)
    dist = softmax(np_result)
    dist_normal = []
    for s in dist:
        dist_normal.append(round(s, 2))

    return dist_normal

# In the database, we keep top representative words for each topic, i.e.,
# ... topic definitions. This function renders these words in a
# ... human-understandable form.


def prettify_topic_definition(topic_definitions):

    output = []

    for topic_definition in topic_definitions:
        out_clean = ""

        # Separate parts of the definition using "\"
        separated = topic_definitions[topic_definition].split("\"")

        # In case the separated part is a word, it will be added to the final
        # ... output, "out_clean"
        for s in separated:
            if s.find("+") == -1 and s.find(" ") == -1 and s.find("*") == -1:
                out_clean += s + ", "
        output.append(out_clean[:-3])

    return output

# This shows a review (identified by "review_id") in a human-understandable form.
# Note: This function is only functional in terminal. The equivalent functionality
# ... for the GUI is provided in the class "show_review" in the library "data_utilities".
# review_id:        The ID of the review under investigation
# output_position:  The position where this review should be shown (between 1 and k)
# show_review_text: If True, the textual content of the review will be printed as well.


# def show_review_details(review_id, output_position, show_review_text=False):

#     # Step 1: Get the information about the item (product) under review
#     product_info_query = queries.get_query("product_info", [review_id])
#     product_info = database_query.execute_query(
#         "product_info", product_info_query)

#     # Step 2: Get the information about the review identified with "review_id"
#     reviews_info_query = queries.get_query("reviews_info", [review_id])
#     reviews_info = database_query.execute_query(
#         "reviews_info", reviews_info_query)

#     # Step 3: Get the sentiments of the review "review_id"
#     sentiment_query = queries.get_query("sentiment", [review_id])
#     sentiment = prettify_sentiment(
#         database_query.execute_query("sentiment", sentiment_query))

#     # Step 4: Get the topics of the review "review_id"
#     topic_query = queries.get_query("topic", [review_id])
#     topic = prettify_topic(database_query.execute_query("topic", topic_query))

#     # Step 5: Print the review information
#     print("\n*** Showing review "+str(output_position)+" ***")
#     print("Review ID: "+str(review_id))
#     if product_info != False:
#         print("Product: "+product_info[1]+" ("+product_info[0]+")")
#     if product_info != False and product_info[3] != "":
#         print("Price: "+product_info[3])
#     if product_info != False:
#         print("Category: "+str(product_info[2]))
#     print("Rating: "+str(reviews_info[1])+" / 5.0")
#     print("Review summary: "+reviews_info[2])
#     print("Sentiment polarization: "+str(sentiment))
#     print("Topic distribution: "+str(topic))
#     if show_review_text == True:
#         print("\nReview text: "+reviews_info[3])
#     print()

# # Given the list of categories for an item (product), this function delivers
# # ... a clean representation of them.


def prettify_category(categories):
    cats = categories.replace("[", "")
    cats = cats.replace("]", "")
    cats = cats.replace("'", "")
    cats_parts = cats.split(",")
    cats_str = ", ".join(cats_parts)
    return cats_str


# ***** PART 3: OTHER HELPER FUNCTIONS *****

# Given a set of keywords, this function prepares the necessary string to
# ... query reviews and find ones which contain those keywords.
def keywords_to_query(keywords):
    keywords = keywords.strip()
    keywords = keywords.replace(" ", "%")
    query_parameters = "%"+keywords+"%"
    return query_parameters

# This function is a simple randomization function. In case a min-max limit
# ... is defined, the function returns a random between min and max. Otherwise,
# ... returns a value truly at random.


def roll_dice(min_=False, max_=False):
    if min_ == False and max_ == False:
        return np.random.random()
    return np.random.randint(min_, max_)

# This is a simple implementation of the Softmax function. The function is handy
# ... where for any reason, we're not interested to import numpy.


def softmax(vector):
    e = exp(vector)
    return e / e.sum()

# Given a relevance function ("sim", "tag_sim", etc.), this function returns a
# ... textual description of the function.


def relevance_function_explain(relevance_function):
    relevance_function_label = {"sim": "textual content", "summary_sim": "summary",
                                "sentiment_sim": "sentiments", "tag_sim": "auto-generated tags", "topic_sim": "topics", "attribute_sim": "attributes"}
    return relevance_function_label[relevance_function]

import sys

# Colored printing functions for strings that use universal ANSI escape sequences.
# fail: bold red, pass: bold green, warn: bold yellow, 
# info: bold blue, bold: bold white

class ColorPrint:

    @staticmethod
    def print_fail(message, end = '\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_title(message, end = '\n'):
        sys.stdout.write('\n\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_episode(episode_number, R, targets_found, perc, end = '\n'):
        sys.stderr.write('\x1b[1;33m'+'episode\x1b[0m: ' + str(episode_number) 
            + '\t\x1b[1;33m R\x1b[0m: ' + str(R) + '\t\x1b[1;33m targets found\x1b[0m: ' + str(targets_found) + ' (' + str(perc) + '%)' + end)

    @staticmethod
    def print_info(message, end = '\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_param(message, param, end = '\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m: ' + str(param) + end)

    @staticmethod
    def print_bold(message, end = '\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)

def parameter_category(parameter):
    if parameter in configuration.learning_configurations.keys():
        return "learning"
    if parameter in configuration.exploration_configurations.keys():
        return "exploration"
    if parameter in configuration.environment_configurations.keys():
        return "environment"
    return ""