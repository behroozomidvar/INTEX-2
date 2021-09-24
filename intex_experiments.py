# This library defines the experiment variants for INTEX.

import random
import configuration
from data.db_interface import DBInterface

# If test set to True, the target will be returned for testing.
def target_query(test=False):

    # For Amazon: Top-5 rated items in "Computers" (T1) and "Camera & Photo" (T2) categories and their reviews
    # For IMDb: Top-5 rated items in "Comedy" (T1) and "Crime" (T2) categories and their reviews
    dataset = configuration.learning_configurations["dataset"]
    target_variant = configuration.learning_configurations["target_variant"]
    transfer_variant = configuration.learning_configurations["transfer_variant"]
    focus_attribute = ""
    focus_values = ""

    # By default the limit is to set to 5 to pick 5 items and their reviews. In testing we change the limit to
    # ... simulate specialization or generalization.
    limit = 5 

    if test and transfer_variant == "G2S": limit = 8
    if test and transfer_variant == "S2G": limit = 2

    item_offset_clause = "offset {}".format(limit) if test and transfer_variant == "DIFF" else ""
    element_offset_clause = "order by type desc offset 100" if test and transfer_variant == "SIM" else ""

    if dataset == "amazon":
        focus_attribute = "main_cat"
        focus_values = ["Computers"] if target_variant == "T1" else ["Camera &amp; Photo", "Camera & Photo"]
    
    else: # dataset = "imdb"
        focus_attribute = "genre"
        focus_values = ["Comedy"] if target_variant == "T1" else ["Crime"]

    value_condition = ""
    if len(focus_values) == 1:
        value_condition = "value='{}'".format(focus_values[0])
    else:
        for focus_value in focus_values:
            value_condition += "value='{}' or ".format(focus_value)
        value_condition = value_condition[:-3] # remove last "or"

    target_query = "select item_id from elements where item_id in (select element_id from attributes where name = '{}' and ({})) group by item_id order by sum(rating) desc {} limit {};".format(focus_attribute, value_condition,item_offset_clause, limit)
    target_element_ids = []

    with DBInterface(dataset) as db_interface:
        target_element_ids = db_interface.get_target_item_ids(target_query)
    
    target_element_ids_str = ','.join(str(value) for value in target_element_ids)

    return "select id, rating, type from elements where id in ({}) UNION select id, rating, type from elements where item_id in ({}) {}".format(target_element_ids_str,target_element_ids_str,element_offset_clause)


def apply_operator_variant(relevance_function):

    operator_variant = configuration.exploration_configurations["operator_variant"]

    relevance_functions = []

    if operator_variant == "TSG":
       relevance_functions = ["sentiment_sim", "tag_sim", "topic_sim"]
    if operator_variant == "TEXT":
        relevance_functions = ["sim", "summary_sim"] 
    if operator_variant == "ATTRIB":
        relevance_functions = ["attribute_sim"]
    else: # operator_variant == "ALL"
        relevance_functions = ["sim", "summary_sim", "sentiment_sim", "tag_sim", "topic_sim", "attribute_sim"]

    if relevance_function in relevance_functions:
        return relevance_function

    relevance_functions_count = len(relevance_functions)
    coin = random.randint(0, relevance_functions_count-1)
    return relevance_functions[coin]
