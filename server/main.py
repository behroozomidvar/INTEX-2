import json
import aiofiles
import configuration
from os import error
from typing import Dict, List, Optional
from data import data_utilities
from rl import environment
import exploration_actions
import numpy as np
import utilities
from data.db_interface import DBInterface
from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import pandas as pd

from server.custom_json_encoder import CustomJSONEncoder

app = FastAPI(title="INTEX", version="1.0.0",)
app.mount("/intex",
          StaticFiles(directory="client", html=True), name="client")


@app.get("/")
async def read_index():
    return FileResponse('./client/redirect.html')


@app.get("/start-exploration",
         description="",
         tags=["info"])
async def start_exploration(dataset_name: str, session_name: str, keywords:  str):
    keywords = utilities.keywords_to_query(keywords)
    with DBInterface(dataset_name) as db_interface:
        # Create a session (if not existed) or load the session (if existed)
        # Note: a dummy review ID is provided as target for a new session.
        session_history = data_utilities.session_management(
            session_name, db_interface)

        # Given a set of keywords, find one single item relevant to those
        # keywords, as the input element for iteration 0.
        item_for_keywords_id = db_interface.get_item_id_from_keywords(
            keywords)

        # In case no review is returned for the keywords, a random review
        # ... will be returned.
        if item_for_keywords_id == None:
            item_for_keywords_id = db_interface.get_random_element_id()
        input_element = pd.Series()
        output_elements = pd.DataFrame()
        # config = configuration.exploration_configurations

        # start_elements = db_interface.get_episode_start_elements(
        #     config["k"])
        # input_element = start_elements.iloc[0]
        # output_elements = start_elements.iloc[1:]
        # item_for_keywords_id = input_element.id

        result = explore_execution(db_interface, session_name, item_for_keywords_id, relevance_function="sim",
                                   quality_function="diverse_numerical", iteration=session_history.iterations_so_far, terminate_session=False, input_element=input_element, output_elements=output_elements)
        # Get the topic definitions from the database

        topic_definitions = utilities.prettify_topic_definition(
            db_interface.execute_query("topic_definition", db_interface.get_query("topic_definition")))
        result["topic_definitions"] = topic_definitions

        return Response(media_type="application/json", content=json.dumps(result, cls=CustomJSONEncoder))


@app.get("/explore",
         description="",
         tags=["info"])
async def explore(dataset_name: str, session_name: str, input_element_id: int, relevance_function: str,
                  quality_function: str, iteration: int, terminate_session: bool, element_type: str):
    with DBInterface(dataset_name) as db_interface:
        result = explore_execution(db_interface, session_name, input_element_id, relevance_function,
                                   quality_function, iteration, terminate_session)
        return Response(media_type="application/json", content=json.dumps(result, cls=CustomJSONEncoder))


def explore_execution(db_interface: DBInterface, session_name: str, input_element_id: int, relevance_function: str,
                      quality_function: str, iteration: int, terminate_session: bool, input_element=pd.Series(), output_elements=pd.DataFrame()):
    if iteration != 0:
        # Update the session in the database
        next_iteration = iteration + 1
        iteration_update_query = db_interface.get_query(
            "iteration_update", [next_iteration, session_name])
        iteration_update = db_interface.manipulation_query(
            iteration_update_query)
    else:
        next_iteration = 1

    # # Compute reward for the input element
    # input_element_reward = data_utilities.RewardManager(
    #     input_element_id, None, db_interface, session_name)

    if input_element.empty:
        input_element = db_interface.get_element(
            input_element_id)

    if len(output_elements) == 0:
        # Obtain output elements
        exploration_engine = exploration_actions.exploration(db_interface)
        exploration_engine.config["relevance_function"] = relevance_function
        exploration_engine.config["quality_function"] = quality_function
        output_elements = exploration_engine.explore(
            input_element)

    if iteration != 0:
        # Insert the information about the new iteration into the database
        exploration_iteration_register_query = db_interface.get_query("iteration_register", [
            session_name, next_iteration, input_element_id, output_elements.id.dropna().to_list(), terminate_session])
        exploration_iteration_registered = db_interface.manipulation_query(
            exploration_iteration_register_query)

        # Send previously used relevance function and quality function to the next iteration (for explainability purposes.)
        previous_output_elements_makers = [
            utilities.relevance_function_explain(relevance_function), quality_function]

        # Define the state the agent is currently at
        env = environment.Environment(db_interface)
        exploration_action = env.mystify_action(
            relevance_function, quality_function)
        state = env.get_bucketized_state(
            output_elements, input_element, exploration_action)
        try:
            # Given the current state, pick the best action recommended by the learned policy
            policy = np.load('model/policy.npy')
            recommended_exploration_action = np.argmax(policy[state])
            recommended_quality_function, recommended_relevance_function = env.demystify_action(
                recommended_exploration_action)
            recommendations = [recommended_quality_function,
                               recommended_relevance_function]
        except IndexError:
            recommendations = ["none", "sim"]
    else:
        previous_output_elements_makers = ["sim", "diverse_numerical"]
        recommendations = ["none", "sim"]

    # Determine the next HTML page to head to
    # In case the exploration session is terminated, we head back to the initial page.
    # if terminate_session == "1":
    #     self.redirect('static/index.html')

    # In case the exploration continues, we proceed to the next iteration.
    related_item_ids = output_elements.item_id.dropna().to_list()
    if input_element.item_id != None and not np.isnan(input_element.item_id):
        related_item_ids.append(input_element.item_id)

    input_element = json.loads(input_element.to_json())
    output_elements = json.loads(output_elements.to_json(orient='records'))

    if len(related_item_ids) != 0:
        related_items = db_interface.get_related_items(related_item_ids)
        if "item_id" in input_element and input_element["item_id"] != None:
            input_element["item"] = related_items[related_items.id ==
                                                  input_element["item_id"]].iloc[0].to_dict()

        for element in output_elements:
            if "item_id" in element and element["item_id"] != None and not np.isnan(element["item_id"]):
                element["item"] = related_items[related_items.id ==
                                                element["item_id"]].iloc[0].to_dict()
            element["reward"] = data_utilities.RewardManager(
                [element["id"]], None, db_interface, session_name).rewards[0].to_dict()

    return {
        "iteration": next_iteration,
        "input_element": input_element,
        "output_elements": output_elements,
        "previous_output_elements_makers": previous_output_elements_makers,
        "recommendations": recommendations
    }
