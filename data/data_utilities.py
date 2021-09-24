# This library consists of several functions for interacting with the data at online time.

from data.db_interface import DBInterface
import statistics
import json
import configuration

# Given a new input element, this class encapsulates the computations
# ... of the reward for this new element. The reward is assumed to be
# ... a single value between 0 and 1.


class Reward:
    def __init__(self, is_target=False, similarity=None, neutral=False) -> None:

        if (is_target and not(configuration.environment_configurations["target_reward_active"])) or (not(is_target) and not(configuration.environment_configurations["sim_reward_active"])):
            neutral = True

        if neutral:
            neutral_reward = configuration.environment_configurations["neutral_reward"]
            self.text_reward = neutral_reward
            self.summary_reward = neutral_reward
            self.topic_reward = neutral_reward
            self.sentiment_reward = neutral_reward
            self.tag_reward = neutral_reward
            self.mean_reward = neutral_reward
            self.max_reward = neutral_reward
        elif is_target and configuration.environment_configurations["target_reward_active"]:
            # self.text_reward = 2
            # self.summary_reward = 2
            # self.topic_reward = 2
            # self.sentiment_reward = 2
            # self.tag_reward = 2
            # self.mean_reward = 2
            # self.max_reward = 2
            self.text_reward = 1
            self.summary_reward = 1
            self.topic_reward = 1
            self.sentiment_reward = 1
            self.tag_reward = 1
            self.mean_reward = 1
            self.max_reward = 1
        else:
            self.text_reward = round(similarity.sim / 100.0, 2)
            self.summary_reward = round(similarity.summary_sim / 100.0, 2)
            self.topic_reward = round(similarity.topic_sim, 2)
            self.sentiment_reward = round(similarity.sentiment_sim, 2)
            self.tag_reward = round(similarity.tag_sim / 100.0, 2)
            self.max_reward = max(self.text_reward, self.summary_reward,
                                  self.topic_reward, self.tag_reward, self.sentiment_reward)
            self.mean_reward = round(statistics.mean(
                [self.text_reward, self.summary_reward, self.topic_reward, self.tag_reward, self.sentiment_reward]), 2)

    def to_dict(self):
        return self.__dict__


class RewardManager():

    # The target of the class is either explicitly given (using the
    # ... variable "target", or the "session_name" is given whose
    # ... associated target can be retrieved from the database.
    def __init__(self, output_elements, targets, db_interface: DBInterface, session_name=""):

        # The new input element under investigation
        # self.input_element = str(input_element)

        self.output_elements = output_elements

        # The name of the current active session
        self.session_name = session_name

        # The number of iterations passed so far in the current active session
        self.iterations_so_far = 1

        # The target review (indicated with a review ID)
        if targets != None:
            self.targets = targets
        else:
            # with open('model/training_data.json') as json_file:
            #     training_data = json.load(json_file)
            # self.target = training_data["target_item_id"]
            self.targets = [db_interface.get_random_element_id()]

        self.rewards = []
        # if input_element in self.targets:
        # self.rewards.append(Reward(is_target=True))
        # self.targets.remove(input_element)
        for output_element in self.output_elements:
            if output_element in self.targets:
                self.rewards.append(Reward(is_target=True))
                self.targets.remove(output_element)

        if len(self.targets) != 0:
            # similarities = db_interface.get_element_to_element_list_similarities(input_element, self.targets)
            similarities = db_interface.get_element_to_element_list_similarities(
                output_elements, self.targets)

            if len(similarities) == 0:
                self.rewards.append(Reward(neutral=True))
            else:
                for index, similarity in similarities.iterrows():
                    self.rewards.append(Reward(similarity=similarity))

        sum_rewards = 0

        self.text_reward = max([x.text_reward for x in self.rewards])
        # self.text_reward = sum([x.text_reward for x in self.rewards])
        sum_rewards += self.text_reward

        self.summary_reward = max([x.summary_reward for x in self.rewards])
        # self.summary_reward = sum([x.summary_reward for x in self.rewards])
        sum_rewards += self.summary_reward

        self.topic_reward = max([x.topic_reward for x in self.rewards])
        # self.topic_reward = sum([x.topic_reward for x in self.rewards])
        sum_rewards += self.topic_reward

        self.sentiment_reward = max([x.sentiment_reward for x in self.rewards])
        # self.sentiment_reward = sum([x.sentiment_reward for x in self.rewards])
        sum_rewards += self.sentiment_reward

        self.tag_reward = max([x.tag_reward for x in self.rewards])
        # self.tag_reward = sum([x.tag_reward for x in self.rewards])
        sum_rewards += self.tag_reward

        self.max_reward = max([x.max_reward for x in self.rewards])
        sum_rewards += self.max_reward

        # self.mean_reward = statistics.mean([x.mean_reward for x in self.rewards])
        self.mean_reward = sum_rewards / 6.0  # 5 is the number of reward signals

# At online time in the GUI, this class manages the creation of a new
# ... exploration session and loading a previously created session.


class session_management():

    def __init__(self, session_name, db_interface: DBInterface, target_item_id=None):
        self.db_interface = db_interface
        self.iterations_so_far = 1

        # Check if the session "session_name" already exists in the database
        session_existed = db_interface.execute_query(
            "session_existed", db_interface.get_query(
                "session_existed", [session_name]))

        # Exploration session "session_name" already exists, hence it will be loaded.
        if session_existed == True:
            self.iterations_so_far = session_existed[0]

        # Exploration session "session_name" does not exist, hence one will be created.
        else:
            if target_item_id == None:
                target_item_id = db_interface.get_random_element_id()
            session_create = db_interface.manipulation_query(
                db_interface.get_query(
                    "session_create", [session_name, target_item_id]))
