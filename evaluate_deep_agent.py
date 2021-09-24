# This library provides functions to evaluate the learned text-based exploration policies.
# There is no argument passing in the evaluation and all parameters are assumed to be set by their default.

from intex_env.envs.intex_env import intex_env       # INTEX environment
from data.db_interface import DBInterface
import intex_experiments
import pfrl             # A library for Deep Reinforcement Learning
import gym              # The environment enabler
import configuration
from rl.deep_q import q_function
import torch
import numpy
import wandb
import math
from utilities import ColorPrint as uc
import pandas as pd
import numpy as np
import sys

dataset = configuration.learning_configurations["dataset"]
algorithm = configuration.learning_configurations["algorithm"]
target_variant = configuration.learning_configurations["target_variant"]
transfer_variant = configuration.learning_configurations["transfer_variant"]
reward_power = configuration.environment_configurations["reward_power"]
operator_variant = configuration.exploration_configurations["operator_variant"]
reward_variant = configuration.environment_configurations["reward_variant"]

# set parameters based on input arguments from the command line (if any)
args = [arg[2:] for arg in sys.argv[1:] if arg.startswith("--")]
for arg in args:
    parameter, value = arg.split("=")
    value = value.strip()
    if parameter == "transfer_variant":
        transfer_variant = value
    elif parameter == "dataset":
        dataset == value
    elif parameter == "target_variant":
        target_variant = value
    else:
        continue

torch.set_num_threads(5)

# Define an instance of the INTEX environment
env: intex_env = gym.make('intex-env-v1')
with DBInterface(dataset) as db_interface:

    target_query = intex_experiments.target_query(test=True)
    target_element_ids = db_interface.get_target_ids(target_query)

    # Define the starting point and the target of the environment
    env.initialize(k=configuration.exploration_configurations["k"], target_element_ids=target_element_ids, 
        reward_variant=reward_variant, db_interface=db_interface, reward_power = reward_power,
        input_element_selection_strategy=configuration.learning_configurations["input_element_selection_strategy"])

    # Define an instance of the deep Q function
    observation_size = configuration.environment_configurations["nb_state_features"]
    # Set episode parameters
    nb_episodes = configuration.learning_configurations["test_nb_episodes"]
    # episode_length = configuration.learning_configurations["episode_length"]
    episode_length = 50
    show_every = configuration.learning_configurations["show_every"]

    epsilon_strategy = configuration.learning_configurations["epsilon_strategy"]
    if epsilon_strategy == "constant":
        # Set epsilon-greedy as the explorer function
        epsilon = configuration.learning_configurations["epsilon"]
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=epsilon, random_action_func=env.choose_random_action)
    elif epsilon_strategy == "linear decay":
        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
            configuration.learning_configurations["start_epsilon"],
            configuration.learning_configurations["end_epsilon"],
            nb_episodes * episode_length,
            random_action_func=env.choose_random_action
        )
    else:
        explorer = pfrl.explorers.ExponentialDecayEpsilonGreedy(
            configuration.learning_configurations["start_epsilon"],
            configuration.learning_configurations["end_epsilon"],
            configuration.learning_configurations["epsilon_decay_factor"],
            random_action_func=env.choose_random_action
        )
    # Set the discount factor for future rewards.
    gamma = configuration.learning_configurations["gamma"]

    # Now create an agent that will interact with the environment.
    agent = None
    # As PyTorch only accepts numpy.float32 by default, specify ...
    # a converter as a feature extractor function phi.
    def phi(x): return x.astype(numpy.float32, copy=False)
    if algorithm == "DQN":

        q_function = q_function(observation_size, env.get_action_space_size())
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 5)
        # Use Adam optimizer to optimize the Q function. We set eps=1e-2 for stability.
        optimizer = torch.optim.Adam(q_function.parameters(
        ), lr=configuration.learning_configurations["alpha"], eps=1e-2)
        agent = pfrl.agents.DoubleDQN(q_function, optimizer, replay_buffer, gamma, explorer,
                                      replay_start_size=50, update_interval=1, target_update_interval=100, phi=phi, gpu=-1, recurrent=False)
        network_width = q_function.network_width
    elif algorithm == "DQN Recurrent":
        network_width = 1024
        q_func = pfrl.nn.RecurrentSequential(
            nn.Linear(observation_size, network_width),
            nn.ReLU(),
            torch.nn.Linear(network_width, network_width),
            nn.ReLU(),
            torch.nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.LSTM(input_size=network_width, hidden_size=network_width),
            nn.Linear(network_width, env.get_action_space_size()),
            DiscreteActionValueHead(),
        )

        replay_buffer = pfrl.replay_buffers.EpisodicReplayBuffer(
            capacity=10 ** 5)
        # Use Adam optimizer to optimize the Q function. We set eps=1e-2 for stability.
        optimizer = torch.optim.Adam(q_func.parameters(
        ), lr=configuration.learning_configurations["alpha"], eps=1e-2)
        agent = pfrl.agents.DoubleDQN(q_func,
                                      optimizer,
                                      replay_buffer,
                                      gamma,
                                      explorer,
                                      replay_start_size=50,
                                      update_interval=4,
                                      target_update_interval=100,
                                      phi=phi,
                                      gpu=-1,
                                      recurrent=True,
                                      batch_accumulator="mean",
                                      episodic_update_len=10)
    elif algorithm == "A2C":
        network_width = 1024
        model = nn.Sequential(
            torch.nn.Linear(observation_size, network_width), nn.ReLU(),
            torch.nn.Linear(network_width, network_width), nn.ReLU(),
            nn.Linear(network_width, network_width), nn.ReLU(),
            pfrl.nn.Branched(
                nn.Sequential(nn.Linear(network_width, env.get_action_space_size()),
                              SoftmaxCategoricalHead(),),
                nn.Linear(network_width, 1)
            )
        )
        optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
            model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)
        agent = a2c.A2C(model, optimizer, gamma,
                        num_processes=1, phi=phi, max_grad_norm=None)

    # WandB init and config
    wandb.init(project='ddqn', entity='intex')
    config = wandb.config
    config.dataset = db_interface.dataset
    config.algorithm = algorithm
    config.alpha = configuration.learning_configurations["alpha"]
    config.gamma = agent.gamma
    config.strategy = env.input_element_selection_strategy
    config.epsilon_strategy = epsilon_strategy
    config.network_width = network_width
    config.episode_length = episode_length
    config.nb_episodes = nb_episodes
    config.k = env.output_element_count
    if "DQN" in algorithm:
        config.replay_start_size = agent.replay_start_size
    if "DQN" in algorithm:
        config.update_interval = agent.update_interval
    if "DQN" in algorithm:
        config.target_update_interval = agent.target_update_interval
    config.target_query = target_query
    config.target_count = len(target_element_ids)

    uc.print_title("parameters for testing")
    uc.print_param("dataset", dataset)
    uc.print_param("target variant", target_variant)
    uc.print_param("transfer variant", transfer_variant)
    uc.print_param("algorithm", algorithm)
    uc.print_param("operator variant", operator_variant)
    uc.print_param("reward variant", reward_variant)
    uc.print_param("number of output elements",
                   configuration.exploration_configurations["k"])
    uc.print_param("episode length", episode_length)
    uc.print_param("nb targets to seek", str(len(target_element_ids)))

    uc.print_title("simulating "+str(nb_episodes)+" exploration sessions ...")

    feature_activation_columns = ["none", "diverse_numerical", "diverse_review", "coverage_review", "sim", "summary_sim", "sentiment_sim", "tag_sim", "topic_sim", "attribute_sim"]
    feature_activation = pd.DataFrame(0, index=np.arange(observation_size), columns = feature_activation_columns)

    # load the trained model based on the testing variant
    agent_model_address = "model/{}_t{}/170000_finish".format(dataset,target_variant[1])
    agent.load(agent_model_address)
    # agent.load("dqn-agent")

    for episode in range(1, nb_episodes + 1):

        uc.print_param("episode",episode)
        # Receive the first observation by resetting the environment
        observation = env.reset()

        # return is the sum of all reward
        Return = 0

        # Current time step in the episode
        time_step = 0
        reducer = 0

        # Action counters for logs
        quality_function_counters = {
            "none": 0, "diverse_numerical": 0, "diverse_review": 0, "coverage_review": 0}
        relevance_function_counters = {
            "sim": 0, "summary_sim": 0, "sentiment_sim": 0, "tag_sim": 0, "topic_sim": 0, "attribute_sim": 0}
        done = reset = False
        # The episode loop
        while not done and not reset:

            # Choose an action based on the initial observation
            action = agent.act(observation)

            # Update action counters for logs
            quality_function, relevance_function, input_element_index = env.demystify_action(
                action)
            quality_function_counters[quality_function] += 1
            relevance_function_counters[relevance_function] += 1

            for i in range(observation_size):
                if observation[i] == 1:
                    feature_activation.loc[i, quality_function] += 1
                    feature_activation.loc[i, relevance_function] += 1

            # Apply the action and receive the next state (observation) and the reward
            observation, reward, done, _ = env.step(action)
            reward = math.pow(reward, reward_power)
            Return += reward
            time_step += 1

            reset = True if time_step == episode_length else False

            # Update the agent
            # if reward > 0:
            #     agent.observe(observation, reward, done, reset)
            # else:
            #     reducer += 1


        episode_log = {
            "reward": Return,
            "steps": time_step,
            "step_reward": Return/time_step,
            "targets_found": env.get_found_target_count(),
            "epsilon": agent.explorer.epsilon if type(agent) == pfrl.agents.DoubleDQN else 0,
            "distinct_item_seen": len(env.item_ids_seen),
            "distinct_review_seen": len(env.review_ids_seen)
        }

        episode_log.update(quality_function_counters)
        episode_log.update(relevance_function_counters)
        wandb.log(episode_log)

    uc.print_title("testing finished.")
    feature_activation.to_csv("feature_activation.csv")
