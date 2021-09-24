#!/usr/bin/python3

# This library provides functions to learn text-based exploration policies using Deep Reinforcement Learning.

import sys
import configuration

import pfrl  # A library for Deep Reinforcement Learning
from pfrl import experiments, utils
from pfrl.agents import a3c
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers
from pfrl.q_functions import DiscreteActionValueHead

import torch
import torch.nn
from torch import nn
import gym  # the environment enabler
from intex_env.envs.intex_env import intex_env  # intext environment
from rl.deep_q import q_function
from data.db_interface import DBInterface
import numpy
import random
import wandb
import intex_experiments
from pfrl.agents import a2c
from pfrl.experiments import train_agent_with_evaluation, EvaluationHook
import math
import utilities
from utilities import ColorPrint as uc

# set parameters based on input arguments from the command line (if any)
args = [arg[2:] for arg in sys.argv[1:] if arg.startswith("--")]
for arg in args:
    parameter, value = arg.split("=")
    if utilities.parameter_category(parameter) == "learning":
        value_type = type(configuration.learning_configurations[parameter])
        configuration.learning_configurations[parameter] = value_type(value)
    elif utilities.parameter_category(parameter) == "exploration":
        value_type = type(configuration.exploration_configurations[parameter])
        configuration.exploration_configurations[parameter] = value_type(value)
    elif utilities.parameter_category(parameter) == "environment":
        value_type = type(configuration.environment_configurations[parameter])
        configuration.environment_configurations[parameter] = value_type(value)
    else:
        continue

if configuration.learning_configurations["nb_threads"] > 0:
    torch.set_num_threads(configuration.learning_configurations["nb_threads"])

dataset = configuration.learning_configurations["dataset"]
algorithm = configuration.learning_configurations["algorithm"]
target_variant = configuration.learning_configurations["target_variant"]
transfer_variant = configuration.learning_configurations["transfer_variant"]
reward_power = configuration.environment_configurations["reward_power"]
trained_model_name = "dqn-agent"
operator_variant = configuration.exploration_configurations["operator_variant"]
reward_variant = configuration.environment_configurations["reward_variant"]

# Define an instance of the INTEX environment
env: intex_env = gym.make('intex-env-v1')
with DBInterface(dataset) as db_interface:

    target_query = intex_experiments.target_query()
    target_element_ids = db_interface.get_target_ids(target_query)

    # Define the starting point and the target of the environment
    env.initialize(k=configuration.exploration_configurations["k"], target_element_ids=target_element_ids, 
        reward_variant=reward_variant, db_interface=db_interface, reward_power = reward_power,
        input_element_selection_strategy=configuration.learning_configurations["input_element_selection_strategy"])

    # Define an instance of the deep Q function
    observation_size = configuration.environment_configurations["nb_state_features"]
    # Set episode parameters
    nb_episodes = configuration.learning_configurations["nb_episodes"]
    episode_length = configuration.learning_configurations["episode_length"]
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

    uc.print_title("parameters for learning")
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

    eval_env: intex_env = gym.make('intex-env-v1')

    test_target_query = intex_experiments.target_query(test=True)
    test_target_element_ids = db_interface.get_target_ids(test_target_query)

    eval_env.initialize(k=configuration.exploration_configurations["k"], target_element_ids=test_target_element_ids,
                        reward_variant=reward_variant, reward_power= reward_power, db_interface=db_interface,
                        input_element_selection_strategy=configuration.learning_configurations[
                        "input_element_selection_strategy"], eval_mode=True)
    # test_env = pfrl.wrappers.RandomizeAction(test_env, 0.05)

    def log_episode(env, agent, t):
        if t % episode_length == 0:
            episode_log = {
                "reward": env.cumulated_reward,
                "steps": t,
                "step_reward": env.cumulated_reward/t,
                "targets_found": env.get_found_target_count(),
                "epsilon": agent.explorer.epsilon if type(agent) == pfrl.agents.DoubleDQN else 0,
                "distinct_item_seen": len(env.item_ids_seen),
                "distinct_review_seen": len(env.review_ids_seen),
                "discarded_steps": env.discarded_steps
            }

            episode_log.update(env.quality_function_counters)
            episode_log.update(env.relevance_function_counters)
            wandb.log(episode_log)
            targets_found = env.get_found_target_count()
            perc = round(
                float(targets_found / len(target_element_ids)) * 100, 2)
            uc.print_episode(
                t/episode_length, round(env.cumulated_reward, 2), targets_found, perc)

    class EvaluationLog(EvaluationHook):
        support_train_agent = True

        def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
            eval_stats.update(agent_stats)
            eval_stats.pop('length_mean')
            eval_stats.pop('length_median')
            eval_stats.pop('length_stdev')
            eval_stats.pop('length_max')
            eval_stats.pop('length_min')
            # eval_stats.append(eval_env.get_found_target_count())
            wandb.log(eval_stats)
            print(eval_stats)
            print(agent_stats)
            print(env_stats)

    train_agent_with_evaluation(agent, env, steps=episode_length * nb_episodes,
                                outdir=f'./model/{wandb.run.name}', train_max_episode_len=episode_length,
                                step_hooks=[log_episode], eval_n_steps=None,
                                eval_n_episodes=configuration.learning_configurations["test_episode_length"],
                                eval_interval= 25 * episode_length, eval_env=eval_env, 
                                evaluation_hooks=[EvaluationLog()])
