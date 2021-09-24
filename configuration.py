# This library contains all configurable parameters for learning policies and environment initialization.
# All such parameters are organized in the "learning_configurations" and "environment_configurations" dictionaries.


# ***** PARAMETERS FOR LEARNING POLICIES *****

learning_configurations = {}

learning_configurations["nb_episodes"] = 850
learning_configurations["test_nb_episodes"] = 850
learning_configurations["test_episode_length"] = 20
learning_configurations["episode_length"] = 200
learning_configurations["alpha"] = 0.0003				# Learning rate
learning_configurations["gamma"] = 0.9				# Reward discount factor
# Defines how often the learning progress should be reported
learning_configurations["show_every"] = 1
# The exploitation-exploration trade off parameter
learning_configurations["epsilon"] = 0.5
# One of "constant", "linear decay", "exponential_decay"
learning_configurations["epsilon_strategy"] = "constant"
learning_configurations["start_epsilon"] = 1
learning_configurations["end_epsilon"] = 0.01
learning_configurations["epsilon_decay_factor"] = 0.99995
# Strategy to pick the input element. One of : 'random', 'rating' or 'agent_selection'
learning_configurations["input_element_selection_strategy"] = "agent_selection"

start_epsilon_decaying = 1
end_epsilon_decaying = learning_configurations["nb_episodes"]
learning_configurations["end_epsilon_decaying"] = end_epsilon_decaying
learning_configurations["start_epsilon_decaying"] = start_epsilon_decaying
learning_configurations["epsilon_decay_value"] = learning_configurations["epsilon"] / \
    (end_epsilon_decaying-start_epsilon_decaying)

learning_configurations["dataset"] = "imdb"  # "imdb" or "amazon"
learning_configurations["target_variant"] = "T1"  # "T1" or "T2"
learning_configurations["transfer_variant"] = "SIM" # the variant for the transfer learning experiments: "S2G", "G2S", "SIM", "DIFF"

# If true, the target will be limited to "target_size", otherwise it will be unlimited
learning_configurations["target_limit"] = False
learning_configurations["target_size"] = 250

# "DQN" or "A2C" or "DQN Recurrent"
learning_configurations["algorithm"] = "DQN"

# 0 for default, larger than 0 to define the number of PyTorch threads
learning_configurations["nb_threads"] = 0



exploration_configurations = {}
# Number of elements in the exploration output
exploration_configurations["k"] = 5

# Define whether the optimization is minimization or maximization.
# Possible values: "max" (set by default) and "min"
exploration_configurations["optimization_direction"] = "max"

# The maximum number of relevant items to retrieve (for efficiency reasons)
# Based on Stoic settings, 200 is adequate.
exploration_configurations["k_prime"] = 200
# Time limit in milliseconds. Based on Stoic settings, 100ms is adequate.
exploration_configurations["time_limit"] = 100
# Number of optimization improvement loops
exploration_configurations["nb_optimization_loops"] = 150
# "time_limit", "nb_optimization_loops"
exploration_configurations["optimization_meter"] = "nb_optimization_loops"

# Assuming the set {"sim", "summary_sim", "sentiment_sim", "tag_sim", "topic_sim"}, hence 5 relevance functions
exploration_configurations["nb_relevance_functions"] = 5

# Assuming the set {"no_otim", "diverse_numerical", "diverse_review", "coverage_review"}, hence 4 quality functions
exploration_configurations["nb_quality_functions"] = 4

# Maximum number of unique words in reviews
exploration_configurations["max_nb_words"] = 300
exploration_configurations["max_text_size"] = 1000
exploration_configurations["max_tag_count"] = 100

# By default, it remains blank. It can also be TSG, TEXT, ATTRIB, ALL.
exploration_configurations["operator_variant"] = "ALL"

# ***** PARAMETERS FOR ENVIRONMENT INITIALIZATION *****

environment_configurations = {}

# Number of state features
environment_configurations["nb_state_features"] = 20 + \
    (36 * (exploration_configurations["k"]+1))

# For 50% reach, set it to 0.5. For 80% reach, set it to 0.2. For full reach set it to zero (write it 0.0).
environment_configurations["nb_reach"] = 0.0

environment_configurations["show_targets"] = False

environment_configurations["reward_buffer_size"] = 50
environment_configurations["reward_vector_size"] = 5

environment_configurations["reward_for_seen"] = False
environment_configurations["start_random"] = True

environment_configurations["target_reward_active"] = True
environment_configurations["sim_reward_active"] = True

environment_configurations["reward_power"] = 4 # power function which is neutralized if equal to 1
environment_configurations["neutral_reward"] = 0
environment_configurations["reward_variant"] = "SCL" # "SCL"  # "SCL" or "MOO"

