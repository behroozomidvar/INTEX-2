import configuration
from utilities import ColorPrint as cp

cp.print_title("learning configurations")

for config in configuration.learning_configurations:
	cp.print_param(config, configuration.learning_configurations[config])

cp.print_title("exploration configurations")

for config in configuration.exploration_configurations:
	cp.print_param(config, configuration.exploration_configurations[config])

cp.print_title("environment configurations")

for config in configuration.environment_configurations:
	cp.print_param(config, configuration.environment_configurations[config])