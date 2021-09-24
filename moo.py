# The library for the multi-objective optimization of rewards
import configuration  
import random

buffer_size = configuration.environment_configurations["reward_buffer_size"]
reward_vector_size = configuration.environment_configurations["reward_vector_size"]

def dominated(reward1, reward2):
	# if reward2 is better than reward1, return "1", if reward1 is better than reward2, return "-1", otherwise "0"
	count1 = 0
	count2 = 0
	for i in range(0,reward_vector_size):
		if reward2[i] > reward1[i]:
			count2 += 1
		if reward1[i] > reward2[i]:
			count1 += 1
	if count2 == 4:
		return 1
	if count1 == 4:
		return -1
	return 0

def buffer_eligible(reward, buffer):
	for b in buffer:
		d = dominated(b,reward)
		if d == -1:
			return False
	return True

def buffer_dominance(reward, buffer):
	out = []
	for b in buffer:
		d = dominated(b,reward)
		if d == 1:
			out.append(b)
	return out

# for _ in range(0,100):
# 	reward_vector = []
# 	for _ in range(0,4):
# 		value = round(random.uniform(0, 1), 2)
# 		reward_vector.append(value)
# 	print("New reward is ", reward_vector)
# 	if len(buffer) < buffer_size:
# 		if buffer_eligible(reward_vector, buffer):
# 			dominance = buffer_dominance(reward_vector, buffer)
# 			print("Will dominate ", dominance)
# 			for d in dominance:
# 				buffer.remove(d)
# 			buffer.append(reward_vector)
# 	print("CURRENT BUFFER with size ", len(buffer),":")
# 	for b in buffer:
# 		print(b)
