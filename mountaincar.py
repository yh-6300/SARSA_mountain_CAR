from tqdm import *
import sys
#sys.path.append('/Users/park-yeohyeon/opt/anaconda3/envs/gym/lib/python3.5/site-packages')

import gym
import numpy as np

env = gym.make("MountainCar-v0")

Size = [20, 20]
s_size = (env.observation_space.high - env.observation_space.low)/Size

q_table = np.random.uniform(low=-2, high=0, size=(Size + [env.action_space.n]))


def s_state(state):
    discrete_state = (state - env.observation_space.low)/s_size
    return tuple(discrete_state.astype(np.int))  


done = False

alpha = 0.1
gamma = 0.95
nb_episodes = 200000
epsilon = 0.05
episode = 1

for i in range(nb_episodes):
	score = 0
	step = 0
	done = False
	state = s_state(env.reset())
	if np.random.rand() > epsilon:
		action = np.argmax(q_table[state])
	else:
		action = np.random.randint(2)
	while not done:
		env.render()
		new_state, reward, done, _ = env.step(action)
		new_state = s_state(new_state)
		if np.random.rand() > epsilon:
			new_a = np.argmax(q_table[new_state])
		else:
			new_a = np.random.randint(2)
		if not done:
			max_future_q = q_table[new_state+(new_a,)]
			current_q = q_table[state + (action,)]
			new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
			q_table[state + (action,)] = new_q
			state = new_state
			action = new_a
		elif new_state[0] >= env.goal_position:
			q_table[state + (action,)] = 0
		step += 1
		score +=reward
	print('episode:',episode)
	print('step:',step)
	#state = new_state
	print('Final score:',score)
	if step < 200:
		input()
		break
	episode += 1

input()
env.close()
