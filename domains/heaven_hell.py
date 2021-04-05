import gym
import gym_pomdps
from gym import spaces
import numpy as np
from scipy.stats import entropy

from collections import namedtuple

compass_to_action = {
	'N' : 'up',
	'S' : 'down',
	'E' : 'right',
	'W' : 'left'
}
Actions = namedtuple("Actions", f"{compass_to_action['N']} {compass_to_action['S']} {compass_to_action['E']} {compass_to_action['W']}")
actions = Actions(0, 1, 2, 3)

class HeavenHellEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.env = gym.make("POMDP-heavenhell-episodic-v0")
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(low=0, high=11, shape=(1,), dtype=np.int32)
		self.observation_space_dim = 11
		self.state_space = self.env.state_space
		self.visited_priest = False

	def close(self):
		pass

	def seed(self, seed):
		self.env.seed(seed)

	def get_state(self):
		return self.state

	def get_belief(self):
		return self.belief

	def get_entropy(self):
		return self.entropy

	def reset(self):
		self.traj = []
		self.count = 0
		self.state = self.env.reset_functional()
		self.belief = gym_pomdps.belief.belief_init(self.env)
		self.entropy = entropy(self.belief)
		self.visited_priest = False

		initial_action = np.random.randint(0, self.action_space.n)
		initial_obs, _, _, _ = self.step(initial_action)
		return initial_obs

	def step(self, action):
		if isinstance(action, np.ndarray):
			action = action[0]

		self.traj.append([self.state, action])

		# Next state
		# r = gym_pomdps.belief.expected_reward(self.env, self.belief, action)
		self.state, o, r, done, info = self.env.step_functional(self.state, action)

		if self.state in [9, 19]:
			self.visited_priest = True

		self.belief = gym_pomdps.belief.belief_step(self.env, self.belief, action, o)

		if r in [-1, 1]:
			done = True

		return np.array([o]), r, done, info

	def get_fully_observable_expert_actions(self):

		# heaven on the left - top corridor

		if   self.state == 4:
			raise NotImplementedError  # at heaven
		elif self.state == 3:
			return actions.left
		elif self.state == 2:
			return actions.left
		elif self.state == 5:
			return actions.left
		elif self.state == 6:
			raise NotImplementedError  # at hell

		# heaven on the right - top corridor

		if   self.state == 14:
			raise NotImplementedError  # at hell;
		elif self.state == 13:
			return actions.right
		elif self.state == 12:
			return actions.right
		elif self.state == 15:
			return actions.right
		elif self.state == 16:
			raise NotImplementedError  # at heaven

		# shared

		if   self.state in [1, 11]:
			return actions.up
		elif self.state in [0, 10]:
			return actions.up
		elif self.state in [7, 17]:
			return actions.up
		elif self.state in [8, 18]:
			return actions.left
		elif self.state in [9, 19]:
			return actions.left

	def get_entropy_reduction_expert_actions(self):

		if self.visited_priest is False:
			if   self.state in [4, 14]:
				return actions.right
			elif self.state in [3, 13]:
				return actions.right
			elif self.state in [2, 12]:
				return actions.down
			elif self.state in [5, 15]:
				return actions.left
			elif self.state in [6, 16]:
				return actions.left
			elif self.state in [1, 11]:
				return actions.down
			elif self.state in [0, 10]:
				return actions.down
			elif self.state in [7, 17]:
				return actions.right
			elif self.state in [8, 18]:
				return actions.right
			elif self.state in [9, 19]:
				raise NotImplementedError
		else:
			return self.action_space.sample()

	def get_expert_actions(self):
		return self.get_fully_observable_expert_actions() + self.get_entropy_reduction_expert_actions()