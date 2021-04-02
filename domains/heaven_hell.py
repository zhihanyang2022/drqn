import gym
import gym_pomdps
from gym import spaces
import numpy as np
from scipy.stats import entropy

class HeavenHellEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.env = gym.make("POMDP-heavenhell-episodic-v0")
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(low=0, high=11, shape=(1,), dtype=np.int32)
		self.observation_space_dim = 11
		self.state_space = self.env.state_space

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

		self.belief = gym_pomdps.belief.belief_step(self.env, self.belief, action, o)

		if r in [-1, 1]:
			done = True

		return np.array([o]), r, done, info
