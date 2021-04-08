import gym
import os
from gym.envs.registration import register
import yaml
import numpy as np

with open('domains_conf/heavenhell.yaml') as file:
    env_conf = yaml.load(file, Loader=yaml.FullLoader)
    print(env_conf)

register(
    id=env_conf['name'],
    entry_point=env_conf['entry_point'],
    kwargs=env_conf['config'],
    max_episode_steps=env_conf['max_episode_steps']
)

env = gym.make(env_conf['name'])

print('Action space:', env.action_space)

obs_to_left_right_counts = {}
from collections import namedtuple
Indexes = namedtuple('Actions', 'left right')
indices = Indexes(0, 1)

counts = 0

for i in range(1000):

    env.reset()
    done = False
    tau = []

    while not done:
        action = env.action_space.sample()
        o, r, done, info = env.step(action)
        o = int(o[0])
        if o not in obs_to_left_right_counts:
            obs_to_left_right_counts[o] = [0, 0]
        tau.append(o)

    if r == 0:
        pass
    elif r == -1:  # hell
        counts += 1
        if action == 2:  # right but punished -> heaven on the left
            for o in tau:
                obs_to_left_right_counts[o][indices.left] += 1
        elif action == 3:  # left but punished -> heaven on the right
            for o in tau:
                obs_to_left_right_counts[o][indices.right] += 1
    elif r == 1:  # heaven
        counts += 1
        if action == 2:  # right and rewarded -> heaven on the right
            for o in tau:
                obs_to_left_right_counts[o][indices.right] += 1
        elif action == 3:  # left and rewarded -> heaven on the left
            for o in tau:
                obs_to_left_right_counts[o][indices.left] += 1

obs_to_relative_diff = {
    obs : np.abs(left_right_counts[indices.left] - left_right_counts[indices.right]) / np.max(left_right_counts) for obs, left_right_counts in obs_to_left_right_counts.items()
}

print(counts)
print(obs_to_relative_diff)
print(obs_to_left_right_counts)