import gym
import os
from gym.envs.registration import register
import yaml

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

initial_obs_list = []
for i in range(100):
    initial_obs_list.append(env.reset()[0])

for index in range(10):
    if index in initial_obs_list:
        print(index)