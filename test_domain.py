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
print(len(env.reset()))
print('=== State ===')
print(env.get_state())
print('=== Belief ===')
print(env.get_belief())
print(len(env.get_belief()))
# env.render()