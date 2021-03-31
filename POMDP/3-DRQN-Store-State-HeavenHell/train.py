import os
import sys
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DRQN
from memory import Memory
from tensorboardX import SummaryWriter

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, sequence_length

from collections import deque

def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)
    
    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())

def one_hot_encode_obs(obs:np.array):
	index = int(obs[0])
	one_hot_repr = np.zeros((11, ))
	one_hot_repr[index] = 1
	return one_hot_repr

def main():

    import gym
    from gym.envs.registration import register
    import yaml

    import sys
    sys.path.append(os.getcwd())

    with open('domains_conf/heavenhell.yaml') as file:
        env_conf = yaml.load(file, Loader=yaml.FullLoader)

    register(
        id=env_conf['name'],
        entry_point=env_conf['entry_point'],
        kwargs=env_conf['config'],
        max_episode_steps=env_conf['max_episode_steps']
    )

    env = gym.make(env_conf['name'])

    env.seed(500)
    torch.manual_seed(500)

    num_inputs = 11
    num_actions = env.action_space.n
    print('observation size:', num_inputs)
    print('action size:', num_actions)

    online_net = DRQN(num_inputs, num_actions)
    target_net = DRQN(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0

    for e in range(1000):
        done = False

        score = 0
        obs = env.reset()
        obs = one_hot_encode_obs(obs)
        obs = torch.Tensor(obs).to(device)

        hidden = (torch.Tensor().new_zeros(1, 1, 16), torch.Tensor().new_zeros(1, 1, 16))

        while not done:
            steps += 1

            action, new_hidden = get_action(obs, target_net, epsilon, env, hidden)
            next_obs, reward, done, _ = env.step(action)
            next_obs = one_hot_encode_obs(next_obs)

            next_obs = torch.Tensor(next_obs)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            memory.push(obs, next_obs, action, reward, mask, hidden)
            hidden = new_hidden

            score += reward
            obs = next_obs
            
            if steps > initial_exploration and len(memory) > batch_size:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = DRQN.train_model(online_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        score = score if score == 500.0 else score + 1
        if running_score == 0:
            running_score = score
        else:
            running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break

if __name__=="__main__":
    main()
