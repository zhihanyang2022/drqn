import os
import sys
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DRQN
from memory import Memory
# from tensorboardX import SummaryWriter

import argparse
import wandb

import time

# ==================================================
# for debugging

use_wandb = True

# ==================================================

# ==================================================
# hyper-parameters that need tuning

"""
test run code:

python algorithms/POMDP/3-DRQN-Store-State-HeavenHell/train.py \
--lr=0.00001 \
--use_experts=0 \
--debug_mode=0 \
--device_str=cuda \
--use_deeper_net=1 \
--use_early_stopping=0 \
--use_reward_shaping=1 \
--seed=1

first group (seed 1 2 3):

python algorithms/POMDP/3-DRQN-Store-State-HeavenHell/train.py \
--lr=0.00001 \
--use_experts=0 \
--debug_mode=0 \
--device_str=cuda \
--use_deeper_net=1 \
--use_early_stopping=0 \
--use_reward_shaping=0 \
--seed=1

second group (seed 1 2 3):

python algorithms/POMDP/3-DRQN-Store-State-HeavenHell/train.py \
--lr=0.00001 \
--use_experts=1 \
--debug_mode=0 \
--device_str=cuda \
--use_deeper_net=1 \
--use_early_stopping=0 \
--use_reward_shaping=0 \
--seed=1

"""

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate (e.g., 0.001)')
parser.add_argument('--use_experts', type=int, help='whether to use two experts to guide exploration (0 for on; 1 for off)')
parser.add_argument('--seed', type=int, help='seed for np.random.seed and torch.manual_seed (e.g., 42)')
parser.add_argument('--debug_mode', type=int, default=0)
parser.add_argument('--device_str', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--use_deeper_net', type=int)
parser.add_argument('--use_early_stopping', type=int)
parser.add_argument('--use_reward_shaping', type=int)

args = parser.parse_args()
lr = args.lr
use_experts = bool(args.use_experts)
seed = args.seed
debug_mode = bool(args.debug_mode)
device = torch.device(args.device_str)
use_deeper_net = bool(args.use_deeper_net)
use_early_stopping = bool(args.use_early_stopping)
use_reward_shaping = bool(args.use_reward_shaping)

if debug_mode: print('Running debug mode (i.e., without wandb)')

# ==================================================

# ==================================================
# fixed hyper-parameters

env_name = "HeavenHell"

gamma = 0.99
sequence_length = 20

max_episodes = int(100 * 1e3)  # 100k episodes; less than or equal to 100k * 20 = 2000k or 2M steps
epsilon = 1.0  # initial uniform exploration
terminal_epsilon = 0.1
decay_over_episodes = int(3 * 1e3)  # 30k episodes
decay_per_episode = (epsilon - terminal_epsilon) / decay_over_episodes

replay_memory_capacity = int(10 * 1e3)  # 20k episodes
batch_size = 32
update_target = 1000  # once per 1000 steps
log_interval = 10  # one console log per 10 episodes

target_score = 0.99
patience = 3

# ==================================================

# ==================================================
# logging settings

if debug_mode is False:

    group_name = f"env_name={env_name} lr={lr} use_experts={use_experts} use_deeper_net={use_deeper_net} use_early_stopping={use_early_stopping} use_reward_shaping={use_reward_shaping}"
    run_name = f"seed={seed}"

    wandb.init(
        project="drqn",
        entity='pomdpr',
        group=group_name,
        settings=wandb.Settings(_disable_stats=True),
        name=run_name
    )

# ==================================================

def get_action(obs, target_net, epsilon, env, hidden, expert_actions=None):
    action, hidden = target_net.get_action(obs, hidden)
    
    if np.random.rand() <= epsilon:
        if expert_actions is None:
            return env.action_space.sample(), hidden
        else:
            return np.random.choice(expert_actions), hidden
    else:
        return action, hidden

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())

import gym
from gym.envs.registration import register
import yaml

import sys
sys.path.append(os.getcwd())  # current working directory should be drqn/

with open('domains_conf/heavenhell.yaml') as file:
    env_conf = yaml.load(file, Loader=yaml.FullLoader)

register(
    id=env_conf['name'],
    entry_point=env_conf['entry_point'],
    kwargs=env_conf['config'],
    max_episode_steps=env_conf['max_episode_steps']
)

env = gym.make(env_conf['name'])

def one_hot_encode_obs(obs:int):
    one_hot_repr = np.zeros((env.observation_space_dim, ))
    one_hot_repr[obs] = 1
    return one_hot_repr

np.random.seed(seed)
torch.manual_seed(seed)

num_inputs = env.observation_space_dim
num_actions = env.action_space.n
print('observation size:', num_inputs)
print('action size:', num_actions)

online_net = DRQN(num_inputs, num_actions, use_deeper_net)
target_net = DRQN(num_inputs, num_actions, use_deeper_net)
update_target_model(online_net, target_net)

optimizer = optim.Adam(online_net.parameters(), lr=lr)
# if use_experts is False:
#     writer = SummaryWriter('logs/normal')
# else:
#     writer = SummaryWriter('logs/experts')

online_net.to(device)
target_net.to(device)
online_net.train()
target_net.train()
memory = Memory(replay_memory_capacity, sequence_length)

steps = 0  # number of actions taken in the environment / number of parameter updates
loss = 0
running_score = 0
converged = False
patience_used = 0

start_time = time.perf_counter()

for e in range(max_episodes):

    done = False

    obs = env.reset()
    obs = one_hot_encode_obs(obs)
    obs = torch.Tensor(obs).to(device)

    hidden = (torch.Tensor().new_zeros(1, 1, 16).to(device), torch.Tensor().new_zeros(1, 1, 16).to(device))

    episode_len = 0  # incremented per action taken
    met_priest = False

    while not done:

        if use_experts is False:  # do the normal thing
            action, new_hidden = get_action(obs, target_net, epsilon, env, hidden)
        else:
            action, new_hidden = get_action(obs, target_net, epsilon, env, hidden, expert_actions=env.get_expert_actions())

        episode_len += 1

        next_obs, reward, done, _ = env.step(action)

        if use_reward_shaping:
            if next_obs == int(torch.argmax(obs)):  # the agent just took an action into the wall
                total_reward = reward - 0.1
            if (met_priest is False) and (next_obs == 9 or next_obs == 10):  # the agent visits the priest for the first time
                total_reward = reward + 1
                met_priest = True
        else:
            total_reward = reward

        next_obs = one_hot_encode_obs(next_obs)

        next_obs = torch.Tensor(next_obs).to(device)

        mask = 0 if done else 1

        if use_early_stopping is False:
            memory.push(obs, next_obs, action, total_reward, mask, hidden)
        else:
            if converged is False:
                memory.push(obs, next_obs, action, total_reward, mask, hidden)
        hidden = new_hidden

        obs = next_obs

        if len(memory) > batch_size and (use_early_stopping is False or converged is False) and (epsilon < 0.2):

            # Result of use_early_stopping is False or converged is False
            # use_early_stopping | converged | results
            # True                 True        False -> avoid updated
            # True                 False       True  -> do update
            # False                True        True  -> do update
            # False                False       True  -> do update

            batch = memory.sample(batch_size)
            loss = DRQN.train_model(online_net, target_net, optimizer, batch, batch_size, gamma, use_deeper_net, device)

            if steps % update_target == 0:
                update_target_model(online_net, target_net)

        steps += 1

    if epsilon > terminal_epsilon:
        epsilon -= decay_per_episode

    # wandb logging

    if debug_mode is False:
        wandb.log({
            "return": reward,
            "episode_len": episode_len
        })

    # console logging

    running_score = 0.95 * running_score + 0.05 * reward
    if running_score >= target_score:
        patience_used += 1
        if patience_used >= patience:
            converged = True
    else:
        patience_used = 0

    if e % log_interval == 0 and e != 0:  # prevent division by zero
        current_time = time.perf_counter()
        print('==========')
        print(f'Iteration {e} / {max_episodes} | Running score {round(running_score, 2)} | Epsilon {round(epsilon, 2)}')
        average_duration_per_episode = (current_time - start_time) / e
        remaining_duration = average_duration_per_episode * (max_episodes - e)
        print(f'Time remaining: {round((remaining_duration) / 60 / 60, 2)} hours')
        print('==========')