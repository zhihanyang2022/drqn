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

from heaven_hell_simple import HeavenHellSimple

# ==================================================
# hyper-parameters that need tuning

"""
python algorithms/POMDP/3-DRQN-Store-State-HeavenHellSimple/train.py \
--lr=0.00001 \
--use_experts=0 \
--debug_mode=0 \
--device_str=cpu \
--use_deeper_net=1 \
--use_early_stopping=1 \
--seed=1
"""

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate (e.g., 0.001)')
parser.add_argument('--use_experts', type=int, help='whether to use two experts to guide exploration (0 for on; 1 for off)')
parser.add_argument('--seed', type=int, help='seed for np.random.seed and torch.manual_seed (e.g., 42)')
parser.add_argument('--debug_mode', type=int)
parser.add_argument('--device_str', type=str)
parser.add_argument('--use_deeper_net', type=int)
parser.add_argument('--use_early_stopping', type=int)

args = parser.parse_args()
lr = args.lr
use_experts = bool(args.use_experts)
seed = args.seed
debug_mode = bool(args.debug_mode)
device = torch.device(args.device_str)
use_deeper_net = bool(args.use_deeper_net)
use_early_stopping = bool(args.use_early_stopping)

if debug_mode: print('Running debug mode (i.e., without wandb)')

# ==================================================

# ==================================================
# fixed hyper-parameters

gamma = 0.99
sequence_length = 8

max_episodes = int(10 * 1e3)  # 10k episodes; less than or equal to 10k * 20 = 200k steps
epsilon = 1.0  # initial uniform exploration
terminal_epsilon = 0.1
decay_over_episodes = int(3 * 1e3)  # 3k episodes
decay_per_episode = (epsilon - terminal_epsilon) / decay_over_episodes

replay_memory_capacity = max_episodes  # can store 500 episodes
batch_size = 32
update_target = 1000  # once per 1000 steps
log_interval = 10  # one console log per 10 episodes

target_score = 0.99
patience = 3

# ==================================================

# ==================================================
# logging settings

if debug_mode is False:

    group_name = f"lr={lr} use_experts={use_experts} use_deeper_net={use_deeper_net} use_early_stopping={use_early_stopping}"
    run_name = f"seed={seed}"

    print('Group name:')
    print(group_name)
    print('Run name:')
    print(run_name)

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

def one_hot_encode_obs(obs:int):
    one_hot_repr = np.zeros((HeavenHellSimple().observation_space_dim, ))
    one_hot_repr[obs] = 1
    return one_hot_repr

env = HeavenHellSimple()

np.random.seed(seed)
torch.manual_seed(seed)

num_inputs = env.observation_space_dim
num_actions = env.action_space_dim
print('observation size:', num_inputs)
print('action size:', num_actions)

online_net = DRQN(num_inputs, num_actions, sequence_length, use_deeper_net)
target_net = DRQN(num_inputs, num_actions, sequence_length, use_deeper_net)
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

for e in range(max_episodes):

    done = False

    obs = env.reset()
    obs = one_hot_encode_obs(obs)
    obs = torch.Tensor(obs).to(device)

    hidden = (torch.Tensor().new_zeros(1, 1, 16), torch.Tensor().new_zeros(1, 1, 16))

    episode_len = 0  # incremented per action taken
    trajectory = []

    while not done:

        if use_experts is False:  # do the normal thing
            action, new_hidden = get_action(obs, target_net, epsilon, env, hidden)
        else:
            action, new_hidden = get_action(obs, target_net, epsilon, env, hidden, expert_actions=env.get_expert_actions())

        episode_len += 1

        trajectory.append(
            (episode_len, int(torch.argmax(obs)), int(action))
        )

        next_obs, reward, done = env.step(action)
        next_obs = one_hot_encode_obs(next_obs)

        next_obs = torch.Tensor(next_obs)

        mask = 0 if done else 1

        if use_early_stopping is False:
            memory.push(obs, next_obs, action, reward, mask, hidden)
        else:
            if converged is False:
                memory.push(obs, next_obs, action, reward, mask, hidden)
        hidden = new_hidden

        obs = next_obs

        if len(memory) > batch_size and (use_early_stopping is False or converged is False):

            # Result of use_early_stopping is False or converged is False
            # use_early_stopping | converged | results
            # True                 True        False -> avoid updated
            # True                 False       True  -> do update
            # False                True        True  -> do update
            # False                False       True  -> do update

            batch = memory.sample(batch_size)

            loss = DRQN.train_model(online_net, target_net, optimizer, batch, batch_size, sequence_length, gamma, use_deeper_net)

            if steps % update_target == 0:
                update_target_model(online_net, target_net)

        steps += 1

    trajectory.append(
        (episode_len + 1, int(torch.argmax(obs)), None)
    )

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

    if e % log_interval == 0:
        print(f'Iteration {e} / {max_episodes} | Running score {round(running_score, 2)} | Epsilon {round(epsilon, 2)}')
        print(trajectory)