import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 200
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi_Step
n_step = 3

# PER
small_epsilon = 0.0001
alpha = 0.5
beta_start = 0.1

# Noisy Net
sigma_zero = 0.5
