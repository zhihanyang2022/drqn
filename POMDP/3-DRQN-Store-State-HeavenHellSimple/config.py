import torch

env_name = "HeavenHell-v0"
gamma = 0.99
batch_size = 32  # number of frames to sample from the memory in each training iteration
lr = 0.001
initial_exploration = 1000
log_interval = 10
update_target = 100
replay_memory_capacity = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 8  # more than necessary (6) TODO
burn_in_length = 0  # TODO