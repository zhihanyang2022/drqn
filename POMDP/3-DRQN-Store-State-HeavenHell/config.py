import torch

env_name = "HeavenHell-v0"
gamma = 0.99
batch_size = 32  # number of frames to sample from the memory in each training iteration
lr = 0.001
initial_exploration = 1000  # TODO
goal_score = 1.1  # we don't want it to determine upon finishing the first time; train for 1000 iterations always
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 25  # more than necessary (6) TODO
burn_in_length = 0  # TODO