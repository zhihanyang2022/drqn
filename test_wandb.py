import wandb
import numpy as np

run = wandb.init(
    project="drqn",
    entity='pomdpr',
    group='test',
    settings=wandb.Settings(_disable_stats=True),
    name='test-01',
    reinit=True
)

for i in range(100):
    wandb.log({'y': i + np.random.normal()})

run.finish()

run = wandb.init(
    project="drqn",
    entity='pomdpr',
    group='test',
    settings=wandb.Settings(_disable_stats=True),
    name='test-02',
    reinit=True
)

for i in range(100):
    wandb.log({'y': i + np.random.normal()})

run.finish()

wandb.init(
    project="drqn",
    entity='pomdpr',
    group='test',
    settings=wandb.Settings(_disable_stats=True),
    name='test-03'
)

for i in range(100):
    wandb.log({'y': i + np.random.normal()})