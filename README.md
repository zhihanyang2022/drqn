Modified from https://github.com/g6ling/Reinforcement-Learning-Pytorch-Cartpole.

How to use this repo:
- Make sure that you have installed miniconda (for Linux, see `https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html`)
- `cd` into this repo
- `cd packages`
- `conda create --name pomdpr python=3.7` where `pomdpr` stands for POMDP and Robotics
- `conda activate pomdpr`
- `chmod +x install_packages.sh` where `chmod` makes the bash script executable on your device
- `./install_packages.sh` installs `numpy scipy torch gym PyYAML wandb` and `rl_parsers` and `gym-pomdp` (these two are stored inside `drqn/packages`)
- `cd ..` back to the top level of the repo
- Test your installation using `python algorithms/POMDP/3-DRQN-Store-State-HeavenHell/train.py --lr=0.00005 --use_experts=0 --seed=1 --debug_mode=1` where `debug_mode=1` makes sure that `wandb` is not used
- `wandb login`
- Do anything you want now.

Note that if you modify the Heaven-Hell pomdp file (e.g., modify the initial belief or the starting state distribution) you will need to re-install gym-pomdp for the change to take effect.

---

Below is README from the original repo.

# PyTorch CartPole Example
Simple Cartpole example writed with pytorch.

## Why Cartpole?
Cartpole is very easy problem and is converged very fast in many case.
So you can run this example in your computer(maybe it take just only 1~2 minitue).

## Rainbow
- [x] DQN [[1]](#reference)
- [x] Double [[2]](#reference)
- [x] Duel [[3]](#reference)
- [x] Multi-step [[4]](#reference)
- [x] PER(Prioritized Experience Replay) [[5]](#reference)
- [x] Nosiy-Net [[6]](#reference)
- [x] Distributional(C51) [[7]](#reference)
- [x] Rainbow [[8]](#reference)

## PG(Policy Gradient)
- [x] REINFORCE [[9]](#reference)
- [x] Actor Critic [[10]](#reference)
- [x] Advantage Actor Critic
- [x] GAE(Generalized Advantage Estimation) [[12]](#reference)
- [x] TNPG [[20]](#reference)
- [x] TRPO [[13]](#reference)
- [x] PPO - Single Version [[14]](#reference)

## Parallel
- [x] Asynchronous Q-learning [[11]](#reference)
- [x] A3C (Asynchronous Advantage Actor Critic) [[11]](#reference)
- [x] ACER [[21]](#reference)
- [ ] PPO [[14]](#reference)
- [x] APE-X DQN [[15]](#reference)
- [ ] IMPALA [[23]](#reference)
- [ ] R2D2 [[16]](#reference)

## Distributional DQN
- [x] QRDQN [[18]](#reference)
- [x] IQN [[19]](#reference)

## Exploration
- [ ] ICM [[22]](#refercence)
- [ ] RND [[17]](#reference)

## POMDP (With RNN)
- [x] DQN (use state stack)
- [x] DRQN [[24]](#reference) [[25]](#reference)
- [x] DRQN (use state stack)
- [x] DRQN (store Rnn State) [[16]](#reference)
- [x] R2D2 - Single Version [[16]](#reference)


## Reference
[1][Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2][Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[3][Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[4][Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[5][Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[6][Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[7][A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[8][Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[9][Policy Gradient Methods for Reinforcement Learning with Function Approximation ](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)  
[10][Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)  
[11][Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)  
[12][HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438.pdf)  
[13][Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)  
[14][Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf)  
[15][DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/pdf/1803.00933.pdf)  
[16][RECURRENT EXPERIENCE REPLAY IN DISTRIBUTED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=r1lyTjAqYX)  
[17][EXPLORATION BY RANDOM NETWORK DISTILLATION](https://openreview.net/pdf?id=H1lJJnR5Ym)  
[18][Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf)  
[19][Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/pdf/1806.06923.pdf)  
[20][A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)  
[21][SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY](https://arxiv.org/pdf/1611.01224.pdf)  
[22][Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363.pdf)  
[23][IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/pdf/1802.01561.pdf)  
[24][Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/pdf/1507.06527.pdf)  
[25][Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.05521.pdf)  

## Acknowledgements
- https://github.com/openai/baselines
- https://github.com/reinforcement-learning-kr/pg_travel
- https://github.com/reinforcement-learning-kr/distributional_rl
- https://github.com/Kaixhin/Rainbow
- https://github.com/Kaixhin/ACER
- https://github.com/higgsfield/RL-Adventure-2

## Use Cuda
check this issue. https://github.com/g6ling/Reinforcement-Learning-Pytorch-Cartpole/issues/1
