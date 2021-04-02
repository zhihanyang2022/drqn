import numpy as np
from collections import namedtuple
Actions = namedtuple("Actions", "up right down left")
actions = Actions(0, 1, 2, 3)

class ActionSpace:

    def __init__(self, action_space_dim):
        self.action_space_dim = action_space_dim

    def sample(self):
        return np.random.randint(self.action_space_dim)

class HeavenHellSimple:

    """
    Starting states are 3 and 8

    States:
    
    Heaven left
    0 1 2
      3
      4

    Observation:
    0 1 2
      3
      4
      
    Heaven right
    5 6 7
      8
      9

    Observation:
    0 1 2
      3
      5
    """

    def __init__(self):
        self.time_steps_elapsed = 0
        self.visited_priest = False  # not exposed to the learning agent, only to the entropy reduction expert
        self.action_space_dim = 4
        self.action_space = ActionSpace(action_space_dim=self.action_space_dim)
        self.observation_space_dim = 6  # 0, 1, 2, 3, 4, 5
        self.timeout = 20
        self.state_to_obs = {
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4,  # priest
            5 : 0,
            6 : 1,
            7 : 2,
            8 : 3,
            9 : 5   # priest
        }

    def convert_state_to_obs(self, state):
        return self.state_to_obs[state]

    def reset(self):
        self.state = np.random.choice([3, 8])
        self.time_steps_elapsed = 0
        self.visited_priest = False
        return self.convert_state_to_obs(self.state)

    def get_fully_observable_expert_actions(self):
        """
        Heaven left
        0 1 2
          3
          4

        Heaven right
        5 6 7
          8
          9
        """
        if self.state in [0, 5]:
            raise NotImplementedError
        elif self.state in [1]:
            return [actions.left]
        elif self.state in [2, 7]:
            raise NotImplementedError
        elif self.state in [3, 8]:
            return [actions.up]
        elif self.state in [4, 9]:
            return [actions.up]
        elif self.state in [6]:
            return [actions.right]

    def get_entropy_reduction_expert_actions(self):
        if self.visited_priest is False:  # try to get to the priest as fast as possible
            if self.state in [0, 5]:
                raise NotImplementedError
            elif self.state in [1, 6]:
                return [actions.down]
            elif self.state in [2, 7]:
                raise NotImplementedError
            elif self.state in [3, 8]:
                return [actions.down]
            elif self.state in [4, 9]:
                raise NotImplementedError  # 4, 9 are priest states; so self.visited_priest can't be false
        else:  # uniform distribution over actions
            return list(range(self.action_space_dim))

    def get_expert_actions(self):
        return self.get_fully_observable_expert_actions() + self.get_entropy_reduction_expert_actions()

    def step(self, action):

        done = False
        reward = 0

        # ========== heaven on the left ==========

        if self.state == 0:

            raise NotImplementedError

        elif self.state == 1:

            if action == actions.right:  # right
                self.state = 2
                done = True
                reward = -1  # hell on the right
            elif action == actions.down:  # down
                self.state = 3
            elif action == actions.left:  # left
                self.state = 0
                done = True
                reward = 1  # heaven on the left

        elif self.state == 2:

            raise NotImplementedError

        elif self.state == 3:  # only reachable states are 2 and 4

            if action == actions.up:  # up
                self.state = 1
            elif action == actions.down:  # down
                self.state = 4

        elif self.state == 4:  # only reachable state is 3

            if action == actions.up:  # up
                self.state = 3

        # ========== heaven on the right ==========

        elif self.state == 5:

            raise NotImplementedError

        elif self.state == 6:  # only reachable states are 5, 7, 8

            if action == actions.left:  # left
                self.state = 5
                done = True
                reward = -1  # hell on the left
            elif action == actions.right:  # right
                self.state = 7
                done = True
                reward = 1  # heaven on the right
            elif action == actions.down:  # down
                self.state = 8

        elif self.state == 7:

            raise NotImplementedError

        elif self.state == 8:  # only reachable states are 6, 9

            if action == actions.up:  # up
                self.state = 6
            elif action == actions.down:  # down
                self.state = 9

        elif self.state == 9:  # only reachable state is 8

            if action == actions.up:
                self.state = 8

        self.time_steps_elapsed += 1

        if self.state in [4, 9]:
            self.visited_priest = True

        if self.time_steps_elapsed >= self.timeout:
            done = True

        return self.convert_state_to_obs(self.state), reward, done