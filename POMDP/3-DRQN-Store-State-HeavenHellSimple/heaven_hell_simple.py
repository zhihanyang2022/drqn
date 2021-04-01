import numpy as np

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
        return self.convert_state_to_obs(self.state)

    def step(self, action):

        done = False
        reward = 0

        # ========== heaven on the left ==========

        if self.state == 0:

            raise NotImplementedError

        elif self.state == 1:

            if action == 1:  # right
                self.state = 2
                done = True
                reward = -1  # hell on the right
            elif action == 2:  # down
                self.state = 3
            elif action == 3:  # left
                self.state = 0
                done = True
                reward = 1  # heaven on the left

        elif self.state == 2:

            raise NotImplementedError

        elif self.state == 3:  # only reachable states are 2 and 4

            if action == 0:  # up
                self.state = 1
            elif action == 2:  # down
                self.state = 4

        elif self.state == 4:  # only reachable state is 3

            if action == 0:  # up
                self.state = 3

        # ========== heaven on the right ==========

        elif self.state == 5:

            raise NotImplementedError

        elif self.state == 6:  # only reachable states are 5, 7, 8

            if action == 3:  # left
                self.state = 5
                done = True
                reward = -1  # hell on the left
            elif action == 1:  # right
                self.state = 7
                done = True
                reward = 1  # heaven on the right
            elif action == 2:  # down
                self.state = 8

        elif self.state == 7:

            raise NotImplementedError

        elif self.state == 8:  # only reachable states are 6, 9

            if action == 0:  # up
                self.state = 6
            elif action == 2:  # down
                self.state = 9

        elif self.state == 9:  # only reachable state is 8

            if action == 0:
                self.state = 8

        self.time_steps_elapsed += 1

        if self.time_steps_elapsed >= self.timeout:
            done = True

        return self.convert_state_to_obs(self.state), reward, done