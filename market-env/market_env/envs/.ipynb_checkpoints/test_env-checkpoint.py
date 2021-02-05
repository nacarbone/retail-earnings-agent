import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, Discrete # remove discrete eventually

#x = range(20,120)
#x = np.array([j + 10 if i % 2 == 0 else j - 10 for i, j in enumerate(range(20,120))]).reshape((len(x), 1))

# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

import gym
from gym.utils import seeding



class TestEnv_v0(gym.Env):
    
#     EPISODE = np.array([j + 10 if i % 2 == 0 else j - 10 for i, j in enumerate(range(20,120))]).reshape((100, 1))
    N_STEPS = 100
    OBS_LOW = 10.
    OBS_HIGH = 130.
    ACT_LOW = -1.
    ACT_HIGH = 1.
    MAX_STEPS = 100
    metadata = {
        "render.modes": ["human"]
        }
    
    def __init__ (self, start_balance=10000):
        
        self.observation_space = Box(low=self.OBS_LOW, high=self.OBS_HIGH, shape=(1,), dtype=np.float16)

        self.action_space = Box(low=self.ACT_LOW, high=self.ACT_HIGH, shape=(1,), dtype=np.float16)   
#        self.action_space = Discrete(2)
        
        self.period = np.array([j + 10 if i % 2 == 0 else j - 10 for i, j in enumerate(range(20,120))], dtype=np.float16).reshape(100,1)
        
        self.start_balance = start_balance
        
        # add render mode ?
        self.seed()
        self.reset()


    def reset (self):
        self.cash_balance = self.start_balance
        self.account_value = self.start_balance
        self.position_value = 0.
        self.n_shares = 0.
        
        self.current_step = 0
#        self.position = self.np_random.choice([1,2])

        self.state = self.period[self.current_step]
        self.reward = 0
        self.done = False
        self.info = {}
        
        return self.state

    def step (self, action):
        self.position_value = self.n_shares * self.state[0]
        
        if self.done:
            print('EPISODE DONE!!!')
        elif self.current_step == self.MAX_STEPS - 1:
            self.done = True;
        else:
            assert self.action_space.contains(action)
            
#            self.current_step += 1
            self.action = action
            action = action[0]

            if action < 0:
                shares_to_sell = self.n_shares * action
                value_sold = self.state[0] * shares_to_sell
                self.n_shares += shares_to_sell
                self.position_value += value_sold
                self.cash_balance -= value_sold
            if action > 0:
                value_bought = self.cash_balance * action
                shares_to_buy = value_bought / self.state[0]
                self.n_shares += shares_to_buy
                self.position_value += value_bought
                self.cash_balance -= value_bought
            else:
                pass
            
            self.current_step += 1
            
#            self.position_value = self.n_shares * self.current_step
            self.account_value = self.cash_balance + self.position_value   
            
            self.reward = self.account_value
#            self.state = np.array([self.current_step])
            self.state = self.period[self.current_step]

#         if self.account_value <= 0 or self.current_step == self.N_STEPS:
#             self.done = True
#         else:
# #            self.state = self.EPISODE[self.current_step]
#             self.state
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print('INVALID STATE', self.state)
        
        self.info['account_value'] = self.account_value

        
        return [self.state, self.reward, self.done, self.info]

        
    def render (self, mode='human'):
        s = 'Account Value: ${,.2f} || Price: ${,.2f} || Number of shares: {.0f}'
        print(s.format(self.account_value, self.state, self.n_shares))
        
    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close (self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass        
