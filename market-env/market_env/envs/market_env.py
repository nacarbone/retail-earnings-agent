# Follows the paradigm described by:
# https://github.com/DerwenAI/gym_example
# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

# TODO
# 1) REPLACE HARD-CODED INTEGER VALUES WITH CLASS ATTRIBUTES
# 2) ADD INPUT PATH AS CLASS ATTRIBUTE
# 3) CONVERT ENTIRE JSON DATA TO NUMPY INSTEAD OF DOING SO AT EACH TIMESTEP (IN STEP)
# 4) MASK OUT ESTIMATE IN OBSERVATIONS PRE-EARNINGS DATE

import os
import json
from collections import defaultdict

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict # remove discrete eventually

class MarketEnv_v0(gym.Env):
    
    MAX_AVAIL_ACTIONS = int(1e4)
    ACTION_EMBEDDING_SIZE = 10

    FILES = [file for file in os.listdir('processed_data/train') if '.json' in file]
    
    metadata = {
        "render.modes": ["human"]
        }
    
    def __init__ (self, config):        
        self.config = defaultdict(lambda: None)
        self.config.update(config)
        
        self.seq_len = 0 # update this in case the RNN version is implemented
        self.n_features = 6
        self.start_balance = 10000.

        self.obs_dim = np.zeros((6,))
        
        self.obs_dim_low = self.obs_dim.copy()
        self.obs_dim_low.fill(-10e2)
        self.obs_dim_high = self.obs_dim.copy()
        self.obs_dim_high.fill(10e2)
        
        self.est_dim = np.zeros((7,))
        self.est_dim_low = self.est_dim.copy()
        self.est_dim_low.fill(-10e2)
        self.est_dim_high = self.est_dim.copy()
        self.est_dim_high.fill(10e2)   
        
        self.observation_space = Dict({
            'action_type_mask' : Box(0, 1, shape=(3,)),
            'action_mask' : Box(0, 1, shape=(3, 10000)),
            'action_embeddings' : Box(-5, 5, shape=(3, 10000, 10)),
            'cash_balance' : Box(low=0, high=1e6, shape=(1,)),
            'n_shares' : Discrete(1000),
            'real_obs' : Box(self.obs_dim_low, self.obs_dim_high),
            'estimate' : Box(self.est_dim_low, self.est_dim_high)
        })
    
        self.action_space = Dict({
            'buy/sell/hold' : Discrete(3), # add another input for holding
            'amount' : Discrete(10000)
        })
        
        self.seed(1)
        
        self.buying_embeddings = self.np_random.randn(self.action_space['amount'].n, self.ACTION_EMBEDDING_SIZE)
        self.buying_embeddings = np.clip(self.buying_embeddings, -5, 5)
        self.selling_embeddings = self.np_random.randn(self.action_space['amount'].n, self.ACTION_EMBEDDING_SIZE)
        self.selling_embeddings = np.clip(self.selling_embeddings, -5, 5)
        self.holding_embeddings = np.zeros((self.action_space['amount'].n, self.ACTION_EMBEDDING_SIZE))
        self.action_embeddings = np.stack([self.buying_embeddings, self.selling_embeddings, self.holding_embeddings])
   
        # add render mode ?

        self.reset()

    def update_avail_actions(self):        
        self.max_shares_to_buy = self.action_space['amount'].n
        self.max_shares_to_sell = self.action_space['amount'].n
        
        self.shares_avail_to_buy = int(np.floor(self.cash_balance / self.current_price))
        self.shares_avail_to_sell = self.n_shares

        self.action_type_mask = np.array([min(1, self.shares_avail_to_buy), min(1, self.shares_avail_to_sell), 1])
        
        self.buying_action_mask = np.zeros((self.max_shares_to_buy))
        self.selling_action_mask = np.zeros((self.max_shares_to_sell))
        self.holding_action_mask = np.zeros((self.max_shares_to_sell))
        
        self.buying_action_mask[:self.shares_avail_to_buy] = 1
        self.selling_action_mask[:self.shares_avail_to_sell] = 1
        
        self.action_mask = np.stack([self.buying_action_mask, self.selling_action_mask, self.holding_action_mask])
        
    def reset (self):
        self.current_file = self.np_random.choice(self.FILES)
        self.current_symbol, self.current_earnings_date = self.current_file.replace('.json', '').split('-')
        
        with open('processed_data/train/' + self.current_file, 'r') as f:
            self.episode = json.load(f)
        
        self.current_step = 0
        
        self.timesteps = self.episode['data']
        self.price = self.episode['price']
        self.estimate = np.array(self.episode['estimate']) # UPDATE THIS FOR THE PROPER SPELLING
        
        self.current_timestep = np.array(self.timesteps[self.current_step])
        self.current_price = self.price[self.current_step]
        
        self.MAX_STEPS = len(self.timesteps) - self.seq_len

#        self.MAX_STEPS = 10        
#        self.max_episode_steps = self.MAX_STEPS
        
        self.cash_balance = self.start_balance
        self.account_value = self.start_balance
        self.position_value = 0.
        self.n_shares = 0
        self.reward = 0.

        self.done = False
        self.info = {}

        self.update_avail_actions()
        
        self.state = {
            'action_type_mask' : self.action_type_mask,
            'action_mask' : self.action_mask,
            'action_embeddings' : self.action_embeddings,
                'cash_balance' : np.array([self.cash_balance]),
                'n_shares' : self.n_shares,
            'real_obs' : self.current_timestep,
            'estimate' : self.estimate
        }
        
        if self.config['write']:
            results_files = [x for x in os.listdir('rl_model_results') if '.txt' in x]
            if len(results_files) > 0:
                max_file_number = max([int(file.split('_')[0]) for file in results_files])
            else:
                max_file_number = 0
            self.output_file = 'rl_model_results/' + '_'.join([str(max_file_number), self.current_file]) + '.txt'
        
        return self.state

    def step (self, action):
        # TODO
        # Add penalty for too frequent trading?
        # Could mask the actions to prevent this
        
        # for debugging
#         print('file: {} | step: {} | cash: {:.2f} | n_shares: {:.2f} | reward: {:.6f} | account_value: {:.2f} | action: '.format(self.current_file, self.current_step, self.cash_balance, self.n_shares, self.reward, self.account_value), action)
        
        if self.done:
            print('EPISODE DONE!!!')
        elif self.current_step == self.MAX_STEPS:
            self.done = True;
        elif self.cash_balance <= 0:
            print('Cash balance gone!')
            self.done = True
            self.cash_balance = 0
        else:
            try:
                assert self.action_space.contains(action)
            except:
                raise
        
            buy_sell_hold = action['buy/sell/hold']
            amount = action['amount']
        
            if buy_sell_hold == 0:
                value_to_buy = self.current_price * amount
                self.n_shares += amount
                self.cash_balance -= value_to_buy
                self.position_value += value_to_buy
            elif buy_sell_hold == 1:
                value_to_sell = self.current_price * amount
                self.n_shares -= amount
                self.cash_balance += value_to_sell
                self.position_value -= value_to_sell
            else:
                pass
            
            
            self.current_step += 1

            
        self.current_timestep = np.array(self.timesteps[self.current_step])

        self.current_price = self.price[self.current_step]            
            
        self.position_value = self.n_shares * self.current_price

        new_account_value = self.cash_balance + self.position_value
        self.reward = (new_account_value - self.account_value) / self.account_value

        self.account_value = new_account_value   

        self.update_avail_actions()
        
        self.state = {
            'action_type_mask' : self.action_type_mask,
            'action_mask' : self.action_mask,
            'action_embeddings' : self.action_embeddings,
                'cash_balance' : np.array([self.cash_balance]),
                'n_shares' : self.n_shares,
            'real_obs' : self.current_timestep,
            'estimate' : self.estimate
        }

        if self.config['write']:
            line = 'file: {} | step: {} | cash: {:.2f} | n_shares: {:.0f} | reward: {:.6f} | account_value: {:.2f} | action_type: {}, | amount: {}'.format(self.current_file, self.current_step, self.cash_balance, self.n_shares, self.reward, self.account_value, buy_sell_hold, amount)

            with open(self.output_file, 'a') as f:                
                f.write("%s\n" % line)
        
        
# THIS FAILS WHEN CASH BALANCE < 0
# Should the above condition be alleviated
# by the masking?
#         try:
#             assert self.observation_space.contains(self.state)
#         except AssertionError:
#             print('INVALID STATE', self.state)
        
        self.info['account_value'] = self.account_value
        
        return [self.state, self.reward, self.done, self.info]

        
    def render (self, mode='human'):
        s = 'Account Value: ${.2f} || Price: ${.2f} || Number of shares: {.0f}'
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
