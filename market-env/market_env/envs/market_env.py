# Follows the paradigm described by:
# https://github.com/DerwenAI/gym_example
# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

import os
import json
from collections import deque

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict

DEFAULT_CONFIG = {
    '_seed' : None,
    'start_balance' : 10000.,
    'seq_len' : 15,
    'n_symbols' : 5,
    'obs_dim' : 7,
    'obs_range_low' : -10e2,
    'obs_range_high' : 10e2,
    'est_dim' : 7,
    'est_range_low' : -5e2,
    'est_range_high' : 5e2,
    'action_embedding_dim' : 50,
    'action_embedding_range_low' : -1,
    'action_embedding_range_high' : 1,
    'max_avail_actions' : 1000,
    'max_cash_balance' : 2e4,
    'max_position_value' : 2e4,
    'max_current_price' : 1e4,
    'max_shares' : 1000,
    'input_path' : 'processed_data/train/',
    'write' : False,
    'output_path' : None,
    'shuffle_files' : True
}

SYMBOL_IDS = {
        'AMZN' : 0,
        'COST' : 1,
        'KR' : 2,
        'WBA' : 3,
        'WMT' : 4
}


class MarketEnv_v0(gym.Env):
        
    def __init__ (self, custom_env_config):
        self.file_num = 0
        
        self.config = DEFAULT_CONFIG
        self.config.update(custom_env_config)
        
        for key in self.config:
            setattr(self, key, self.config[key])        

        self.files = sorted([file for file in os.listdir(self.input_path)
                      if '.json' in file])            
            
        self.obs_dim = np.zeros((self.seq_len, self.obs_dim))
        self.obs_dim_low = self.obs_dim.copy()
        self.obs_dim_low.fill(self.obs_range_low)
        self.obs_dim_high = self.obs_dim.copy()
        self.obs_dim_high.fill(self.obs_range_high)
        
        self.est_dim = np.zeros((self.est_dim,))
        self.est_dim_low = self.est_dim.copy()
        self.est_dim_low.fill(self.est_range_low)
        self.est_dim_high = self.est_dim.copy()
        self.est_dim_high.fill(self.est_range_high)
        
        self.observation_space = Dict({
            'action_type_mask' : Box(0, 1, shape=(3,)),
            'action_mask' : Box(
                0, 1, shape=(3, self.max_avail_actions)),
            'action_embeddings' : Box(
                self.action_embedding_range_low, 
                self.action_embedding_range_high, 
                shape=(
                    3, 
                    self.max_avail_actions,
                    self.action_embedding_dim)),
            'symbol_id' : Discrete(self.n_symbols),
            'cash_record' : Box(
                0, 1, shape=(self.seq_len, 1)),
            'n_shares_record' : Box(
                0, 1, shape=(self.seq_len, self.max_shares)),
            'a1_record' : Box(
                0, 1, shape=(self.seq_len, 3)),
            'a2_record' : Box(
                0, 1, shape=(self.seq_len, self.max_avail_actions)),
            'price' : Box(self.obs_dim_low, self.obs_dim_high),
            'estimate' : Box(self.est_dim_low, self.est_dim_high)
        })
    
        self.action_space = Dict({
            'a1' : Discrete(3),
            'a2' : Discrete(self.max_avail_actions)
        })

        self.seed(self._seed)
            
        self.buying_embeddings = self.np_random.rand(
            self.max_avail_actions,
            self.action_embedding_dim
        )
        self.buying_embeddings = np.clip(
            self.buying_embeddings,
            self.action_embedding_range_low,
            self.action_embedding_range_high
        )
        self.selling_embeddings = self.np_random.rand(
            self.max_avail_actions,
            self.action_embedding_dim,
        )
        self.selling_embeddings = np.clip(
            self.selling_embeddings,
            self.action_embedding_range_low,
            self.action_embedding_range_high
        )
        self.holding_embeddings = np.zeros((
            self.max_avail_actions,
            self.action_embedding_dim
        ))
        self.action_embeddings = np.stack(
            [self.buying_embeddings, 
             self.selling_embeddings, 
             self.holding_embeddings]
        )

        self.reset()

    def update_avail_actions(self):
        self.shares_avail_to_buy = int(np.floor(
            self.cash_balance[0] / self.current_price)
                                      )
        self.shares_avail_to_sell = self.n_shares
        
        self.action_type_mask = np.array(
            [min(1, self.shares_avail_to_buy),
             min(1, self.shares_avail_to_sell),
             1]
        )
        
        self.buying_action_mask = np.zeros((self.max_avail_actions))
        self.selling_action_mask = np.zeros((self.max_avail_actions))
        self.holding_action_mask = np.zeros((self.max_avail_actions))
        
        self.buying_action_mask[:self.shares_avail_to_buy] = 1
        self.selling_action_mask[:self.shares_avail_to_sell] = 1
        self.holding_action_mask[0] = 1

        self.action_mask = np.stack([self.buying_action_mask,
                                     self.selling_action_mask,
                                     self.holding_action_mask])
        
    def reset (self):
        if self.config['shuffle_files']:
            self.current_file = self.np_random.choice(self.files)
        else:
            self.current_file = self.files[self.file_num]
            self.file_num += 1
            self.file_num = (self.file_num) % len(self.files)
#        print(self.current_file)
        self.current_symbol, self.current_earnings_date = \
            self.current_file.replace('.json', '').split('-')
        self.symbol_id = symbol_ids[self.current_symbol]
        with open(os.path.join(self.input_path, self.current_file), 'r') as f:
            self.episode = json.load(f)    
        self.timesteps = np.array(self.episode['data'])
        self.price = np.array(self.episode['price'])
        self.estimate = np.array(self.episode['estimate'])
        # the actual value would not be known until after the second day so replace 
        # this value with the mean, will be added back in eventually in step()
        self.actual_value = self.estimate[-1]
        self.estimate[-1] = self.estimate[3]
        self.estimate_swap_flag = False
        
        self.current_step = 0
        self.current_timestep = self.timesteps[self.current_step:self.current_step+self.seq_len]
        self.max_steps = len(self.timesteps) - self.seq_len - 1
        self.current_price = self.price[self.current_step+self.seq_len-1]
        self.cash_balance = np.array([self.start_balance])
        self.account_value = self.start_balance
        self.cash_pct = self.cash_balance / self.account_value
        self.n_shares = 0
        self.position_value = 0

        self.reward = 0.
        
        self.a1_record = np.zeros((self.seq_len, 3))
        self.a1_record[:,-1] = 1
        self.a2_record = np.zeros((self.seq_len, self.max_avail_actions))
        self.a2_record[:,0] = 1
        self.n_shares_record = np.zeros((self.seq_len, self.max_shares))
        self.n_shares_record[:,0] = 1
        self.cash_record = np.zeros((self.seq_len, 1))
        self.cash_record.fill(1)
        

        self.done = False
        self.info = {}
        
        self.update_avail_actions()
        
        self.state = {
            'action_type_mask' : self.action_type_mask,
            'action_mask' : self.action_mask,
            'action_embeddings' : self.action_embeddings,
            'symbol_id' : self.symbol_id,
            'cash_record' : self.cash_record,
            'n_shares_record' : self.n_shares_record,
            'a1_record' : self.a1_record,
            'a2_record' : self.a2_record,
            'price' : self.current_timestep,
            'estimate' : self.estimate
        }
        
        # if specified to record episodes, setup an output file to do so
        if self.config['write']:
            self.init_output_file()
            self.write_state_to_output_file('n/a','n/a')
        
        return self.state

    def step (self, action):
        if self.done:
            print('Episode done!')
        elif self.current_step >= self.max_steps:
            self.done = True;
        else:
            try:
                assert self.action_space.contains(action)
            except:
                raise
        
            a1 = action['a1']
            a2 = action['a2']
        
            if a1 == 0:
                value_to_buy = self.current_price * a2
                self.n_shares += a2
                self.cash_balance -= value_to_buy
                self.position_value += value_to_buy
            elif a1 == 1:
                value_to_sell = self.current_price * a2
                self.n_shares -= a2
                self.cash_balance += value_to_sell
                self.position_value -= value_to_sell
            else:
                pass
            
            self.a1_record = np.roll(self.a1_record, -1, axis=0)
            self.a1_record[-1].fill(0)
            self.a1_record[-1, a1] = 1
            self.a2_record = np.roll(self.a2_record, -1, axis=0)
            self.a2_record[-1].fill(0)
            self.a2_record[-1, a2] = 1            
                      
            self.current_step += 1

        self.current_timestep = self.timesteps[
            self.current_step:self.current_step+self.seq_len]
        last_price = self.current_price
        self.current_price = self.price[self.current_step+self.seq_len-1]
        self.position_value = self.n_shares * self.current_price
        
        new_account_value = self.cash_balance[0] + self.position_value

        holding_cost = new_account_value - self.account_value
        opportunity_cost = ((self.cash_balance[0] / last_price)
            * (self.current_price - last_price))
        self.reward = holding_cost - opportunity_cost

        self.account_value = new_account_value
    
        self.cash_pct = self.cash_balance / self.account_value
        self.cash_record = np.roll(self.cash_record, -1, axis=0)
        self.cash_record[-1] = self.cash_pct
        self.n_shares_record = np.roll(self.n_shares_record, -1, axis=0)
        self.n_shares_record[-1].fill(0)
        self.n_shares_record[-1, self.n_shares] = 1
        
        self.update_avail_actions()
        
        # when the relative offset from the earnings date = 0,
        # swap the actual value back into the estimate
        if self.current_timestep[-1,-1] == 0 and not self.estimate_swap_flag:
            self.estimate[-1] = self.actual_value
            self.estimate_swap_flag = True
        
        self.state = {
            'action_type_mask' : self.action_type_mask,
            'action_mask' : self.action_mask,
            'action_embeddings' : self.action_embeddings,
            'symbol_id' : self.symbol_id,
            'cash_record' : self.cash_record,
            'n_shares_record' : self.n_shares_record,
            'a1_record' : self.a1_record,
            'a2_record' : self.a2_record,
            'price' : self.current_timestep,
            'estimate' : self.estimate
        }
        
        self.info['account_value'] = self.account_value
        
        if self.config['write']:
            if 'a1' and 'a2' in locals():
                self.write_state_to_output_file(a1, a2)
        
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
    
    def init_output_file(self):
        if not self.output_path:
            raise FileNotFoundError('No output path specified.')
            
        self.output_filename = '_'.join([
            self.current_symbol,
            self.current_earnings_date,
        ]) + '.txt'

        self.output_filepath = os.path.join(
            self.output_path, self.output_filename)
        
        header = ','.join([
            'file',
            'step',
            'cash',
            'n_shares',
            'reward',
            'account_value',
            'current_price',
            'action_type',
            'amount'
        ])

        with open(self.output_filepath, 'w') as f:                
            f.write("%s\n" % header)
            
    def write_state_to_output_file(self, buy_sell_hold, amount):
        line = ','.join([
            '{}'.format(self.current_file), 
            '{:d}'.format(self.current_step), 
            '{:.2f}'.format(self.cash_balance[0]), 
            '{:d}'.format(self.n_shares), 
            '{:.4f}'.format(self.reward), 
            '{:.2f}'.format(self.account_value),
            '{:.2f}'.format(self.current_price),
            '{}'.format(buy_sell_hold),
            '{}'.format(amount)
        ])            

        with open(self.output_filepath, 'a') as f:                
            f.write("%s\n" % line)
