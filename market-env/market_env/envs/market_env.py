# Follows the paradigm described by:
# https://github.com/DerwenAI/gym_example
# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

# TODO
# 1) MASK OUT ESTIMATE IN OBSERVATIONS PRE-EARNINGS DATE

import os
import json
from collections import defaultdict

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict # remove discrete eventually

default_config = {
    'start_balance' : 10000.,
    'seq_len' : 0,
    'obs_dim' : 6,
    'obs_range_low' : -10e2,
    'obs_range_high' : 10e2,
    'est_dim' : 7,
    'est_range_low' : -5e2,
    'est_range_high' : 5e2,
    'action_embedding_dim' : 100,
    'action_embedding_range_low' : -5,
    'action_embedding_range_high' : 5,
    'holding_mask_start_probability' : 0,
    'max_avail_actions' : 1000,
    'max_cash_balance' : 1e5,
    'max_position_value' : 1e5,
    'max_current_price' : 1e4,
    'max_shares' : 10000,
    'skip_val' : 4,
    'rand_skip' : False,
    'rand_skip_low' : 2,
    'rand_skip_high' : 5,
    'input_path' : 'processed_data/train/',
    'write' : False
}

class MarketEnv_v0(gym.Env):
        
    def __init__ (self, custom_env_config):
#         self.config = defaultdict(lambda: None)
#         self.config.update(default_config)
        self.config = default_config
        self.config.update(custom_env_config)
        
        for key in self.config:
            setattr(self, key, self.config[key])        

        self.files = [file for file in os.listdir(self.input_path) 
                      if '.json' in file]            
            
        self.obs_dim = np.zeros((self.obs_dim,))
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
                -5, 5, shape=(
                    3, 
                    self.max_avail_actions,
                    self.action_embedding_dim)),
            'cash_balance' : Box(
                0, self.max_cash_balance, shape=(1,)),
            'n_shares' : Discrete(self.max_shares),
            'real_obs' : Box(self.obs_dim_low, self.obs_dim_high),
            'estimate' : Box(self.est_dim_low, self.est_dim_high)
        })
    
        self.action_space = Dict({
            'buy/sell/hold' : Discrete(3),
            'amount' : Discrete(self.max_avail_actions)
        })

        self.seed(1)
            
        self.buying_embeddings = self.np_random.randn(
            self.max_avail_actions,
            self.action_embedding_dim
        )
        self.buying_embeddings = np.clip(
            self.buying_embeddings,
            self.action_embedding_range_low,
            self.action_embedding_range_high
        )
        self.selling_embeddings = self.np_random.randn(
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
        
        self.holding_mask_value = np.random.choice(
            [0,1],
            p=[1-self.holding_mask_start_probability,
               self.holding_mask_start_probability]
        )
        
        self.action_type_mask = np.array(
            [min(1, self.shares_avail_to_buy),
             min(1, self.shares_avail_to_sell),
             self.holding_mask_value]
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
        self.current_file = self.np_random.choice(self.files)
        self.current_symbol, self.current_earnings_date = \
            self.current_file.replace('.json', '').split('-')
        with open(self.input_path + self.current_file, 'r') as f:
            self.episode = json.load(f)    
        self.timesteps = np.array(self.episode['data'])
        self.price = np.array(self.episode['price'])
        self.estimate = np.array(self.episode['estimate'])        

        self.current_step = 0
        self.current_timestep = self.timesteps[self.current_step]
        self.max_steps = len(self.timesteps) - self.seq_len - 1
        self.current_price = self.price[self.current_step]
        self.cash_balance = np.array([self.start_balance])
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
            'cash_balance' : self.cash_balance / self.max_cash_balance,
            'n_shares' : self.n_shares,
            'real_obs' : self.current_timestep,
            'estimate' : self.estimate
        }
        
        # if specified to record episodes, setup an output file to do so
        if self.config['write']:
            results_files = [file
                             for file in os.listdir('rl_model_results')
                             if self.current_symbol 
                             and self.current_earnings_date in file]
            if len(results_files) > 0:
                max_file_number = max([int(file.replace(
                    '.txt', '').split('_')[-1]) 
                                       for file in results_files])
            else:
                max_file_number = 0
            self.output_file = 'rl_model_results/' + '_'.join([
                self.current_symbol,
                self.current_earnings_date,
                str(max_file_number)
            ]) + '.txt'
        
        return self.state

    def step (self, action):
        if self.done:
            print('EPISODE DONE!!!')
        elif self.current_step >= self.max_steps:
            self.done = True;
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
            
            if self.config['write']:
                line = ' '.join([
                    'file: {} | step: {} | cash: {:.2f} |',
                    'n_shares: {:.0f} | reward: {:.6f} |',
                    'account_value: {:.2f} | action_type: {} |',
                    'amount: {}'
                ]).format(
                    self.current_file, 
                    self.current_step, 
                    self.cash_balance[0], 
                    self.n_shares, 
                    self.reward, 
                    self.account_value, 
                    buy_sell_hold, 
                    amount
                )            

                with open(self.output_file, 'a') as f:                
                    f.write("%s\n" % line)            
            
            if self.config['rand_skip']:
                self.skip_val = self.np_random.randint(2,5)

            self.current_step += 1 + self.skip_val
            self.current_step = min(self.current_step, self.max_steps)

        self.current_timestep = self.timesteps[self.current_step]
        self.current_price = self.price[self.current_step]
        self.position_value = self.n_shares * self.current_price
        
        new_account_value = self.cash_balance[0] + self.position_value
        self.reward = (new_account_value - self.account_value) \
            / self.account_value
        self.account_value = new_account_value   

        self.holding_mask_start_probability = \
            self.holding_mask_start_probability ** self.current_step
        self.update_avail_actions()
        
        self.state = {
            'action_type_mask' : self.action_type_mask,
            'action_mask' : self.action_mask,
            'action_embeddings' : self.action_embeddings,
            'cash_balance' : self.cash_balance,
            'n_shares' : self.n_shares,
            'real_obs' : self.current_timestep,
            'estimate' : self.estimate
        }
        
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