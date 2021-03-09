import os
import json

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict

DEFAULT_CONFIG = {
    '_seed' : None,
    'start_balance' : 10000.,
    'seq_len' : 15,
    'n_symbols' : 5,
    'obs_dim' : 6,
    'obs_range_low' : -10e2,
    'obs_range_high' : 10e2,
    'est_dim' : 6,
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
    'input_path' : os.path.join('processed_data', 'train'),
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
    """
    A "stock market" environment to handle agent actions and generate new 
    observations. Adapted from the framework detailed here: 

    https://github.com/DerwenAI/gym_example

    Attributes
    ---
    action_space : gym.Spaces
        The action space for the environment
    observation_space : gym.Spaces
        The observation space for the environment
    buying_embeddings : np.array
        The embeddings used for buying actions
    selling_embeddings : np.array
        The embeddings used for selling actions
    holding_embeddings : np.array
        The embeddings used for holding actions; array of zeroes
    action_embeddings : np.array
        Concatenation of buying, selling and holding embeddings
        to be passed through the model
    buying_action_mask : numpy.array
        A mask passed to the model for invalid buying amounts
    selling_action_mask : numpy.array
        A mask passed to the model for invalid selling amounts
    holding_action_mask : numpy.array
        A mask passed to the model for invalid selling amounts (i.e. any
        non-zero quantities)
    action_mask : numpy.array
        The concatenated buying, selling and holding masks
    file_num : int
        The current file number in the input file directory; only useful if
        shuffle_files is False
    files : list
        A list of filenames in the input file directory
    current_file : str
        The filename for the current episode; must be formatted like
        "<SYMBOL>-<4 DIGIT YEAR><2 DIGIT MONTH><2 DIGIT DAY>.json"
    current_symbol : str in {'AMZN', 'COST', 'KR', 'WBA', 'WMT'}
        The current stock symbol based on current_file
    current_earnings_date : str
        The current stock symbol based on current_file
    symbol_id : int in {0, 1, 2, 3, 4}
        The id for based on current_symbol
    episode : np.array
        The normalized OHLCV data across the current episode
    price : np.array
        The un-normalized mean price across the current episode
    estimate : np.array
        The estimate data for the current episode
    actual : float
        The actual EPS recorded by the company
    estimate_swap_flag : bool
        Used to swap between the masked actual value and the actual
        once earnings are reported
    max_steps : int
        The length of the episode
    current_step : int
        The location of the current timestep within the episode
    current_price : float
        The current un-normalized price of the asset
    reward : float
        The reward observed by the agent at the current timestep
    done : bool
        Whether an episode has ended or not
    info : dict
        Captures some basic information on the environment, but not actually
        used by Ray
    cash_balance : np.array
        The agent's current cash balance
    cash_pct : np.array
        The percentage of the agent's total account value held in cash
    account_value : float
        The current total account value of the agent
    n_shares : int
        The current number of shares held by the agent
    position_value : float
        The current value of the agent's position
    current_timestep : numpy.array
        An of the 15 most recent OHLCV data observed
    a1_record : numpy.array
        An array of the 15 most recent buy, sell or hold actions taken by the
        agent
    a2_record : numpy.array
        An array of the 15 most recent amount actions taken by the agent      
    n_shares_record : numpy.array
        An array of the 15 most recent amounts of shares held by the agent
    cash_record : numpy.array
        An array of the 15 most recent cash balances for the agent    
    shares_avail_to_buy : int
        The number of shares an agent can purchase based on the cash_balance
        and observed price
    shares_avail_to_sell : numpy.array
        The number of shares a user can sell based on n_shares
    state : dict
        A dict containing the agent's state that will be fed to the model 
    config : dict
        The model's configuration containing mostly of its shape and range
        attributes
    _seed : int or None
        The RNG seed for shuffling files and generating embeddings
    np_random : np.random object
        The RNG used by the environment, seeded based on _seed
    start_balance : float
        The agent's starting cash balance
    seq_len : int
        The length of sequences to be fed through the model
    n_symbols : int
        The number of symbols that can be handled by the model; should be
        5 for this model
    obs_dim : int
        The dimension of the OHLCV data
    obs_range_low : float
        The lower bound of the OHLCV data range
    obs_range_high : float
        The upper bound of the OHLCV data range
    est_dim : int
        The dimension of the estimate data
    est_range_low : float
        The lower bound of the estimate data range        
    est_range_high : float
        The upper bound of the estimate data range        
    action_embedding_dim : int
        The dimension of the action embeddings
    action_embedding_range_low : int or float
        The lower bound of the action embeddings range; values will be clipped
        to this amount. A uniform [0,1) distribution is used to generate
        embeddings, so anything outside this is meaningless
    action_embedding_range_high ; int or float
        The upper bound of the action embeddings range; values will be clipped
        to this amount. A uniform [0,1) distribution is used to generate
        embeddings, so anything outside this is meaningless
    max_avail_actions : int
        The maximum number of shares an agent can buy or sell at one time
    max_cash_balance : float
        The maximum cash balance an agent can achieve; theoretically shouldn't
        be bounded, but needs to be work with Gym
    max_position_value : float
        The maximum position value an agent can achieve; theoretically
        shouldn't be bounded, but needs to be work with Gym        
    max_current_price : float
        The maximum normalized price of the asset; theoretically shouldn't be
        bounded, but needs to be work with Gym        
    max_shares : int
        The maximum number of shares an agent is capable of holding at once;
        theoretically shouldn't be bounded, but needs to be work with Gym    
    input_path : str
        The full filepath to where the agent can find input files to read
    output_path : str
        The directory to which to write the agent's state throughout an
        episode
    output_filename : str
        The filename to which to write the agent's state throughout an
        episode
    output_filepath : str
        The full filepath to which to write the agent's state throughout an
        episode
    shuffle_files : bool
        Whether or not to shuffle the files randomly; typically only False for
        testing

    Methods
    ---
    reset : dict
        Resets the environment's state to an initial state
    step : dict
        Takes one step through the environment based on the agent's actions
    update_avail_actions : None
        Updates the action masks for the current price of the asset
    seed : list
        Sets the seed for this env's random number generator(s).
    close : None
        Performs any self-cleanup upon exiting the environment
    init_output_file : None
        Initializes an output file to record an agent's actions
    write_state_to_output_file : None
        Writes an agent's state to the output file for further analysis
    """

    def __init__ (self, custom_env_config):
        """
        Parameters
        ---
        custom_env_config : dict
            A dict containing any attributes in DEFAULT_CONFIG to overwrite
        """
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
        """
        Updates the action masks for the current price of the asset
        """        
        self.shares_avail_to_buy = int(np.floor(
            self.cash_balance[0] / self.current_price))
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
        """
        Resets the environment's state to an initial state. This includes
        reading a new episode file.

        Returns
        ---
        A dict containing the initial state of the environment
        """

        if self.config['shuffle_files']:
            self.current_file = self.np_random.choice(self.files)
        else:
            self.current_file = self.files[self.file_num]
            self.file_num += 1
            self.file_num = (self.file_num) % len(self.files)
        self.current_symbol, self.current_earnings_date = \
            self.current_file.replace('.json', '').split('-')
        self.symbol_id = SYMBOL_IDS[self.current_symbol]
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
        """
        Takes one step through the environment based on the agent's actions,
        updating its state and generating a new observation accordingly.

        Parameters
        ---
        action : dict
            A dict of the action taken by the agent
        
        Returns
        ---
        A dict containing the environment's current state
        """        
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

#     def render (self, mode='human'):
#         """
#         Renders basic info on an agent's state. Can be used in Gym, but not by
#         RLLib.
        
#         Parameters
#         ---
        
#         """
#         s = 'Account Value: ${.2f} || Price: ${.2f} || Number of shares: {.0f}'
#         print(s.format(self.account_value, self.state, self.n_shares))

    def seed (self, seed=None):
        """
        Sets the seed for this env's random number generator(s).

        Returns 
        ---
        The list of seeds used in this env's random
              number generators.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#     def close (self):
#         """
#         Performs any self-cleanup upon exiting the environment.
#         Not currently implemented.
#         """
#         pass

    def init_output_file(self):
        """
        Initializes an output file to record an agent's actions. Raises an
        error if an output path is not found in the class attributes. Filename
        will be like
        "<SYMBOL>_<4 DIGIT YEAR><2 DIGIT MONTH><2 DIGIT DAY>.txt"
        """
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
        """
        Writes an agent's state to the output file for further analysis.

        Parameters
        ---
        buy_sell_hold : int
            The buy, sell or hold action taken by the agent
        amount : int
            The amount action taken by the agent
        """
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
