import json

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import numpy as np
from yahoo_earnings_calendar import YahooEarningsCalendar
from gym.utils import seeding

from market_env.envs.market_env import MarketEnv_v0
from model import AutoregressiveParametricTradingModel
from action_dist import TorchMultinomialAutoregressiveDistribution

import gym

class InvalidInputError(Exception):
    
    def __init__(self, message):
        self.message = message

INPUT_KEYS = [
    'symbol',
    'earnings date',
    'trading date',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'minimum estimate',
    'maximum estimate',
    'mean estimate',
    'median estimate',
    'estimate standard deviation',
    'actual'
]

SYMBOL_KEY = INPUT_KEYS[0]
EARNINGS_DATE_KEY = INPUT_KEYS[1]
TRADING_DATE_KEY = INPUT_KEYS[2]
PRICE_KEYS = INPUT_KEYS[3:8]
EST_KEYS = INPUT_KEYS[8:]
#RESET_FLAG = INPUT_KEYS[-1]

OBS_RANGE_HIGH = 10e2
OBS_RANGE_LOW = -10e2
EST_RANGE_HIGH = 5e2
EST_RANGE_LOW = -5e2

SYMBOL_IDS = {
        'AMZN' : 0,
        'COST' : 1,
        'KR' : 2,
        'WBA' : 3,
        'WMT' : 4
}

# Follows the paradigm described by:
# https://github.com/DerwenAI/gym_example
# https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

# TODO
# Move logic to a handler above this class



# self.np_random, seed = seeding.np_random(self.SEED)


class ActionHandler():
    START_BALANCE = 10000
    MAX_AVAIL_ACTIONS = 1000
    MAX_SHARES = 1000
    # Update this obs_dim when retraining model
    OBS_DIM = 6
    SEQ_LEN = 15
    
    def __init__(self):
        self.cash_balance = self.START_BALANCE
        self.account_value = self.START_BALANCE
        self.n_shares = 0
        self.position_value = 0
        
        
        # seq_len = 15
        # obs_dim = 7
        self.price_roll = np.zeros((self.SEQ_LEN, self.OBS_DIM))
        
#        self.max_avail_actions = 1000
        
        # seed is 1

        
        self.a1_record = np.zeros((self.SEQ_LEN, 3))
        self.a1_record[:,-1] = 1
        self.a2_record = np.zeros((self.SEQ_LEN, self.MAX_AVAIL_ACTIONS))
        self.a2_record[:,0] = 1
        self.n_shares_record = np.zeros((self.SEQ_LEN, self.MAX_SHARES))
        self.n_shares_record[:,0] = 1
        self.cash_record = np.zeros((self.SEQ_LEN, 1))
        self.cash_record.fill(1)
     
    def update_avail_actions(self):
        self.shares_avail_to_buy = max(
            min(
            int(np.floor(self.cash_balance / self.current_price)),
            self.MAX_AVAIL_ACTIONS
        ), 0)
        self.shares_avail_to_sell = self.n_shares
        
        self.action_type_mask = np.array(
            [min(1, self.shares_avail_to_buy),
             min(1, self.shares_avail_to_sell),
             1]
        )
        
        self.buying_action_mask = np.zeros((self.MAX_AVAIL_ACTIONS))
        self.selling_action_mask = np.zeros((self.MAX_AVAIL_ACTIONS))
        self.holding_action_mask = np.zeros((self.MAX_AVAIL_ACTIONS))
        
        self.buying_action_mask[:self.shares_avail_to_buy] = 1
        self.selling_action_mask[:self.shares_avail_to_sell] = 1
        self.holding_action_mask[0] = 1

        self.action_mask = np.stack([self.buying_action_mask,
                                     self.selling_action_mask,
                                     self.holding_action_mask])

    def update_values_for_action(self, action):
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
        self.cash_pct = self.cash_balance / self.account_value
        self.cash_record = np.roll(self.cash_record, -1, axis=0)
        self.cash_record[-1] = self.cash_pct
        self.n_shares_record = np.roll(self.n_shares_record, -1, axis=0)
        self.n_shares_record[-1].fill(0)
        self.n_shares_record[-1, self.n_shares] = 1

    def update_values_for_new_input(self, new_price_data, normalized_new_price_data):
        """
        New price data should be 
        """
        self.price_roll = np.roll(self.price_roll, -1, axis=0)
        self.price_roll[-1] = normalized_new_price_data

        self.current_price = sum(new_price_data[:4]) / 4
        self.position_value = self.current_price * self.n_shares 
        self.account_value = self.cash_balance + self.position_value
        self.update_avail_actions()

    def get_state(self):
        return {
            'action_type_mask' : self.action_type_mask,
            'action_mask' : self.action_mask,
            'cash_record' : self.cash_record,
            'n_shares_record' : self.n_shares_record,
            'a1_record' : self.a1_record,
            'a2_record' : self.a2_record,
            'price' : self.price_roll,            
        }

class InputDataHandler():
    VALID_START = pd.Timestamp('2003-01-01', tz='UTC')
    # change to today
    VALID_END = pd.Timestamp('2022-12-31', tz='UTC')
    NYSE = mcal.get_calendar('NYSE')
    
    NORMALIZE_DAYS = 10
    
    def __init__(self, symbol):
        self.calendar = self.NYSE.schedule(start_date=self.VALID_START, end_date=self.VALID_END)
        self.valid_market_days = self.calendar['market_open']\
            .dt.normalize().reset_index(drop=True)
        
        self.symbol = symbol
        self.earnings_date = None
        self.trading_date = None        
        
        self.ticker = yf.Ticker(symbol)
        
        yec = YahooEarningsCalendar()
        earnings_data = yec.get_earnings_of(symbol)
        valid_earnings_dates = [pd.Timestamp(date['startdatetime']).normalize()
                                for date in earnings_data]
        self.valid_earnings_dates = [date for date in valid_earnings_dates
                               if date > self.VALID_START and date < self.VALID_END]        
        
        self.valid_model_periods = {}

        for date in self.valid_earnings_dates:
            date_ix = self.valid_market_days.loc[self.valid_market_days == date].index[0]
            # valid model days back
            valid_model_days = self.valid_market_days.loc[date_ix-1:date_ix+2]
            valid_model_days = [pd.Timestamp(day) for day in valid_model_days]
            self.valid_model_periods.update({date : 
                                        valid_model_days})
        
    
    def get_data_for_normalization(self):
        this_date_ix = self.valid_market_days.loc[
            self.valid_market_days == self.trading_date].index[0]
        normalize_back_date = self.valid_market_days.loc[
            this_date_ix - self.NORMALIZE_DAYS
        ]
        last_10_days = self.ticker.history(
            start=normalize_back_date, end=self.trading_date)[
            ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.historical_mean = last_10_days.mean().values
        self.historical_std = last_10_days.std().values
        
    def normalize_data(self, price_data):
        return (price_data - self.historical_mean) / self.historical_std
        
    def validate_dates(self, user_input):
        try:
            assert user_input[EARNINGS_DATE_KEY] in self.valid_earnings_dates
        except AssertionError:
            raise InvalidInputError('Earnings date is not a valid earnings date.')
        try:
            assert (self.valid_market_days == user_input[TRADING_DATE_KEY]).sum() > 0
        except AssertionError:
            raise InvalidInputError('Trading date passed is not a valid trading date in the market.')
        try:
            assert user_input[TRADING_DATE_KEY] in self.valid_model_periods[user_input['earnings date']]
        except AssertionError:
            raise InvalidInputError(
                'Model only accepts trading dates within the 3-day period beginning on the earnings date.'
            )
            
    def validate_model_inputs(self, price_data, est_data):
        try:
            assert price_data.max() <= OBS_RANGE_HIGH and \
                price_data.min() >= OBS_RANGE_LOW
        except AssertionError:
            s = 'Pricing data does not comply with model\'s accepted '\
            + 'input range. Are you sure prices are correct? '\
            + 'The means over the previous 10 days are: \n{}'
            mean_s = ', '.join([
                '{}: {:.2f}'.format(key, value) 
                for key, value in 
                dict(zip(PRICE_KEYS, self.historical_mean)).items()])
            raise InvalidInputError(s.format(mean_s))
        try:
            assert est_data.max() <= EST_RANGE_HIGH and \
                est_data.min() >= EST_RANGE_LOW
        except AssertionError:
            s = 'Estimate data does not comply with model\'s accepted '\
            + 'input range. Are you sure prices are correct? '\
            + 'The model only accepts values in range -500, 500.'
            raise InvalidInputError(s)
            
    def trading_date_to_int(self):
        ix = [i for i, date in 
              enumerate(self.valid_model_periods[self.earnings_date])
              if date == self.trading_date][0]
        
        self.trading_date_int = ix - 2
        
    def update_handler_for_trading_date(self):
        self.get_data_for_normalization()
        self.trading_date_to_int()
    
    def build_model_inputs(self, user_input):
        try:
            self.validate_dates(user_input)
        except InvalidInputError:
            raise
            
        if user_input[EARNINGS_DATE_KEY] != self.earnings_date:
            self.earnings_date = user_input[EARNINGS_DATE_KEY]
        if user_input[TRADING_DATE_KEY] != self.trading_date:
            self.trading_date = user_input[TRADING_DATE_KEY]
            self.update_handler_for_trading_date()
                
        price_data = [user_input[key] for key in PRICE_KEYS]
        price_data = np.array(price_data, dtype=np.float32)
        normalized_price_data = self.normalize_data(price_data)
        
        est_data = [user_input[key] for key in EST_KEYS]
        est_data = np.array(est_data)
        
        self.validate_model_inputs(normalized_price_data, est_data)
        
        trading_date_int = np.array([
            self.trading_date_int], dtype=np.float32)
        
        normalized_price_data = np.concatenate([normalized_price_data, 
                                                trading_date_int])
        
        return price_data, normalized_price_data, est_data

class TradingServer():
    SEED = 1
    SELECT_ENV = 'marketenv-v0'
    MAX_AVAIL_ACTIONS = 1000
    MAX_SHARES = 1000
    ACTION_EMBEDDING_DIM = 50
    ACTION_EMBEDDING_RANGE_LOW = -1
    ACTION_EMBEDDING_RANGE_HIGH = 1

    
    
    def __init__(self):
        self.symbol = None
        self.earnings_date = None
        
        self.np_random, seed = seeding.np_random(self.SEED)
        
        ray.shutdown()
        ray.init(
            ignore_reinit_error=True, 
            num_gpus=0, 
            num_cpus=1
        )
        
        register_env(self.SELECT_ENV, lambda config: 
                     MarketEnv_v0({}))
        ModelCatalog.register_custom_model(
            'autoregressive_model',
            AutoregressiveParametricTradingModel)
        ModelCatalog.register_custom_action_dist(
            'multinomial_autoreg_dist', 
            TorchMultinomialAutoregressiveDistribution)        
                
        with open('config.json', 'r') as f:
            ppo_config = json.load(f)
        keys_to_del = []
        for key, val in ppo_config.items():
            if 'class' in str(val):
                keys_to_del.append(key)
        for key in keys_to_del:
            del ppo_config[key]
        
        ppo_config['env'] = MarketEnv_v0
        ppo_config['num_gpus'] = 0
        ppo_config['num_workers'] = 1
        ppo_config['num_envs_per_worker'] = 1
#        ppo_config['explore'] = False
        
        self.agent = ppo.PPOTrainer(
            ppo_config,
            env=self.SELECT_ENV
        )
#        test_dir = 'ray_results/final_model/PPO_MarketEnv_v0_959f6_00000_0_2021-03-03_17-26-03/'

        self.agent.restore('checkpoint/checkpoint-150')
        
        self.buying_embeddings = self.np_random.rand(
            self.MAX_AVAIL_ACTIONS,
            self.ACTION_EMBEDDING_DIM
        )
        self.buying_embeddings = np.clip(
            self.buying_embeddings,
            self.ACTION_EMBEDDING_RANGE_LOW,
            self.ACTION_EMBEDDING_RANGE_HIGH
        )
        self.selling_embeddings = self.np_random.rand(
            self.MAX_AVAIL_ACTIONS,
            self.ACTION_EMBEDDING_DIM,
        )
        self.selling_embeddings = np.clip(
            self.selling_embeddings,
            self.ACTION_EMBEDDING_RANGE_LOW,
            self.ACTION_EMBEDDING_RANGE_LOW
        )
        self.holding_embeddings = np.zeros((
            self.MAX_AVAIL_ACTIONS,
            self.ACTION_EMBEDDING_DIM
        ))
        self.action_embeddings = np.stack(
            [self.buying_embeddings, 
             self.selling_embeddings, 
             self.holding_embeddings]
        )                      

    def get_action_from_model(self, est_data, action_state):
        obs =  {
            'action_type_mask' : action_state['action_type_mask'],
            'action_mask' : action_state['action_mask'],
            'action_embeddings' : self.action_embeddings,
            'symbol_id' : self.symbol_id,
            'cash_record' : action_state['cash_record'],
            'n_shares_record' : action_state['n_shares_record'],
            'a1_record' : action_state['a1_record'],
            'a2_record' : action_state['a2_record'],
            'price' : action_state['price'],
            'estimate' : est_data
        }      
        
    
        state = [np.zeros((256), np.float32),
         np.zeros((256), np.float32)]
                
        actions, state, logits = self.agent.compute_action(obs, state)
        
        return actions    
    
    
    def process_user_input(self, user_input):
        #TODO:
        # Improve user input date handling
        
        symbol = user_input[SYMBOL_KEY]
        user_input[EARNINGS_DATE_KEY] = pd.Timestamp(user_input[EARNINGS_DATE_KEY], tz='UTC')
        user_input[TRADING_DATE_KEY] = pd.Timestamp(user_input[TRADING_DATE_KEY], tz='UTC')        

        if symbol != self.symbol:
            # use logging here
            self.symbol = symbol
            # 5 symbols
            self.symbol_id = np.zeros(5)
            self.symbol_id = SYMBOL_IDS[symbol]
            self.input_handler = InputDataHandler(symbol)
        user_input[EARNINGS_DATE_KEY] = pd.Timestamp(
            user_input[EARNINGS_DATE_KEY])
        earnings_date = user_input[EARNINGS_DATE_KEY]
        if self.earnings_date != earnings_date:
            # use logging here too
            self.earnings_date = earnings_date
            self.action_handler = ActionHandler()
        
        price_data, normalized_price_data, est_data = self.input_handler\
            .build_model_inputs(user_input)
        
        self.action_handler.update_values_for_new_input(price_data, normalized_price_data)
        action_state = self.action_handler.get_state()
        
        action = self.get_action_from_model(est_data, action_state)
        
        self.action_handler.update_values_for_action(action)
        
        state_for_client = {
            'action type' : int(action['a1']),
            'amount' : int(action['a2']),
            'shares held' : int(self.action_handler.n_shares),
            'position value' : round(float(self.action_handler.position_value), 2),
            'cash balance' : round(float(self.action_handler.cash_balance), 2),
            'account value' : round(float(self.action_handler.account_value), 2)
        }
                
        return state_for_client 