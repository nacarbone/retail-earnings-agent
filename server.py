import logging
import os
import json

import ray
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from gym.utils import seeding
from yahoo_earnings_calendar import YahooEarningsCalendar

from market_env.envs.market_env import MarketEnv_v0
from model import AutoregressiveParametricTradingModel
from action_dist import TorchMultinomialAutoregressiveDistribution

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

CHKPT_PATH = os.path.join('checkpoint', 'checkpoint')

class InvalidInputError(Exception):
    """Custom exception used in validation of user input"""


    def __init__(self, message):
        """
        Parameters
        ---
        message : str
            Message to be returned when error is raised
        """
        self.message = message

class ActionHandler():
    """
    Handles actions given by the model and updates user's state (e.g. cash
    balance, number of shares etc.). This essentially mimics the environment
    on which the model was trained, but splits its step function into
    updating for actions and for new observations.

    Attributes
    ---
    cash_balance : float
        The user's current cash balance
    account_value : float
        The current total account value of the user
    n_shares : int
        The current number of shares held by the user
    position_value : float
        The current value of the user's position
    price_roll : numpy.array
        An of the 15 most recent prices passed by the user
    a1_record : numpy.array
        An array of the 15 most recent buy, sell or hold actions taken by the
        model
    a2_record : numpy.array
        An array of the 15 most recent amount actions taken by the model        
    n_shares_record : numpy.array
        An array of the 15 most recent amounts of shares held by the user
    cash_record : numpy.array
        An array of the 15 most recent cash balances for the user    
    shares_avail_to_buy : int
        The number of shares a user can purchase based on the cash_balance
        and observed price
    shares_avail_to_sell : numpy.array
        The number of shares a user can sell based on n_shares
    buying_action_mask : numpy.array
        A mask passed to the model for invalid buying amounts
    selling_action_mask : numpy.array
        A mask passed to the model for invalid selling amounts
    holding_action_mask : numpy.array
        A mask passed to the model for invalid selling amounts (i.e. any
        non-zero quantities)
    action_mask : numpy.array
        The concatenated buying, selling and holding masks

    Methods
    ---
    update_avail_actions : None
        Updates the action masks for a new price passed by the user and the
        user's current state
    update_values_for_action : None
        Updates cash_balance, n_shares and position_value for the action taken
        by the model
    update_values_for_new_input : None
        Updates position_value and account_value and available actions for new
        prices passed by user
    get_state : dict
        Gets the user's current state (e.g. cash balance, number of shares, 
        etc.)
    """
    START_BALANCE = 10000
    MAX_AVAIL_ACTIONS = 1000
    MAX_SHARES = 1000
    OBS_DIM = 6
    SEQ_LEN = 15

    
    def __init__(self):
        self.cash_balance = self.START_BALANCE
        self.account_value = self.START_BALANCE
        self.n_shares = 0
        self.position_value = 0

        self.price_roll = np.zeros((self.SEQ_LEN, self.OBS_DIM))        
        self.a1_record = np.zeros((self.SEQ_LEN, 3))
        self.a1_record[:,-1] = 1
        self.a2_record = np.zeros((self.SEQ_LEN, self.MAX_AVAIL_ACTIONS))
        self.a2_record[:,0] = 1
        self.n_shares_record = np.zeros((self.SEQ_LEN, self.MAX_SHARES))
        self.n_shares_record[:,0] = 1
        self.cash_record = np.zeros((self.SEQ_LEN, 1))
        self.cash_record.fill(1)

    def update_avail_actions(self):
        """
        Updates the action masks for a new price passed by the user and the
        user's current state        
        """
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

    def update_values_for_action(self, action: dict):
        """
        Updates cash_balance, n_shares, position_value and records for the 
        action taken by the model
        """
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

    def update_values_for_new_input(self, 
                                    new_price_data: 'np.ndarray', 
                                    normalized_new_price_data):
        """
        Updates position_value, account_value, price_roll and available
        actions for new prices passed by user
        """
        self.price_roll = np.roll(self.price_roll, -1, axis=0)
        self.price_roll[-1] = normalized_new_price_data

        self.current_price = sum(new_price_data[:4]) / 4
        self.position_value = self.current_price * self.n_shares 
        self.account_value = self.cash_balance + self.position_value
        self.update_avail_actions()

    def get_state(self):
        """
        Gets the user's current state (e.g. cash balance, number of shares, 
        etc.) to be passed to the model

        Returns
        ---
        A dict of the user's current state
        """
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
    """
    The handler to preprocess and validate user input.

    Attributes
    ---
    calendar : pandas.DataFrame
        The NYSE market calendar
    valid_market_days : pandas.Series
        The market dates from calendar
    symbol : str
        The symbol passed by the user
    valid_earnings_date : list
        A list of pandas.Timestamp object representing valid earnings dates
        for the symbol
    valid_model_periods : dict
        A dict mapping earnings dates to lists of valid model trading dates
        (+- 1 day from earnings date)
    ticker : yfinance.Ticker
        A Ticker object to facilitate getting historical data for 
        normalization
    historical_mean : pandas.Series
        The OHCLV mean for the 10 trading days prior to the trading date 
        passed by the user; not set until first user input and will be reset
        if user passes new trading date
    historical_mean : pandas.Series
        The OHCLV standard deviation for the 10 trading days prior to the 
        trading date passed by the user; not set until first user input is
        passed will be reset if user passes new trading date
    trading_date_int : int in [-1,1]
        The integer value representing the trading date 

    Methods
    ---
    get_data_for_normalization : None
        Sets class attributes historical_mean and historical_std for use in
        the normalize_data class method
    normalize_data : numpy.array
        Normalizes the OHLCV data from user input using 10-day historical mean
        and standard deviation
    validate_dates : None
        Valdidate that the earnings date is valid for the symbol and the
        trading date is valid for the model
    validate_model_inputs : None
        Validates that the OHLCV and estimate data complies with model's
        accepted input range
    trading_date_to_int : None
        Converts the trading date passed by the user to an integer in
        [-1, 1] based on offset from the earnings date
    update_handler_for_trading_date : None
        Resets class attributes for new trading dates
    build_model_inputs : tuple
        Unpacks and processes user inputs using class methods
    """
    VALID_START = pd.Timestamp('2003-01-01', tz='UTC')
    # change to today
    VALID_END = pd.Timestamp('2022-12-31', tz='UTC')
    NYSE = mcal.get_calendar('NYSE')

    NORMALIZE_DAYS = 10

    def __init__(self, symbol: str):
        """
        Parameters
        ---
        symbol : str in {'AMZN', 'COST', 'KR', 'WBA', 'WMT'}
            The symbol whose input will be handled by the class
        """
        self.calendar = self.NYSE.schedule(
            start_date=self.VALID_START, end_date=self.VALID_END)
        self.valid_market_days = self.calendar['market_open']\
            .dt.normalize().reset_index(drop=True)

        self.symbol = symbol
        self.earnings_date = None
        self.trading_date = None        

        self.ticker = yf.Ticker(symbol)

        yec = YahooEarningsCalendar()
        earnings_data = yec.get_earnings_of(symbol)
        valid_earnings_dates = [pd.Timestamp(
            date['startdatetime']
        ).normalize() for date in earnings_data]
        self.valid_earnings_dates = [
            date for date in valid_earnings_dates 
            if date > self.VALID_START and date < self.VALID_END]        

        self.valid_model_periods = {}

        for date in self.valid_earnings_dates:
            date_ix = self.valid_market_days.loc[
                self.valid_market_days == date].index[0]
            # valid model days back
            valid_model_days = self.valid_market_days.loc[date_ix-1:date_ix+2]
            valid_model_days = [pd.Timestamp(day) for day in valid_model_days]
            self.valid_model_periods.update({date : 
                                        valid_model_days})

    def get_data_for_normalization(self):
        """
        Sets class attributes historical_mean and historical_std for use in
        the normalize_data class method
        """
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

    def normalize_data(self, price_data: 'numpy.ndarray'):
        """
        Normalizes the OHLCV data from user input using 10-day historical mean
        and standard deviation.

        Parameters
        ---
        price_data : numpy.array
            The OHLCV data passed by the user
        """
        return (price_data - self.historical_mean) / self.historical_std

    def validate_dates(self, user_input: dict):
        """
        Valdidates that the earnings date is valid for the symbol and the
        trading date is valid for the model. Raises an InvalidInputError
        otherwise.
        
        Parameters
        ---
        user_input : dict
            The user input with dates converted to pandas.Timestamp objects
        """
        try:
            assert user_input[EARNINGS_DATE_KEY] in self.valid_earnings_dates
        except AssertionError:
            s - 'Earnings date is not a valid earnings date.'
            raise InvalidInputError(s)
        try:
            assert (self.valid_market_days == \
                    user_input[TRADING_DATE_KEY]).sum() > 0
        except AssertionError:
            s = 'Trading date passed is not a valid trading date in the '\
            + 'market.'
            raise InvalidInputError()
        try:
            assert user_input[TRADING_DATE_KEY] in \
                self.valid_model_periods[user_input['earnings date']]
        except AssertionError:
            s = 'Model only accepts trading dates within the 3-day period '\
            + 'beginning on the earnings date.'
            raise InvalidInputError(s)

    def validate_model_inputs(self, 
                              price_data: 'numpy.ndarray', 
                              est_data: 'numpy.ndarray'):
        """
        Validates that the OHLCV and estimate data complies with model's
        accepted input range. Raises an InvalidInputError otherwise.
        
        Parameters
        ---
        price_data : numpy.array
            The normalized OHLCV data
        est_data : numpy.array
            The estimate data passed by the user
        """
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
        """
        Converts the trading date passed by the user to an integer in
        [-1, 1] based on offset from the earnings date. Sets a class attribute
        for the value.
        """
        ix = [i for i, date in 
              enumerate(self.valid_model_periods[self.earnings_date])
              if date == self.trading_date][0]

        self.trading_date_int = ix - 2

    def update_handler_for_trading_date(self):
        """
        Resets class attributes for new trading dates.
        """
        self.get_data_for_normalization()
        self.trading_date_to_int()

    def build_model_inputs(self, user_input: dict):
        """
        Unpacks and processes user inputs using class methods

        Parameters
        ---
        user_input : dict
            the user input with dates converted to pandas.Timestamp objects
            
        Returns
        ---
        A tuple of numpy arrays containing the original OHLCV data, the 
        normalized OHLCV data, estimate data
        """
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
    """
    Main object for handling user input and trading actions. 

    Attributes
    ---
    symbol : str
        The current symbol based on user input will be reset if user passes
        a new symbol
    symbol_id : int
        The current symbol id based on user input; not set until first user
        input is passed and will be reset if user passes a new symbol
    earnings_date : 
        The current earnings date based on user input
    np_random : numpy.random object
        A seeded random object from gym.seeding
    agent : ray.trainable
        The model to draw actions from; restored from a previous checkpoint
    buying_embeddings : numpy.ndarray
        The embeddings used for buying actions
    selling_embeddings : numpy.ndarray
        The embeddings used for selling actions
    holding_embeddings : numpy.ndarray
        The embeddings used for holding actions; array of zeroes
    action_embeddings : numpy.ndarray
        Concatenation of buying, selling and holding embeddings
        to be passed through the model
    input_handler : class InputHandler
        The handler to preprocess and validate user input; not set until first
        user input is passed and will be re-instantiated if user
        passes a new symbol
    action_handler : class ActionHandler
        The handler to process actions from the model and update user's state
        (e.g. cash balance, number of shares etc.) accordingly; will be
        reset if user passes new symbol-earnings date combination

    Methods
    ---
    get_action_from_model : dict
        Returns actions from model based on user input and user's state (e.g. 
        cash balance, number of shares etc.)
    process_user_input : dict
        Unpacks user input and sends through through complete model/
        environment pipeline
    """
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
            num_cpus=1,
            log_to_driver=False,
            logging_level=logging.FATAL
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
        # Ray periodically changes its configuration keys; any that are
        # specific to this model should be stable, so we'll simply
        # remove these from the config, but this should be noted
        for key, val in ppo_config.items():
            if 'class' in str(val) or key not in list(
                ppo.DEFAULT_CONFIG.keys()):
                keys_to_del.append(key)
        for key in keys_to_del:
            del ppo_config[key]

        ppo_config['env'] = MarketEnv_v0
        ppo_config['num_gpus'] = 0
        ppo_config['num_workers'] = 1
        ppo_config['num_envs_per_worker'] = 1
        ppo_config['log_level'] = 'ERROR'

        self.agent = ppo.PPOTrainer(
            ppo_config,
            env=self.SELECT_ENV
        )
        
        self.agent.restore(CHKPT_PATH)
        
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

    def get_action_from_model(self, 
                              est_data: 'numpy.ndarray', 
                              action_state: dict):
        """
        Returns actions from model based on user input and user's state (e.g. 
        cash balance, number of shares etc.)
        
        Parameters
        ---
        est_data : np.ndarray
            The estimate data passed by the user
        action_state : dict
            A dict of numpy arrays representing the user's current state
            
        Returns
        ---
        A dict of with values of type int for a1 (buy, sell or hold) and 
        a2 (amount)
        """
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

    def process_user_input(self, user_input: dict):
        """
        Unpacks user input and sends through through complete model/
        environment pipeline        
        
        Parameters
        ---
        user_input : dict
            The raw input from the user; see input-spec.json for details on
            expected contents

        Returns
        ---
        A dict containing the action taken by the model and the user's state
        (e.g. cash balance, number of shares etc.)        
        """
        #TODO:
        # (1) Improve user input date handling
        for key in user_input.keys():
            try:
                assert key in INPUT_KEYS
            except AssertionError:
                s = '{} is not a valid input key!'.format(key)
                raise InvalidInputError(s)
        
        symbol = user_input[SYMBOL_KEY]
        
        try:
            assert symbol in list(SYMBOL_IDS.keys())
        except AssertionError:
            s = '{} is not a valid symbol. Please choose from the following: {}.'
            str_symbols = ', '.join(list(SYMBOL_IDS.keys()))
            s = s.format(symbol, str_symbols)
            raise InvalidInputError(s)
        
        user_input[EARNINGS_DATE_KEY] = pd.Timestamp(
            user_input[EARNINGS_DATE_KEY], tz='UTC')
        user_input[TRADING_DATE_KEY] = pd.Timestamp(
            user_input[TRADING_DATE_KEY], tz='UTC')

        if symbol != self.symbol:
            # use logging here
            self.symbol = symbol
            self.symbol_id = np.zeros(len(SYMBOL_IDS))
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

        self.action_handler.update_values_for_new_input(
            price_data, normalized_price_data)
        action_state = self.action_handler.get_state()

        action = self.get_action_from_model(est_data, action_state)

        self.action_handler.update_values_for_action(action)

        state_for_client = {
            'action type' : int(action['a1']),
            'amount' : int(action['a2']),
            'shares held' : int(self.action_handler.n_shares),
            'position value' : round(
                float(self.action_handler.position_value), 2),
            'cash balance' : round(
                float(self.action_handler.cash_balance), 2),
            'account value' : round(
                float(self.action_handler.account_value), 2)
        }

        return state_for_client
