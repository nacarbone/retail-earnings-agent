import os
import sys
import pdb

import numpy as np
import pandas as pd

from ppo_earnings_trader.server import TradingServer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AutoregressiveParametricTradingModel as input_model

np.random.seed(1)

TEST_INPUT = {
    'symbol': 'WMT', 
    'earnings date': '2021-02-18', 
    'trading date': '2021-02-19', 
    'open': 101, 
    'high': 102, 
    'low': 99, 
    'close': 101, 
    'volume': 10000, 
    'minimum estimate': 1.7, 
    'maximum estimate': 1.78, 
    'mean estimate': 1.72, 
    'median estimate': 1.74, 
    'estimate standard deviation': 0.25, 
    'actual': 1.75
}

NUM_KEYS = list(TEST_INPUT.keys())[3:]
OFFSET_MAGNITUDE = .1
N_ITER = 1000

def test_model_produces_valid_output():
    server = TradingServer(input_model)
    
    for _ in range(N_ITER):
        for KEY in NUM_KEYS:
            offset = np.random.rand() * OFFSET_MAGNITUDE * TEST_INPUT[KEY]
            offset = np.random.choice([-offset, offset])
            TEST_INPUT[KEY] += offset
        state = server.process_user_input(TEST_INPUT.copy())
        a1, a2 = state['action type'], state['amount']
        shares_avail_to_buy = server.action_handler.shares_avail_to_buy
        shares_avail_to_sell = server.action_handler.shares_avail_to_sell   
    
        if a1 == 0:
            assert a2 <= shares_avail_to_buy
        elif a1 == 1:
            assert a2 <= shares_avail_to_sell
        else:
            assert a2 == 0

    assert True
    