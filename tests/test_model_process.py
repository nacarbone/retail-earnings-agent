import os
import sys

import pandas as pd

from ppo_earnings_trader.server import TradingServer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AutoregressiveParametricTradingModel as input_model


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

N_ITER = 10

def test_model_can_process_input():
    server = TradingServer(input_model) 
    for _ in range(N_ITER):
        server.process_user_input(TEST_INPUT.copy())
    assert True