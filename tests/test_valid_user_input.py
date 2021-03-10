import unittest

import pandas as pd

from ppo_earnings_trader.server import InputDataHandler, InvalidInputError

GOOD_INPUTS = {
    'symbol': 'WMT', 
    'earnings date': pd.Timestamp('2021-02-18', tz='UTC'), 
    'trading date': pd.Timestamp('2021-02-19', tz='UTC'), 
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

BAD_INPUTS = {
    'symbol': '123',
    'earnings date': pd.Timestamp('2021-02-12', tz='UTC'), 
    'trading date': pd.Timestamp('2021-02-25', tz='UTC'), 
    'open': 1e10,
    'high': 1e10, 
    'low': 1e10, 
    'close': 1e10, 
    'volume': 1e20, 
    'minimum estimate': 1e10, 
    'maximum estimate': 1e10, 
    'mean estimate': 1e10,
    'median estimate': 1e10, 
    'estimate standard deviation': 1e10, 
    'actual': 1e10
}


def test_symbol_validation_on_creation():
    try:
        input_handler = InputDataHandler(BAD_INPUTS['symbol'])
        assert False
    except InvalidInputError:
        assert True
        
def test_input_validation():
    input_handler = InputDataHandler(GOOD_INPUTS['symbol'])
    for key, value in BAD_INPUTS.items():
        this_iter_input = GOOD_INPUTS.copy()
        this_iter_input[key] = value
        try:
            input_handler.build_model_inputs(this_iter_input)
            assert False
        except InvalidInputError:
            assert True