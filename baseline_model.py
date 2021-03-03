import gym
import numpy as np
import pandas as pd

from market_env.envs.market_env import MarketEnv_v0

np.random.seed(1)

def run_one_episode(env):
    env.reset()
    sum_reward = 0
    results = []
    
    for i in range(env.MAX_STEPS - 1):
        action_type = env.action_space.sample()['buy/sell/hold']
    
        if action_type == 0:
            amount = np.random.choice(range(env.shares_avail_to_buy + 1))
        elif action_type == 1:
            amount = np.random.choice(range(env.shares_avail_to_sell + 1))
        else:
            amount = 0

        action = {'a1' : action_type, 'a2' : amount}

        state, reward, done, info = env.step(action)
        sum_reward += 0
        
        results.append([env.current_file, action_type, amount, reward, env.account_value, env.cash_balance, env.n_shares, done])
    
        if done:
            break

    return results
    
env = gym.make('marketenv-v0', config={})

all_results = {}
for i in range(100):
    results = run_one_episode(env)
    all_results[i] = results

dfs = []
for run_number, results in all_results.items():
    results_df = pd.DataFrame(results, columns=['file', 'action_type', ])
    results_df['run_number'] = run_number
    dfs.append(results_df)
df = pd.concat(dfs)

df.to_csv('baseline_model_results/results.csv')