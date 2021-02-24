import os
import re

from manager import run_experiment

with open('selections.txt', 'r') as f:
    selections = [int(x.replace('\n', '')) for x in f.readlines()]

trials_dir = os.path.join('ray_results', 'PPO')
trial_names = sorted([x for x in os.listdir(trials_dir) if 'PPO_MarketEnv_v0' in x])

for selection in selections:
    patn = '_[0-9]{5}_([0-9]{1,3})_'
    expr = re.compile(patn)
    
    i = 0
    
    selection_match = False
    while not selection_match:
        n_part = re.search(expr, trial_names[i])
        if n_part:
            n_part = int(n_part.group(1))
            if selection == n_part:
                selection_match = True
                break
            i += 1
        
    trial_name = trial_names[i]
    trial_dir = os.path.join(trials_dir, trial_name)
    run_experiment('single', trial_dir)

