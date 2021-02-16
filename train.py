import time
import os
from collections import defaultdict

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from market_env.envs.market_env import MarketEnv_v0
from model import AutoregressiveParametricTradingModel
from action_dist import TorchMultinomialAutoregressiveDistribution

#TODO
# (1) CHANGE HARD-CODED PARAMETERS TO COMMAND LINE ARGUMENTS

default_ppo_config = {
    'env' : MarketEnv_v0,
    'model' : {
        'custom_model': 'autoregressive_model',
        'custom_action_dist': 'multinomial_autoreg_dist',
        'custom_model_config' : {}
    },
    'seed' : 1,
    'gamma' : .99,
    'lr' : .001,
    'kl_coeff' : 1.0,
    'kl_target' : 0.03,
    'framework' : 'torch',
    'batch_mode' : 'complete_episodes',
    'sgd_minibatch_size' : 250,
    'train_batch_size' : 500,
    'num_workers' : 0,
    'num_gpus' : 0,
}

class TrainingHelper:
    
    
    def __init__(self, 
                 custom_ppo_config={}, 
                 custom_train_config={},
                 custom_env_config={},
                 custom_model_config={}
                ):


        self.ppo_config = ppo.DEFAULT_CONFIG.copy()
        self.ppo_config.update(default_ppo_config)
        self.ppo_config.update(custom_ppo_config)
        
        self.train_config = defaultdict(lambda: None)
        self.train_config.update(custom_train_config)

        self.env_config = defaultdict(lambda: None)
        self.env_config.update(custom_env_config)

        self.model_config = defaultdict(lambda: None)
        self.model_config.update(custom_model_config)        
        
        self.select_env = 'marketenv-v0'

        self.chkpt_root = 'ray_results/'
        
        self.register()
        self.init_agent()
        
    def register(self):
        register_env(self.select_env, lambda config: 
                     MarketEnv_v0(self.env_config))
        ModelCatalog.register_custom_model(
            'autoregressive_model', 
            AutoregressiveParametricTradingModel)
        ModelCatalog.register_custom_action_dist(
            'multinomial_autoreg_dist', 
            TorchMultinomialAutoregressiveDistribution)

    def init_agent(self):
        self.agent = ppo.PPOTrainer(self.ppo_config, env=self.select_env)
        if self.train_config['restore']:
            max_chkpt = max([int(chkpt.split('_')[1]) for chkpt
                             in os.listdir('ray_results')])
            max_chkpt_str = 'checkpoint_{}/checkpoint-{}'.format(
                max_chkpt, max_chkpt)
            self.agent.restore(self.chkpt_root + max_chkpt_str)

    def train(self, n_iter=10):
        status = "{} -- {:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

        for n in range(n_iter):
            result = self.agent.train()
            chkpt_file = self.agent.save(self.chkpt_root)

            print(status.format(
                    time.strftime('%H:%M:%S'),
                    n + 1,
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"],
                    chkpt_file
                    ))