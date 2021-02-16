import time
import os
import json
from collections import defaultdict

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from market_env.envs.market_env import MarketEnv_v0
from market_env.envs.market_env import default_config as default_env_config
from model import AutoregressiveParametricTradingModel
from model import default_config as default_model_config
from action_dist import TorchMultinomialAutoregressiveDistribution

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

env_layer_map = {
    'obs_dim' : 'obs_dim',
    'est_dim' : 'est_dim',
    'max_shares' : 'position_dim'
}

class TrainingHelper:
    
    
    def __init__(self, 
                 custom_ppo_config={}, 
                 custom_train_config={},
                 custom_env_config={},
                 custom_model_config={}
                ):

        self.train_config = defaultdict(lambda: None)
        self.train_config.update(custom_train_config)

        self.select_env = 'marketenv-v0'

        if not self.train_config['chkpt_root']:
            self.chkpt_root = 'ray_results/{}/'.format(
                time.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self.chkpt_root = self.train_config['chkpt_root']
        
        if self.train_config['restore']:
            self.load_configs()
        else:
            os.mkdir(self.chkpt_root)
            self.build_configs(
                custom_ppo_config,
                custom_env_config,
                custom_model_config
            )
        
        self.register()
        self.init_agent()
        
    def get_layer_config_from_env(self):
        for env_key, model_key in env_layer_map.items():
            self.model_config[model_key] = self.env_config[env_key]
    
    def build_configs(
        self,
        custom_ppo_config,
        custom_env_config,
        custom_model_config
    ):
        
        self.ppo_config = ppo.DEFAULT_CONFIG.copy()
        self.ppo_config.update(default_ppo_config)
        self.ppo_config.update(custom_ppo_config)        
        
        self.env_config = default_env_config
        self.env_config.update(custom_env_config)

        self.model_config = default_model_config
        self.model_config.update(custom_model_config)        
        self.get_layer_config_from_env() 
    
        self.ppo_config['model']['custom_model_config'].update(
            self.model_config)
        
        self.save_configs(custom_ppo_config)
    
    def save_configs(self, custom_ppo_config):
        configs = {
            'ppo' : custom_ppo_config,
            'model' : dict(self.model_config),
            'env' : dict(self.env_config)
        }

        with open(self.chkpt_root + 'configs.json', 'w') as f:
            json.dump(configs, f)

    def load_configs(self):
        with open(os.path.join(self.chkpt_root, 'configs.json'), 'r') as f:
            configs = json.load(f)
        
        self.ppo_config = ppo.DEFAULT_CONFIG.copy()
        self.ppo_config.update(default_ppo_config)
        self.ppo_config.update(configs['ppo'])
        self.model_config = configs['model']
        self.ppo_config['model']['custom_model_config'].update(
            self.model_config)        
        self.env_config = configs['env']
        
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
            max_chkpt = max([int(chkpt.split('_')[-1]) for chkpt
                             in os.listdir(os.path.join(self.chkpt_root, 'checkpoints'))])
            max_chkpt_str = 'checkpoint_{}/checkpoint-{}'.format(
                max_chkpt, max_chkpt)
            self.agent.restore(os.path.join(self.chkpt_root, 'checkpoints', max_chkpt_str))

    def train(self, n_iter=10):
        status = "{} -- {:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

        for n in range(n_iter):
            result = self.agent.train()
            chkpt_file = self.agent.save(os.path.join(self.chkpt_root, 'checkpoints'))

            print(status.format(
                    time.strftime('%H:%M:%S'),
                    n + 1,
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"],
                    chkpt_file
                    ))