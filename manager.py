import time
import os
import json
from collections import defaultdict

import numpy as np
import gym
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.schedules.piecewise_schedule import PiecewiseSchedule

from market_env.envs.market_env import MarketEnv_v0
from market_env.envs.market_env import default_config as default_env_config
from model import AutoregressiveParametricTradingModel
from model import default_config as default_model_config
from action_dist import TorchMultinomialAutoregressiveDistribution

lr_schedule = [
    (0, 5e-4),
    (15000, 5e-5),
#     (30000, 5e-4),
#     (60000, 5e-5)
]



entropy_coeff_schedule = [
    (0,.06),
    (8000,.04),
#    (20000, .2),
    (15000, .015),
#    (75000, .001)

]

default_ppo_config = {
    'env' : MarketEnv_v0,
    'model' : {
        'custom_model': 'autoregressive_model',
        'custom_action_dist': 'multinomial_autoreg_dist',
        'custom_model_config' : {}
    },
    'seed' : 5,
    'clip_param' : tune.grid_search([.2,.4]),
    'gamma' : tune.grid_search([.4,.9]),
    'lr' : 5e-4,
    'lambda': .95,
    'kl_coeff' : tune.grid_search([.2, .4]),
    'kl_target' : tune.grid_search([.01, .03]),
    'framework' : 'torch',
    'entropy_coeff': tune.grid_search([.01, .03]),
    'batch_mode' : 'complete_episodes',
    'num_workers' : 0,
    'num_gpus' : 0,
}

env_layer_map = {
    'n_symbols' : 'id_dim',
    'obs_dim' : 'obs_dim',
    'est_dim' : 'est_dim',
    'max_shares' : 'n_shares_dim',
    'action_embedding_dim' : 'action_embedding_dim'
}

class ExperimentManager:
    
    
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
        
        if self.env_config['write']:
            self.env_config['output_path'] = os.path.join(self.chkpt_root, 
                                                          'results')
                
            if not os.path.exists(self.env_config['output_path']):
                os.makedirs(self.env_config['output_path'])
            
        self.model_config = default_model_config
        self.model_config.update(custom_model_config)        
        self.get_layer_config_from_env() 
    
        self.ppo_config['model']['custom_model_config'].update(
            self.model_config)
        
    def register(self):
        register_env(self.select_env, lambda config: 
                     MarketEnv_v0(self.env_config))
        ModelCatalog.register_custom_model(
            'autoregressive_model', 
            AutoregressiveParametricTradingModel)
        ModelCatalog.register_custom_action_dist(
            'multinomial_autoreg_dist', 
            TorchMultinomialAutoregressiveDistribution)

    def init_agent(self, chkpt_path=None):
        self.agent = ppo.PPOTrainer(self.ppo_config, env=self.select_env)
        if chkpt_path:
            self.agent.restore(chkpt_path)

    def train(self, n_iter=10, resume=False, restore=None, name=None):
        stop = {'training_iteration': n_iter}
        if not restore:
            tune.run('PPO', config=self.ppo_config, stop=stop, checkpoint_freq=5, local_dir='ray_results', resume=resume)
        else:
            tune.run('PPO', config=self.ppo_config, stop=stop, checkpoint_freq=5, local_dir='ray_results', restore=restore, name=name)
    
    def test(self, n_iter=10, write=True):
        """Test trained agent for a single episode. Return the episode reward"""        
        def format_reward(val):
            if val >= 0:
                return '${:.2f}'.format(val)
            else:
                return '$({:.2f})'.format(abs(val))
        
        input_path = os.path.join('processed_data', 'test')
        output_path = os.path.join('results', 'test')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        env = gym.make('marketenv-v0', custom_env_config={
            '_seed' : 1,
            'input_path' : input_path,
            'write' : write,
            'output_path' : output_path
        })
        
        state = [np.zeros((self.model_config['lstm_state_size']), np.float32),
                 np.zeros((self.model_config['lstm_state_size']), np.float32)]
        
        episode_rewards = []
        account_vals = []
        s = '({}) symbol: {} | period: {} | reward: {} | mean_reward: {} | account_value: {} | mean_account_value: {}'
        
        for i in range(n_iter):
            episode_reward = 0
            done = False
            obs = env.reset()
            
            
            while not done:
                action, state, logits = self.agent.compute_action(obs, state)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            account_vals.append(env.account_value)            
            
            print(s.format(
                i+1,
                env.current_symbol, 
                env.current_earnings_date,
                format_reward(episode_reward),
                format_reward(sum(episode_rewards) / len(episode_rewards)),
                format_reward(env.account_value),
                format_reward(sum(account_vals) / len(account_vals))                
            ))
            
def run_experiment(exp_type, trial_dir, chkpt_n=None):
    if exp_type == 'grid_search':
        ray.shutdown()
        ray.init(ignore_reinit_error=True, num_gpus=1, num_cpus=10, object_store_memory=2.5e10)
        exp_mgr = ExperimentManager()
        custom_ppo_config={
            'num_gpus' : 1,
            'num_workers' : 4,
            'num_envs_per_worker' : 5,
        }
    
        exp_mgr.build_configs(custom_ppo_config, {}, {})
        exp_mgr.register()
        exp_mgr.train(5)
        ray.shutdown()
    elif exp_type == 'search-more':
        ray.shutdown()
        ray.init(ignore_reinit_error=True, num_gpus=1, num_cpus=10, object_store_memory=2.5e10)
        trial_name = os.path.split(trial_dir)[-1]
        chkpt_path = os.path.join(trial_dir, 'checkpoint_5', 'checkpoint-5')

        restored_name = trial_name + '_RESTORED_NEW'
        params_file = os.path.join(trial_dir, 'params.json')
        with open(params_file, 'r') as f:
            original_ppo_config = json.load(f)
        # ray saves string reps of classes, which are incompatible with restoring
        # aside from the env, these are immaterial to training so delete them
        keys_to_del = []
        for key, value in original_ppo_config.items():
            if 'class' in str(value):
                keys_to_del.append(key)
        for key in keys_to_del:
            del original_ppo_config[key]
        original_ppo_config['env'] = MarketEnv_v0
        exp_mgr = ExperimentManager()
        exp_mgr.build_configs(original_ppo_config, {})
        exp_mgr.register()
        exp_mgr.train(15, restore=chkpt_path, name=restored_name)
        ray.shutdown()
    elif exp_type == 'one-off':
        ray.shutdown()
        ray.init(ignore_reinit_error=True, num_gpus=1, num_cpus=10, object_store_memory=2.5e10)
        params_file = os.path.join(trial_dir, 'params.json')
        with open(params_file, 'r') as f:
            original_ppo_config = json.load(f)
        # ray saves string reps of classes, which are incompatible with restoring
        # aside from the env, these are immaterial to training so delete them
        keys_to_del = []
        for key, value in original_ppo_config.items():
            if 'class' in str(value):
                keys_to_del.append(key)
        for key in keys_to_del:
            del original_ppo_config[key]
        original_ppo_config['env'] = MarketEnv_v0
        original_ppo_config['num_envs_per_worker'] = 2
        custom_env_config = {'_seed' : 1}
        exp_mgr = ExperimentManager()
        exp_mgr.build_configs(original_ppo_config, custom_env_config, {})
        exp_mgr.register()
        exp_mgr.train(10)
        ray.shutdown()

    elif exp_type == 'test':
        ray.shutdown()
        ray.init(ignore_reinit_error=True, num_gpus=0, num_cpus=1)
        trial_name = os.path.split(trial_dir)[-1]
        chkpt_path = os.path.join(trial_dir, 'checkpoint_{}'.format(chkpt_n), 'checkpoint-{}'.format(chkpt_n))
        params_file = os.path.join(trial_dir, 'params.json')
        with open(params_file, 'r') as f:
            original_ppo_config = json.load(f)
        # ray saves string reps of classes, which are incompatible with restoring
        # aside from the env, these are immaterial to training so delete them
        keys_to_del = []
        for key, value in original_ppo_config.items():
            if 'class' in str(value):
                keys_to_del.append(key)
        for key in keys_to_del:
            del original_ppo_config[key]
        original_ppo_config['env'] = MarketEnv_v0
        original_ppo_config['num_gpus'] = 0
        original_ppo_config['num_workers'] = 1
        original_ppo_config['num_envs_per_worker'] = 1
        custom_model_config = {'exploration' : False}
        exp_mgr = ExperimentManager()
        exp_mgr.build_configs(original_ppo_config, {}, custom_model_config)
        exp_mgr.register()
        exp_mgr.init_agent(chkpt_path)
        n_files = len(os.listdir('processed_data/test'))
        exp_mgr.test(n_iter=n_files)
        ray.shutdown()
    else:
        raise ValueError('Invalid experiment type "{}"'.format(exp_type))            
            
