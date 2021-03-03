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

from market_env.envs.market_env import MarketEnv_v0
from market_env.envs.market_env import DEFAULT_CONFIG as DEFAULT_ENV_CONFIG
from model import AutoregressiveParametricTradingModel
from model import DEFAULT_CONFIG as DEFAULT_MODEL_CONFIG
from action_dist import TorchMultinomialAutoregressiveDistribution

DEFAULT_PPO_CONFIG = {
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

ENV_LAYER_MAP = {
    'n_symbols' : 'id_dim',
    'obs_dim' : 'obs_dim',
    'est_dim' : 'est_dim',
    'max_shares' : 'n_shares_dim',
    'action_embedding_dim' : 'action_embedding_dim'
}

class ExperimentManager:
    """
    Centralizes processes for training and testing agent.    
    """
    SELECT_ENV = 'marketenv-v0'
    
    def __init__(self, train_config):

#        self.select_env = 'marketenv-v0'

        if not self.train_config['chkpt_root']:
            self.chkpt_root = 'ray_results/{}/'.format(
                time.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self.chkpt_root = self.train_config['chkpt_root']
        
    def get_layer_config_from_env(self):
        """
        Ensures input/output shapes in the environment that directly 
        correspond to input/output shapes in the shapes in the model are in
        agreement.
        """
        for env_key, model_key in ENV_LAYER_MAP.items():
            self.model_config[model_key] = self.env_config[env_key]
    
    def build_configs(
        self,
        custom_ppo_config,
        custom_env_config,
        custom_model_config
    ):
        
        """
        Builds the configs that will be passed to the agent.
        
        Parameters
        ---
        custom_ppo_config : a Python dict with custom configurations for 
            RLLib's PPO algorithm;
            see https://docs.ray.io/en/master/rllib-algorithms.html for valid options
        custom_env_config : a Python dict with custom configurations for
            the MarketEnv_v0 environment; keys not explicitly used by the
            environment class will not throw an error
        custom_model_config : a Python dict with custom configurations for
            the autoregressive trading model; keys not explicitly used by the
            model class will not throw an error
            Note: this function calls the class method 
            get_layer_configs_from_env to ensure the layer dimensions of the  
            model agree with the environment input/output so keys in 
            custom_model_config may be overwritten by this method
        """
        
        self.ppo_config = ppo.DEFAULT_CONFIG.copy()
        self.ppo_config.update(DEFAULT_PPO_CONFIG)
        self.ppo_config.update(custom_ppo_config)        
        
        self.env_config = DEFAULT_ENV_CONFIG
        self.env_config.update(custom_env_config)
        
        if self.env_config['write']:
            self.env_config['output_path'] = os.path.join(
                self.chkpt_root, 
                'results'
            )
                
            if not os.path.exists(self.env_config['output_path']):
                os.makedirs(self.env_config['output_path'])
            
        self.model_config = DEFAULT_MODEL_CONFIG
        self.model_config.update(custom_model_config)        
        self.get_layer_config_from_env() 
    
        self.ppo_config['model']['custom_model_config'].update(
            self.model_config)
        
    def register(self):
        """
        Registers the environment so it will be recognized internally by RLLib.
        Should only be called after build_configs has been called; otherwise,
        an error will be thrown.
        """
        
        register_env(self.SELECT_ENV, lambda config: 
                     MarketEnv_v0(self.env_config))
        ModelCatalog.register_custom_model(
            'autoregressive_model', 
            AutoregressiveParametricTradingModel)
        ModelCatalog.register_custom_action_dist(
            'multinomial_autoreg_dist', 
            TorchMultinomialAutoregressiveDistribution)

    def init_agent(self, chkpt_path):
        """
        Initializes an agent by restoring from the provided checkpoint path.
        Generally only called for testing purposes.
        
        Parameters
        ---
        chkpt_path : the full path to the checkpoint from which to restore
        """
        self.agent = ppo.PPOTrainer(
            self.ppo_config, 
            env=self.SELECT_ENV
        )
        self.agent.restore(chkpt_path)

    def train(self, n_iter=10, name=None, chkpt_path=None):
        """
        Trains a model for a number of iterations specified by n_iter. Can be
        a completely new model, or one restored from a previous checkpoint.
        Checkpoints the model every 5 iterations.
        
        Parameters
        ---
        n_iter : the number of training iterations for the model; the exact
            number of epochs this translates to depends on the PPO config
        name : the directory name where RLLib will save the training results
            and its checkpoints; the specific training directory will be a
            sub-directory of this directory
        chkpt_path : the full path to the checkpoint from which to restore        
        """

        stop = {'training_iteration': n_iter}
        if not name:
            tune.run(
                'PPO', 
                config=self.ppo_config, 
                stop=stop, 
                checkpoint_freq=5, 
                local_dir='ray_results',
                restore=chkpt_path
            )
        else:
            tune.run(
                'PPO', 
                config=self.ppo_config, 
                stop=stop, 
                checkpoint_freq=5, 
                local_dir='ray_results',
                restore=restore,
                name=name
            )
    
    def test(self, n_iter=10, write=True):
        """
        Test trained agent for a number of iterations specified by n_iter.
        Testing data will be read from the processed_data/test folder in
        alphabetical order of filename. Also defines the function format_reward
        to neatly print model results.
        
        Parameters
        ---
        n_iter : the number of episodes to test
        write : whether to write episode data to files in a directory 
            test_results; any files that currently exist there will
            be overwritten if the model re-tests that episode
        """
        def format_reward(val):
            """
            Formats
            """
            if val >= 0:
                return '${:.2f}'.format(val)
            else:
                return '$({:.2f})'.format(abs(val))
        
        input_path = os.path.join('processed_data', 'test')
        if write:
            output_path = 'test_results'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        
        env = gym.make('marketenv-v0', custom_env_config={
            '_seed' : 1,
            'shuffle_files' : False,
            'input_path' : input_path,
            'write' : write,
            'output_path' : output_path
        })
        
        state = [np.zeros((self.model_config['lstm_state_size']), np.float32),
                 np.zeros((self.model_config['lstm_state_size']), np.float32)]
        
        episode_rewards = []
        account_vals = []
        s = '({}) symbol: {} | period: {} | reward: {} | mean reward: {} ' \
            + '| account value: {} | mean account value: {}'
        
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

def run_grid_search(
    num_gpus=1, 
    num_cpus=10, 
    max_memory=2.5e10
):
    """
    Runs a grid_search over the values in DEFAULT_PPO_CONFIG
    """
    ray.shutdown()
    ray.init(ignore_reinit_error=True, 
             num_gpus=num_gpus, 
             num_cpus=num_cpus, 
             object_store_memory=max_memory
    )
    custom_ppo_config={
        'num_gpus' : num_gpus,
        'num_workers' : min(4, num_cpus-1),
        'num_envs_per_worker' : 2,
    }

    custom_env_config = {'_seed' : 1}    
    
    exp_mgr = ExperimentManager()
    exp_mgr.build_configs(custom_ppo_config, custom_env_config, {})
    exp_mgr.register()
    exp_mgr.train(5)
    
    ray.shutdown()

def train_model(
    num_gpus=1, 
    num_cpus=10, 
    max_memory=2.5e10, 
    params_file='final-config.json',
    n_iter=100,
    restore=None
):
    """
    Trains a model using the parameters in params_file
    
    Parameters
    ---
    params_file : a full filepath to the file containing the training
        configuration; defaults to "final-config.json", which
        is expected to have been saved to the top-level directory
        following a grid search
    n_iter : the number of training iterations to run
    restore : optional, the full path to the checkpoint file from which to 
        restore a model for training
    """
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True, 
        num_gpus=num_gpus, 
        num_cpus=num_cpus, 
        object_store_memory=max_memory
    )
    
    with open(params_file, 'r') as f:
        original_ppo_config = json.load(f)
        
    original_ppo_config['num_gpus'] = num_gpus
    original_ppo_config['env'] = MarketEnv_v0
    original_ppo_config['num_envs_per_worker'] = 2
    
    custom_env_config = {'_seed' : 1}
    
    exp_mgr = ExperimentManager()
    exp_mgr.build_configs(original_ppo_config, custom_env_config, {})
    exp_mgr.register()
    exp_mgr.train(n_iter, restore=restore, name='final_model')
    
    ray.shutdown()
    
def test_trained_model(
    trial_dir, 
    chkpt_n=150
):
    """
    Tests a trained model using the testing data in processed_data/test.
    Will load model parameters based on a previously run trial specified
    by trial_dir.
    
    Parameters
    ---
    trial_dir : the full path to the directory where the trial's results
        are stored (e.g. ray_results/PPO/<RAY_TRIAL_DIRNAME>)
    chkpt_n : the checkpoint number from which to restore
    """
    
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_gpus=0, num_cpus=1)

    chkpt_path = os.path.join(
        trial_dir,
        'checkpoint_{}'.format(chkpt_n),
        'checkpoint-{}'.format(chkpt_n)
    )
    
    params_file = os.path.join(
        trial_dir,
        'params.json'
    )
    
    with open(params_file, 'r') as f:
        original_ppo_config = json.load(f)
    # ray saves string reps of classes, which are incompatible with
    # configurations to initialize new models
    # aside from the env, these are immaterial so delete them
    keys_to_del = []
    for key, val in original_ppo_config.items():
        if 'class' in str(val):
            keys_to_del.append(key)
    for key in keys_to_del:
        del original_ppo_config[key]
    original_ppo_config['env'] = MarketEnv_v0
    original_ppo_config['num_gpus'] = 0
    original_ppo_config['num_workers'] = 1
    original_ppo_config['num_envs_per_worker'] = 1
    
    custom_model_config = {'exploration' : False}

    n_files = len(os.listdir('processed_data/test'))
    
    exp_mgr = ExperimentManager()
    exp_mgr.build_configs(original_ppo_config, {}, custom_model_config)
    exp_mgr.register()
    exp_mgr.init_agent(chkpt_path)
    exp_mgr.test(n_iter=n_files)
    
    ray.shutdown()
