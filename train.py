import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from market_env.envs.market_env import MarketEnv_v0
from model import AutoregressiveParametricTradingModel
from action_dist import TorchMultinomialAutoregressiveDistribution

#TODO
# (1) CHANGE HARD-CODED PARAMETERS TO COMMAND LINE ARGUMENTS

ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

config['seed'] = 1

config['env'] = MarketEnv_v0,
config['model'] = {
        'custom_model': 'autoregressive_model',
        'custom_action_dist': "binary_autoreg_dist",
        'custom_model_config' : {}
    }

config['gamma'] = .5 # discount of future reward
config['lr'] = .01 # this could be selected using the tune API
config['kl_coeff'] = 1.0 # action distribution exploration tendency
config['kl_target'] = 0.01


# Currently, EpsilonGreedy exploration isn't working with the model
# Need to debug, but this is how it would it would be implemented at a high level:

# config['exploration_config'] = {
#        "type": "EpsilonGreedy",
#    # Parameters for the Exploration class' constructor:
# #        "initial_epsilon": 1.0,
# #        "final_epsilon": 0,
# #        "epsilon_timesteps": 20000
#         # Add constructor kwargs here (if any).
#     }

config['framework'] = 'torch'

# this means the policy is updated on sequences of 50, which isn't exactly ideal
# (would be better to update after a complete episode) but the number can
# be increased on a different computer with more RAM
config['rollout_fragment_length'] = 50
#config['batch_mode'] = 'complete_episodes'
config['sgd_minibatch_size'] = 8
config['train_batch_size'] = 16

config['num_workers'] = 2

select_env = 'marketenv-v0'
register_env(select_env, lambda config: MarketEnv_v0({}))

ModelCatalog.register_custom_model("autoregressive_model", AutoregressiveParametricTradingModel)
ModelCatalog.register_custom_action_dist("binary_autoreg_dist", TorchMultinomialAutoregressiveDistribution)

agent = ppo.PPOTrainer(config, env=select_env)

status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 1

chkpt_root = 'ray_results/'

for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)

    # these will print nan until an env's done flag = True
    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))