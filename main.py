import os

import ray
from manager import ExperimentManager

if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_gpus=1, num_cpus=10, object_store_memory=2.5e10)
    exp_mgr = ExperimentManager()
    custom_ppo_config={'num_gpus' : 1, 
                       'num_workers' : 2,
#                       'lr' : .0001,
#                           'num_gpus_per_worker' : (1 - .2) / 4,
#                        'num_envs_per_worker' : 5,
#                           'memory_per_worker' : 1*10**9,
#                       'remote_worker_envs' : True,
                      }
    
    
    exp_mgr.build_configs(custom_ppo_config, {}, {})
    exp_mgr.register()
    exp_mgr.train(200)
    ray.shutdown()

# if __name__ == '__main__':
#     ray.shutdown()
#     ray.init(ignore_reinit_error=True)
    
#     exp_mgr = ExperimentManager(
#         custom_train_config = {
#             'restore' : True,
#             'chkpt_root' : 'checkpoints'
#         }
#     )
# #    exp_mgr.init_agent(300)
#     n_files = len(os.listdir('processed_data/test'))
#     exp_mgr.build_configs({},{},{})
#     exp_mgr.override_config('model_config', {'exploration' : False})
#     exp_mgr.override_config('ppo_config', {'num_gpus' : 0})
#     exp_mgr.override_config('env_config', {'shuffle_files' : False})
#     exp_mgr.register()
#     exp_mgr.init_agent(150)
#     exp_mgr.test(n_iter=n_files)
 

