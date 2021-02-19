import os

import ray
from manager import ExperimentManager

if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    exp_mgr = ExperimentManager(
        custom_ppo_config={'num_gpus' : 1, 'lr' : .0001},
        custom_env_config={'write' : True}
    )
    exp_mgr.register()
    exp_mgr.init_agent()    
    exp_mgr.train(10)

# if __name__ == '__main__':
#     ray.shutdown()
#     ray.init(ignore_reinit_error=True)
#     exp_mgr = ExperimentManager(
#         custom_train_config = {
#             'restore' : True,
#             'chkpt_root' : 'ray_results/2021-02-18_18-28-06'
#         }
#     )
    
#     n_files = len(os.listdir('processed_data/test'))
    
#     exp_mgr.override_config('model_config', {'exploration' : False})
#     exp_mgr.override_config('ppo_config', {'num_gpus' : 0})
#     exp_mgr.override_config('env_config', {'shuffle_files' : False})
#     exp_mgr.register()
#     exp_mgr.init_agent(53)
#     exp_mgr.test(n_iter=n_files)
 

