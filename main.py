# https://stackoverflow.com/questions/28208949/log-stack-trace-for-python-warning/29496228
import os

import ray
from train import ExperimentManager

# import traceback
# import warnings
# import sys

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback
# warnings.simplefilter("always")

# if __name__ == '__main__':
#     ray.shutdown()
#     ray.init(ignore_reinit_error=True)
#     exp_mgr = ExperimentManager(
#         custom_ppo_config={'num_gpus' : 1, 'lr' : .0001},
#         custom_env_config={'write' : True}
#     )
#     exp_mgr.register()
#     exp_mgr.init_agent()    
#     exp_mgr.train(100)

if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    exp_mgr = ExperimentManager(
        custom_train_config = {
            'restore' : True,
            'chkpt_root' : 'ray_results/2021-02-18_18-28-06'
        }
    )
    
    n_files = len(os.listdir('processed_data/test'))
    
    exp_mgr.override_config('model_config', {'exploration' : False})
    exp_mgr.override_config('ppo_config', {'num_gpus' : 0})
    exp_mgr.register()
    exp_mgr.init_agent(24)
    exp_mgr.test(n_iter=n_files)
 

