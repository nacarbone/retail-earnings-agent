import ray
from train import TrainingHelper
    
if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    training_helper = TrainingHelper(
        custom_ppo_config={'num_gpus' : 1},
#         custom_train_config={'restore' : True,
#                             'chkpt_root' : 'ray_results/2021-02-16_13-20-56'}
     )
    training_helper.train(500)