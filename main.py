import ray
from train import TrainingHelper
    
if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    training_helper = TrainingHelper(
        custom_ppo_config={'num_gpus' : 1, 'lr' : .001},
        custom_train_config={'restore' : True,
                            'chkpt_root' : 'ray_results/2021-02-17_12-44-47'}
     )
    training_helper.train(50)