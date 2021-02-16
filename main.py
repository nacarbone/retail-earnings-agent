import ray
from train import TrainingHelper
    
if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    training_helper = TrainingHelper(
        custom_ppo_config={'num_gpus' : 1},
    )
    training_helper.train(10)