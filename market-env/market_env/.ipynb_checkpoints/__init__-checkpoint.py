from gym.envs.registration import register

register(
    id='testenv-v0',
    entry_point='gym_test.envs:TestEnv_v0',
)
