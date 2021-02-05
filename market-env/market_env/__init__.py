from gym.envs.registration import register

register(
    id='marketenv-v0',
    entry_point='market_env.envs:MarketEnv_v0',
)
