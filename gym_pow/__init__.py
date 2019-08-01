from gym.envs.registration import register

register(id='pow-v0',
entry_point='gym_pow.envs:PoWEnv',
)