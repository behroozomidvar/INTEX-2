from gym.envs.registration import register

register(
    id='intex-env-v1',
    entry_point='intex_env.envs:intex_env',
)