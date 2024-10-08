from gym.envs.registration import register


register(
    id='TBP_env-v0', # 
    entry_point='env.ThreeBP_env:ThreeBodyProblem_env',
    max_episode_steps=1000,
)
