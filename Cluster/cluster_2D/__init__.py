from gym.envs.registration import register

register(
    id='Cluster2D-v0', # 
    entry_point='cluster_2D.envs:ClusterEnv',
    max_episode_steps=30,
)
