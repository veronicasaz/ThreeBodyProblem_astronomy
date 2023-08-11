import gym 
import numpy as np
import random
from IPython.display import clear_output


# """
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# """

# Environment
env = gym.make('cluster_2D:Cluster2D-v0') # example of how to create the env once it's been registered


env.reset() # Resets the environment and returns a random initial state

# Actions basic code
epochs = 0
penalties, reward = 0, 0
done = False
while not done:
    action = env.action_space.sample()
    state, rewards, done, info = env.step(action) # Returns observation/ reward/ done/ info

    epochs += 1

print("END", rewards, penalties)


# Training Q-learning
# Initialize the Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# # state
# state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index))
# print("State", state)
# env.s = state # asign state to environment
# env.render() # plot

# # Reward table is a dict
# env.P[328] # print the reward matrix at current point for each of the actions

# # Train
# q_table = np.zeros([env.observation_space.n, env.action_space.n])

# # Hyperparameters
# alpha = 0.1
# gamma = 0.6
# epsilon = 0.1

# # For plotting metrics
# all_epochs = []
# all_penalties = []


# # TRAINING
# for i in range(1, 100001):
#     state = env.reset()

#     epochs, penalties, reward, = 0, 0, 0
#     done = False
    
#     while not done:
#         if random.uniform(0, 1) < epsilon: 
#             action = env.action_space.sample() # Explore action space
#         else:
#             action = np.argmax(q_table[state]) # Exploit learned values

#         next_state, reward, done, info = env.step(action) 
        
#         old_value = q_table[state, action]
#         next_max = np.max(q_table[next_state])
        
#         new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#         q_table[state, action] = new_value 

#         if reward == -10:
#             penalties += 1

#         state = next_state
#         epochs += 1
        
#     if i % 100 == 0:
#         clear_output(wait=True)
#         print(f"Episode: {i}")

# print("Training finished.\n")



