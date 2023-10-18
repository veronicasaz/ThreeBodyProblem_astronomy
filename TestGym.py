import numpy as np
import random
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt

import gym
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count

from helpfunctions import load_json
# """
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/https://www.gymlibrary.dev/content/environment_creation/
# https://www.gymlibrary.dev/content/environment_creation/
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# """

# Environment
# env = gym.make('cluster_2D:Cluster2D-v0') # example of how to create the env once it's been registered
env = gym.make('cluster_2D:SympleInt-v0') # example of how to create the env once it's been registered
env.reset() # Resets the environment and returns a random initial state

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

################################################################################
# # Actions basic code
# epochs = 0
# penalties, reward = 0, 0
# done = False
# while not done:
#     action = env.action_space.sample()
#     state, rewards, done, info = env.step(action) # Returns observation/ reward/ done/ info

#     epochs += 1

# print("END", rewards, penalties)
################################################################################
settings = load_json("./settings_symple.json")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, settings['Training']['neurons'])
        self.layer2 = nn.Linear(settings['Training']['neurons'], settings['Training']['neurons'])
        self.layer3 = nn.Linear(settings['Training']['neurons'], n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        for i in range(settings['Training']['hidden_layers']):
            x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# TRAINING
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = settings['Training']['lr']

# Get number of actions from gym action space
n_actions = env.action_space.n
# env.settings['Integration']['seed'] = None # random seed
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_rewards = []

def plot_durations(episode_rewards, episode, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    plt.yscale('symlog', linthresh = 1e-10)
    if episode %50 == 0:
        plt.savefig('./SympleIntegration_training/reward_progress_%i'%episode)
    # plt.close()

    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
        # means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # means = torch.cat((torch.zeros(99), means))
        # plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50


# Training loop
for i_episode in range(settings['Training']['max_iter']):
    print("Training episode: %i"%i_episode)
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in range(settings['Integration']['max_steps']):
        action = select_action(state)
        observation, reward_p, terminated, info = env.step(action.item())
        reward = torch.tensor([reward_p], device=device)
        done = terminated 

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        print(reward)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(reward_p)
            plot_durations(episode_rewards, i_episode)
            break
    env.close()

print('Complete')
plot_durations(episode_rewards, i_episode, show_result=True)
plt.ioff()
plt.show()
# for i in range(1, settings['Training']['max_iter']):
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



