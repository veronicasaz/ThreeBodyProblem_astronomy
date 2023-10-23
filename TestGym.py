from IPython.display import clear_output

import gym
import torch

import matplotlib
import torch.optim as optim

from collections import namedtuple, deque
from itertools import count

from helpfunctions import load_json
from TrainingFunctions import DQN, \
                            ReplayMemory,\
                            select_action, \
                            optimize_model,\
                            plot_durations
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

settings = load_json("./settings_symple.json")

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# TRAINING
BATCH_SIZE = settings['Training']['batch_size'] # number of transitions sampled from the replay buffer
GAMMA = settings['Training']['gamma'] # discount factor
EPS_START = settings['Training']['eps_start'] # starting value of epsilon
EPS_END = settings['Training']['eps_end'] # final value of epsilon
EPS_DECAY = settings['Training']['eps_decay'] # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = settings['Training']['tau'] # update rate of the target network
LR = settings['Training']['lr'] # learning rate of the ``AdamW`` optimizer

# Get number of actions from gym action space
n_actions = env.action_space.n
# env.settings['Integration']['seed'] = None # random seed

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions, settings = settings).to(device)
target_net = DQN(n_observations, n_actions, settings = settings).to(device)
target_net.load_state_dict(policy_net.state_dict())

Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward'))

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000, Transition = Transition)

steps_done = 0
episode_rewards = []

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
        action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, steps_done)
        observation, reward_p, terminated, info = env.step(action.item())
        # reward += reward_p # TODO: is this correct? otherwise rewards are not related
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
        optimize_model(policy_net, target_net, memory, \
                   Transition, device, GAMMA, BATCH_SIZE,\
                   optimizer)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        episode_rewards.append(reward_p)
        if done:
            
            plot_durations(episode_rewards, i_episode)
            break
    env.close()
    
    torch.save(policy_net.state_dict(), settings['Training']['savemodel'] + 'model_weights.pth')

print('Complete')
plot_durations(episode_rewards, i_episode, show_result=True)
plt.ioff()
plt.show()
