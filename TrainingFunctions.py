"""
TrainingFunctions: additional functions for the reinforcement learning training

Author: Veronica Saz Ulibarrena
Last modified: 6-February-2024
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import savgol_filter

import random
import matplotlib
import matplotlib.pyplot as plt
import math

from scipy.signal import butter, lfilter, freqz

from collections import namedtuple, deque

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

class ReplayMemory(object):
    def __init__(self, capacity, Transition = None):
        """
        ReplayMemory: create sample database from memory
        INPUTS:
            capacity: maximum length 
            Transition: 
        """
        self.memory = deque([], maxlen=capacity)
        
        self.Transition = Transition

    def push(self, *args):
        """
        push: save a transition
        """
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        """
        sample: take a random sample from the memory
        INPUTS:
            batch_size: size of the batch to be taken
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        __len__: length of the memory array
        """
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, settings = None):
        """
        DQN: creation of the networks
        INPUTS:
            n_observations: number of observations to use as input
            n_actions: number of actions to use as output size
            settings: dictionary with specific network settings
        """
        self.settings = settings
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, self.settings['Training']['neurons'])
        self.layer2 = nn.Linear(self.settings['Training']['neurons'], self.settings['Training']['neurons'])
        self.layer3 = nn.Linear(self.settings['Training']['neurons'], n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """
        forward: add layers to the network
        INPUTS:
            x: input
        OUTPUTS:
            output
        """
        x = F.relu(self.layer1(x))
        for i in range(self.settings['Training']['hidden_layers']):
            x = F.relu(self.layer2(x))
        return self.layer3(x)
    

def select_action(state, policy_net, Eps, env, device, steps_done):
    """
    select_action: choose best action 
    INPUTS:
        state: state of the system. Input of the network
        policy_net: network with current weight values
        Eps: eps parameter
        env: environment
        device: devie
        steps_done: number of steps already done
    OUTPUTS: 
        action
    """
    EPS_START, EPS_END, EPS_DECAY = Eps
    # global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done


def plot_durations(episode_rewards, episode, show_result=False):
    """
    plot_durations: plot training progress
    INPUTS:
        episode_rewards: reward for each episode
        episode: episode number
        show_result: if True, shows plot
    https://www.linkedin.com/advice/0/how-do-you-evaluate-performance-robustness-your-reinforcement:
    cumulative reward, the average reward per episode, the number of steps per episode, or the success rate over time.
    """
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    x = np.arange(len(rewards_t.numpy()))
    plt.scatter(x, rewards_t.numpy())
    plt.yscale('symlog', linthresh = 1e-4)
    if episode %50 == 0:
        plt.savefig('./SympleIntegration_training/reward_progress_%i'%episode)


def load_reward(a):
    """
    load_reward: load rewards from file 
    INPUTS:
        a: environment
    OUTPUTS:
        score: rewards
        EnergyE: energy error
        HuberLoss: huber loss
    """
    score = []
    with open(a.settings['Training']['savemodel'] + "rewards.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            score_r = list()
            for j in y.split():
                score_r.append(float(j))
            score.append(score_r)

    EnergyE = []
    with open(a.settings['Training']['savemodel'] + "EnergyError.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            Energy_r = list()
            for j in y.split():
                Energy_r.append(float(j))
            EnergyE.append(Energy_r)

    HuberLoss = []

    return score, EnergyE, HuberLoss

def plot_reward(a, reward, Eerror, HuberLoss):
    """
    plot_reward: plot training parameters such as reward
    INPUTS:
        a: environment
        reward: rewards
        Eerror: energy error
        HuberLoss: huber loss
    """
    episodes = len(reward)
    x_episodes = np.arange(episodes)

    steps_perepisode = np.zeros(episodes)
    cumul_reward_perepisode = np.zeros(episodes)
    last_energy_perepisode = np.zeros(episodes)
    reward_flat = list()
    energyerror_flat = list()
    episode_end_list = list()
    huberloss_flat = list()

    for i in range(episodes):
        steps_perepisode[i] = len(reward[i])
        cumul_reward_perepisode[i] = sum(reward[i])
        reward_flat = reward_flat + reward[i][1:]
        try:
            last_energy_perepisode[i] = abs(Eerror[i][-1])
        except:
            last_energy_perepisode[i] = 0
        try:
            energyerror_flat = energyerror_flat + Eerror[i][1:]
        except:
            energyerror_flat = energyerror_flat + [0]
        if len(reward[i][1:])>0:
            episode_end_list = episode_end_list + [1] + [0]*(len(reward[i][1:])-1)

    x_all = np.arange(len(reward_flat))
    
    f, ax = plt.subplots(2, 1, figsize = (10,6))
    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.15, hspace = 0.2)
    fontsize = 20

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2

    pts = 41
    ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(steps_perepisode), pts, 1)    
    ax[0].plot(x_episodes, yy, color = 'black')
    ax[0].set_ylabel('Steps', fontsize = fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize-3)
    ax[0].tick_params(axis='y', labelsize=fontsize-3)

    ax[1].plot(x_episodes, cumul_reward_perepisode, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(cumul_reward_perepisode), pts, 1)
    ax[1].plot(x_episodes, yy, color = 'black')
    ax[1].set_xlabel('Episode', fontsize = fontsize)
    ax[1].set_ylabel('Cumulative reward', fontsize = fontsize)
    ax[1].set_yscale('symlog', linthresh = 1e1)
    ax[1].tick_params(axis='x', labelsize=fontsize-3)
    ax[1].tick_params(axis='y', labelsize=fontsize-3)

    line_x = [2100, 2100]
    line_y1 = [8e1, 1e2]
    line_y2 = [-1.5e3, 1e2]
    ax[0].plot(line_x, line_y1, linestyle = '--', color = 'red', alpha = 1)
    ax[1].plot(line_x, line_y2, linestyle = '--', color = 'red', alpha = 1)

    ax[0].set_xlim([-1, 3000])
    ax[1].set_xlim([-1, 3000])
    ax[0].set_ylim([line_y1[0], line_y1[1]])
    ax[1].set_ylim([line_y2[0], line_y2[1]])

    plt.savefig(a.settings['Training']['savemodel']+'_cumulative_reward.png', dpi = 100)
    plt.show()

def optimize_model(policy_net, target_net, memory, \
                   Transition, device, GAMMA, BATCH_SIZE,\
                   optimizer):
    """
    optimize_model: optimize trained model
    INPUTS: 
        policy_net: policty network
        target_net: target network
        memory: memory with all the training samples
        Transition:
        device:
        GAMMA: gamma parameter
        BATCH_SIZE: batch size
        optimizer: optimization algorithm
    OUTPUTS:
        loss: loss
    """
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

    return loss

