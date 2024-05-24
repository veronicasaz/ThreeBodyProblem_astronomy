"""
TestGym: Training of the reinforcement learning algorithm

Author: Veronica Saz Ulibarrena
Last modified: 6-February-2024

Based on:
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/https://www.gymlibrary.dev/content/environment_creation/
https://www.gymlibrary.dev/content/environment_creation/
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
"""
from IPython.display import clear_output

import gym
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
import random
import math

from collections import namedtuple, deque


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

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, neurons, layers):
        """
        DQN: creation of the networks
        INPUTS:
            n_observations: number of observations to use as input
            n_actions: number of actions to use as output size
            settings: dictionary with specific network settings
        """
        super(DQN, self).__init__()
        self.neurons = int(neurons)
        
        self.layer1 = nn.Linear(n_observations, self.neurons)
        self.layer2 = nn.Linear(self.neurons, self.neurons)
        self.layer3 = nn.Linear(self.neurons, n_actions)

        if layers == None:
            self.hidden_layers = self.settings['Training']['hidden_layers']
        else:
            self.hidden_layers = int(layers)


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
        for i in range(self.hidden_layers):
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


# def load_test_dataset(env):
#     state = np.load(env.settings['Training']['savemodel']  + 'TestDataset.npy')
#     return state

def test_network(env, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_samples = env.settings['Training']['testdata_size']
    Reward = np.zeros((test_samples, 3)) # R, E, Tcomp
    for testcase in range(test_samples): # for each test case
        env.settings['InitialConditions']['seed'] = testcase
        state, info = env.reset()

        tcomp = 0
        rew = 0
        terminated = False
        steps = 0
        while terminated == False: #steps 
            steps += 1
            # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            state, reward, terminated, info = env.step(action.item())

            tcomp += info['tcomp']
            rew += reward

        env.close()
            
        Reward[testcase, 0] = rew/steps # cumulative reward normalized by the number of steps reached
        Reward[testcase, 1] = info['Energy_error'] # last energy error
        Reward[testcase, 2] = tcomp/steps  # normalized by the number of steps reached
    
    return Reward.flatten()

def train_net(env = None, suffix = ''):
    # Environment
    if env == None:
        # env = gym.make('bridgedparticles:ThreeBody-v0') # create the env once it's been registered
        import env
        env = gym.make('TBP_env-v0')

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # settings = load_json("./settings_symple.json")
    
    # TRAINING settings
    BATCH_SIZE = env.settings['Training']['batch_size'] # number of transitions sampled from the replay buffer
    GAMMA = env.settings['Training']['gamma'] # discount factor
    EPS_START = env.settings['Training']['eps_start'] # starting value of epsilon
    EPS_END = env.settings['Training']['eps_end'] # final value of epsilon
    EPS_DECAY = env.settings['Training']['eps_decay'] # controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = env.settings['Training']['tau'] # update rate of the target network
    LR = env.settings['Training']['lr'] # learning rate of the ``AdamW`` optimizer
    NEURONS = env.settings['Training']['neurons']
    LAYERS = env.settings['Training']['hidden_layers']
    env.settings['Integration']['savestate'] = False
    env.settings['Training']['display'] = False

    # test_dataset = load_test_dataset(env)

    # Get number of actions from gym action space
    n_actions = env.action_space.n # TODO: test

    # Get the number of state observations
    n_observations = env.observation_space_n # TODO: test

    # Create nets
    policy_net = DQN(n_observations, n_actions, NEURONS, LAYERS).to(device)
    target_net = DQN(n_observations, n_actions, NEURONS, LAYERS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000, Transition = Transition)

    
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    episode_number = 0 # counter of the number of steps

    # lists to save training progress
    save_reward = list()
    save_EnergyE = list()
    save_huberloss = list()
    save_tcomp = list()
    save_test_reward = list()

    # Training loop
    while episode_number <= env.settings['Training']['max_episodes']:
        print("Training episode: %i/%i"%(episode_number, env.settings['Training']['max_episodes']))

        # Initialize the environment and get it's state
        env.settings['InitialConditions']['seed'] = np.random.randint(1000) # make initial conditions vary
        state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        save_reward_list = list()
        save_EnergyE_list = list()
        save_tcomp_list = list()
        save_huberloss_list = list()

        # Do first step without updating the networks and with the best step
        action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, 0)
        observation, reward_p, terminated, info = env.step(action.item())

        terminated = False
        while terminated == False:
            # Take a step
            action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, steps_done)
            observation, reward_p, terminated, info = env.step(action.item())
            save_reward_list.append(reward_p)
            save_EnergyE_list.append(info['Energy_error'])
            save_tcomp_list.append(info['tcomp'])
            
            reward = torch.tensor([reward_p], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            Huber_loss = optimize_model(policy_net, target_net, memory, \
                    Transition, device, GAMMA, BATCH_SIZE,\
                    optimizer)
            
            if Huber_loss == None:
                save_huberloss_list.append(0)
            else:
                save_huberloss_list.append(Huber_loss.item())

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        env.close()

        test_reward_list = test_network(env, policy_net)

        save_reward.append(save_reward_list)
        save_EnergyE.append(save_EnergyE_list)
        save_huberloss.append(save_huberloss_list)
        save_tcomp.append(save_tcomp_list)
        save_test_reward.append(test_reward_list)
        
        if episode_number %10 == 0:
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights'+str(episode_number)+'.pth') # save model
        else:
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights.pth') # save model

        # save training
        with open(env.settings['Training']['savemodel']+suffix+"rewards.txt", "w") as f:
            for ss in save_reward:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"EnergyError.txt", "w") as f:
            for ss in save_EnergyE:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"Tcomp.txt", "w") as f:
            for ss in save_tcomp:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"HuberLoss.txt", "w") as f:
            for ss in save_huberloss:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"TestReward.txt", "w") as f:
            for ss in save_test_reward:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        episode_number += 1

    env.close()
    print('Complete')

def train_net_pretrained(model_path, env = None, suffix = ''):
    # Environment
    if env == None:
        # env = gym.make('bridgedparticles:ThreeBody-v0') # create the env once it's been registered
        import env
        env = gym.make('TBP_env-v0')

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # settings = load_json("./settings_symple.json")
    
    # TRAINING settings
    BATCH_SIZE = env.settings['Training']['batch_size'] # number of transitions sampled from the replay buffer
    GAMMA = env.settings['Training']['gamma'] # discount factor
    EPS_START = env.settings['Training']['eps_start'] # starting value of epsilon
    EPS_END = env.settings['Training']['eps_end'] # final value of epsilon
    EPS_DECAY = env.settings['Training']['eps_decay'] # controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = env.settings['Training']['tau'] # update rate of the target network
    LR = env.settings['Training']['lr'] # learning rate of the ``AdamW`` optimizer
    NEURONS = env.settings['Training']['neurons']
    LAYERS = env.settings['Training']['hidden_layers']
    env.settings['Integration']['savestate'] = False
    env.settings['Training']['display'] = False

    # test_dataset = load_test_dataset(env)

    # Get number of actions from gym action space
    n_actions = env.action_space.n # TODO: test

    # Get the number of state observations
    n_observations = env.observation_space_n # TODO: test

    # Create nets
    policy_net = DQN(n_observations, n_actions, NEURONS, LAYERS).to(device)
    target_net = DQN(n_observations, n_actions, NEURONS, LAYERS).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    target_net.load_state_dict(policy_net.state_dict())

    Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000, Transition = Transition)

    
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    episode_number = 0 # counter of the number of steps

    # lists to save training progress
    save_reward = list()
    save_EnergyE = list()
    save_huberloss = list()
    save_tcomp = list()
    save_test_reward = list()
    max_reward = -10000 # small number

    # Training loop
    while episode_number <= env.settings['Training']['max_episodes']:
        print("Training episode: %i/%i"%(episode_number, env.settings['Training']['max_episodes']))

        # Initialize the environment and get it's state
        env.settings['InitialConditions']['seed'] = np.random.randint(1000) # make initial conditions vary
        state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        save_reward_list = list()
        save_EnergyE_list = list()
        save_tcomp_list = list()
        save_huberloss_list = list()

        # Do first step without updating the networks and with the best step
        action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, 0)
        observation, reward_p, terminated, info = env.step(action.item())

        terminated = False
        while terminated == False:
            # Take a step
            action, steps_done = select_action(state, policy_net, [EPS_START, EPS_END, EPS_DECAY], env, device, steps_done)
            observation, reward_p, terminated, info = env.step(action.item())
            save_reward_list.append(reward_p)
            save_EnergyE_list.append(info['Energy_error'])
            save_tcomp_list.append(info['tcomp'])
            
            reward = torch.tensor([reward_p], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            Huber_loss = optimize_model(policy_net, target_net, memory, \
                    Transition, device, GAMMA, BATCH_SIZE,\
                    optimizer)
            
            if Huber_loss == None:
                save_huberloss_list.append(0)
            else:
                save_huberloss_list.append(Huber_loss.item())

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        env.close()

        test_reward_list = test_network(env, policy_net)

        save_reward.append(save_reward_list)
        save_EnergyE.append(save_EnergyE_list)
        save_huberloss.append(save_huberloss_list)
        save_tcomp.append(save_tcomp_list)
        save_test_reward.append(test_reward_list)
        
        if episode_number %10 == 0:
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights'+str(episode_number)+'.pth') # save model
        else:
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights.pth') # save model

        # Save model if reward is max
        avg_reward = np.mean(save_reward_list)
        print("reward", episode_number, avg_reward, max_reward, (max_reward*(1 -abs(max_reward)/max_reward*0.3)))

        if avg_reward >= (max_reward*(1 -abs(max_reward)/max_reward*0.3)): # save all models with high rewards
            torch.save(policy_net.state_dict(), env.settings['Training']['savemodel'] +suffix+ 'model_weights'+str(episode_number)+'.pth') # save model
            max_reward = avg_reward

        # save training
        with open(env.settings['Training']['savemodel']+suffix+"rewards.txt", "w") as f:
            for ss in save_reward:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"EnergyError.txt", "w") as f:
            for ss in save_EnergyE:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"Tcomp.txt", "w") as f:
            for ss in save_tcomp:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"HuberLoss.txt", "w") as f:
            for ss in save_huberloss:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        with open(env.settings['Training']['savemodel']+suffix+"TestReward.txt", "w") as f:
            for ss in save_test_reward:
                for s in ss:
                    f.write(str(s) +" ")
                f.write("\n")

        episode_number += 1

    env.close()
    print('Complete')

if __name__ == '__main__':
    train_net()