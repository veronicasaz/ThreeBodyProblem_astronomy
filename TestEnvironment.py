"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 31-May-2024
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import torch

from env.ThreeBP_env import ThreeBodyProblem_env
from TrainRL import DQN

from Plots_TestEnvironment import plot_initializations, plot_rewards_multiple

def run_trajectory(env, action = 'RL', model_path = None):
    """
    run_trajectory: Run one initialization with RL or with an integrator
    INPUTS:
        env: environment to simulate
        action: fixed action or 'RL' 
        model_path: path to the trained RL algorithm
    """
    if model_path == None:
        model_path = env.settings['Training']['savemodel'] +'model_weights.pth'
        
    state, info = env.reset()
    i = 0
    terminated = False

    # Case 1: use trained RL algorithm
    if action == 'RL':
         # Load trained policy network
        n_actions = env.action_space.n
        n_observations = env.observation_space_n
        neurons = env.settings['Training']['neurons']
        layers = env.settings['Training']['hidden_layers']
        
        model = DQN(n_observations, n_actions, neurons, layers)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Take steps
        while terminated == False:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            state, y, terminated, info = env.step(action.item())
        env.close()
    
    elif action == 'random':
        while terminated == False:
            action_i = np.random.randint(len(env.actions), size = 1)
            x, y, terminated, zz = env.step(action_i)
            i += 1
        env.close()
    # Case 3: fixed action throughout the simulation
    else:
        while terminated == False:
            x, y, terminated, zz = env.step(action)
            i += 1
        env.close()


def load_state_files(env, namefile = None):
    """
    load_state_files: Load run information 
    INPUTS: 
        env: environment of the saved files
        namefile: suffix for the loading 
    OUTPUTS:
        state: state of the bodies in the system
        cons: action, energy error, angular momentum error
        tcomp: computation time
    """
    env.suffix = (namefile)
    state = env.loadstate()[0]
    cons = env.loadstate()[1]
    tcomp = env.loadstate()[2]

    return state, cons, tcomp

# def calculate_rewards(env, cons, W):
#     """
#     calculate_rewards: calculate reward 
#     """
#     steps = np.shape(cons)[1]
#     cases = np.shape(W)[0]
#     R = np.zeros(cases, steps)
#     for j in range(cases):
#         env.settings['RL']['reward_f'] = W[j, 0]
#         # env.settings['RL']['weights'] = W[j, 1:]
#         for i in range(1, steps):
#             R[j, i] = env._calculate_reward(cons[i, 2], cons[i-1, 2], 0, cons[i, 0], W[j, 1:])
#     return R

if __name__ == '__main__':
    experiment = 1 # number of the experiment to be run
    seed = 1
            
    if experiment == 0: #  plot trajectory with hermite for a fixed action, many initializations
        cases = 6
        action = 0
        steps = 100
        seeds = np.arange(cases)
        subfolder = '1_Initializations/'

        # Run results
        NAMES = []
        for i in range(cases):
            print(i, seeds[i])
            name = '_traj_action_%i_initialization_%i'%(action, i)
            NAMES.append(name)

            env = ThreeBodyProblem_env()
            env.settings['InitialConditions']['seed'] = seeds[i]
            env.settings['Integration']['subfolder'] = subfolder
            env.settings['Integration']['suffix'] = name
            run_trajectory(env, action = 0)

        # Load results
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = subfolder
        STATE = []
        CONS = [] 
        TCOMP = []
        for i in range(cases):
            env.settings['Integration']['suffix'] = NAMES[i]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + subfolder + 'Initializations.png'
        plot_initializations(STATE, CONS, TCOMP, NAMES, save_path, seeds)
    

    elif experiment == 1: # multiple reward functions, plot function shape
        initializations = 30
        seeds = np.arange(initializations)
        subfolder = "2_RewardStudy/"
        reward_functions = [
                [5, 3000.0, 0.0, 4.0],
                [5, 1000.0, 0.0, 4.0],
                [5, 5000.0, 0, 4.0],
                [3, 1.0, 0.0, 4.0],
                [3, 100.0, 0.0, 4.0]
                ]
        
        # Run 
        NAMES = []
        env = ThreeBodyProblem_env()
        for r_i in range(len(reward_functions)):
            for i in range(initializations): # random actions
                print(seeds[i])
                name = 'R_%i_traj_initialization_%i'%(r_i, i)
                NAMES.append(name)

                env.settings['InitialConditions']['seed'] = seeds[i]
                env.settings['Integration']['subfolder'] = subfolder
                env.settings['Integration']['suffix'] = name
                env.settings['RL']['reward_f'] = reward_functions[r_i][0]
                env.settings['RL']['weights'] = reward_functions[r_i][1:]
                env._initialize_RL()
                # run_trajectory(env, action = 'random')

            for i in range(initializations): # same action per initialization
                print(initializations + i)
                name = 'R_%i_traj_initialization_%i'%(r_i, initializations +i)
                NAMES.append(name)

                env.settings['InitialConditions']['seed'] = seeds[i]
                env.settings['Integration']['subfolder'] = subfolder
                env.settings['Integration']['suffix'] = name
                env.settings['RL']['reward_f'] = reward_functions[r_i][0]
                env.settings['RL']['weights'] = reward_functions[r_i][1:]
                act = np.random.randint(len(env.actions))
                print(act)
                env._initialize_RL()
                # run_trajectory(env, action = act)


        # Load results
        env.settings['Integration']['subfolder'] = subfolder
        STATE = []
        CONS = [] 
        TCOMP = []
        for i in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[i]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        # plot one reward
        save_path = env.settings['Integration']['savefile'] + subfolder + 'Rewards.png'
        plot_rewards_multiple(env, STATE, CONS, TCOMP, reward_functions, initializations*2, save_path, plot_one = True) 
        # plot many rewards
        save_path = env.settings['Integration']['savefile'] + subfolder + 'Rewards_multiple.png'
        plot_rewards_multiple(env, STATE, CONS, TCOMP, reward_functions, initializations*2, save_path, plot_one = False) 

        
   