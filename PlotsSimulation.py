"""
PlotsSimulation: plotting functions

Author: Veronica Saz Ulibarrena
Last modified: 6-February-2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from TrainingFunctions import DQN, load_reward, plot_reward
from Cluster.cluster_2D.envs.HermiteIntegration_env import IntegrateEnv_Hermite

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def run_trajectory(seed = 123, action = 'RL', env = None, 
                   name_suffix = None, steps = None,
                   reward_f = None, model_path = None, steps_suffix= ''):
    """
    run_trajectory: Run one initialization with RL or with an integrator
    INPUTS:
        seed: seed to be used for initial conditions. 
        action: fixed action or 'RL' 
        env: environment to simulate
        name_suffix: suffix to be added for the file saving
        steps: number of steps to simulate
        reward_f: type of reward to use for the simulation and weights for the 3 terms
        model_path: path to the trained RL algorithm
        steps_suffix: suffix for the file with the steps taken

    OUTPUTS:
        reward: reward for each step
    """
    if env == None:
        env = IntegrateEnv_Hermite()

    env.suffix = name_suffix
    if steps == None:
        steps = env.settings['Integration']['max_steps']
    
    if reward_f != None:
        reward_type = reward_f[0] # overload weights
        env.W = reward_f[1:] # overload weights
    else:
        reward_type = env.settings['Training']['reward_f']
    
    if model_path == None:
        model_path = env.settings['Training']['savemodel'] +'model_weights.pth'
    state, info = env.reset(seed = seed, steps = steps, typereward = reward_type)

    reward = np.zeros(steps)
    i = 0

    # Case 1: use trained RL algorithm
    if action == 'RL':
         # Load trained policy network
        n_actions = env.action_space.n
        n_observations = len(state)
        model = DQN(n_observations, n_actions, settings = env.settings) # we do not specify ``weights``, i.e. create untrained model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        steps_taken = list()
        steps_taken.append(0) # initial conditions

        # Take steps
        while i < steps-1:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            steps_taken.append(action.item())
            state, y, terminated, info = env.step(action.item())
            reward[i] = env.reward
            i += 1
        env.close()
        np.save(env.settings['Integration']['savefile'] + 'RL_steps_taken'+steps_suffix, np.array(steps_taken))
    
    # Case 2: random actions 
    elif action == 'random':
        while i < steps-1:
            action = np.random.randint(0, len(env.actions))
            if type(action) == int:
                x, y, terminated, zz = env.step(action)
                reward[i] = env.reward
            else:
                x, y, terminated, zz = env.step(action[i%len(action)])
                reward[i] = env.reward
            i += 1
        env.close()
    # Case 3: fixed action throughout the simulation
    else:
        while i < steps-1:
            if type(action) == int:
                x, y, terminated, zz = env.step(action)
                reward[i] = env.reward
            else:
                x, y, terminated, zz = env.step(action[i%len(action)])
                reward[i] = env.reward
            i += 1
        env.close()
    return reward

def load_state_files(env, steps, namefile = None):
    """
    load_state_files: Load run information 
    INPUTS: 
        env: environment of the saved files
        steps: steps taken
        namefile: suffix for the loading 
    OUTPUTS:
        state: state of the bodies in the system
        cons: action, energy error, angular momentum error
        tcomp: computation time
    """
    env.suffix = (namefile)
    state = env.loadstate()[0][0:steps, :, :]
    cons = env.loadstate()[1][0:steps, :]
    tcomp = env.loadstate()[2][0:steps]

    return state, cons, tcomp

def plot_planets_trajectory(ax, state, name_planets, labelsize = 15, steps = 30, legend_on = True):
    """
    plot_planets_trajectory: plot trajectory of three bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
        legend_on: True or False to display the legend
    """
    n_planets = np.shape(state)[1]
    for j in range(n_planets):
        x = state[0:steps, j, 2]
        y = state[0:steps, j, 3]
        m = state[0, j, 1]
        size_marker = np.log(m)/30

        ax.scatter(x[0], y[0], s = 20*size_marker,\
                   c = colors[j%len(colors)], \
                    label = "Particle "+ name_planets[j])
        ax.plot(x[1:], y[1:], marker = None, 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[j%len(colors)], \
                    alpha = 0.1)
        
        ax.scatter(x[1:], y[1:], s = size_marker, \
                    c = colors[j%len(colors)])        
        
    if legend_on == True:
        ax.legend(fontsize = labelsize)
    ax.set_xlabel('x (au)', fontsize = labelsize)
    ax.set_ylabel('y (au)', fontsize = labelsize)
    
def plot_planets_distance(ax, x_axis, state, name_planets, labelsize = 12, steps = 30):
    """
    plot_planets_distance: plot steps vs pairwise-distance of the bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
    """
    n_planets = np.shape(state)[1]
    Dist = []
    Labels = []
    for i in range(n_planets):
        r1 = state[0:steps, i, 2:5]
        m = state[0, i, 1]
        for j in range(i+1, n_planets):
            r2 = state[0:steps, j, 2:5]
            Dist.append(np.linalg.norm(r2-r1, axis = 1))
            Labels.append('Particle %i-%i'%(i, j))
        
        size_marker = np.log(m)/30
    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i], linewidth = 2.5)
    ax.legend(fontsize =labelsize, framealpha = 0.5)
    ax.set_yscale('log')
    return Dist


def plot_actions_taken(ax, x_axis, y_axis):
    """
    plot_actions_taken: plot steps vs actions taken by the RL algorithm
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
    """
    colors = colors2[0]
    ax.plot(x_axis, y_axis, color = colors, linestyle = '-', alpha = 0.5,
            marker = '.', markersize = 8)
    ax.grid(axis='y')

def plot_evolution(ax, x_axis, y_axis, label = None, color = None, 
                   colorindex = None, linestyle = None, linewidth = 1):
    """
    plot_evolution: plot steps vs another measurement
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
        label: label for the legend of each data line
        color: color selection for each line
        colorindex: index of the general color selection
        linestyle: type of line
        linewidth: matplotlib parameter for the width of the line
    """
    if colorindex != None:
        color = colors[(colorindex+3)%len(colors)] # start in the blues
    ax.plot(x_axis, y_axis, color = color, linestyle = linestyle, label = label, 
            linewidth = linewidth)
