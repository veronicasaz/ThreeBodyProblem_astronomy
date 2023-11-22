import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch

from TrainingFunctions import DQN, load_reward, plot_reward
from Cluster.cluster_2D.envs.SympleIntegration_env import IntegrateEnv


# colors = ['orange', 'green', 'blue', 'red', 'grey', 'black']
colors = ['darkgoldenrod', 'mediumseagreen', 'coral', 'indianred', \
        'navy', 'deepskyblue', 'steelblue', 'mediumslateblue']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def run_trajectory(seed = 123, action = 'RL', env = None, name_suffix = None, steps = None):
    """
    Run one initialization with RL or with an integrator
    """
    if env == None:
        env = IntegrateEnv()
    env.suffix = name_suffix
    state, info = env.reset(seed = seed)

    if steps == None:
        steps = env.settings['Integration']['max_steps']
    
    i = 0
    if action == 'RL':
         # Load trained policy network
        n_actions = env.action_space.n
        n_observations = len(state)
        model = DQN(n_observations, n_actions, settings = env.settings) # we do not specify ``weights``, i.e. create untrained model
        model.load_state_dict(torch.load(env.settings['Training']['savemodel'] +'model_weights.pth'))
        model.eval()
        
        steps_taken = list()
        
        while i < steps:
            print(i)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            steps_taken.append(action.item())
            state, reward_p, terminated, info = env.step(action.item())
            i += 1
        env.close()
        np.save(env.settings['Integration']['savefile'] + 'RL_steps_taken', np.array(steps_taken))
        return steps_taken
    else:
        while i < steps:
            if type(action) == int:
                x, y, terminated, zz = env.step(action)
            else:
                x, y, terminated, zz = env.step(action[i%len(action)])
            i += 1
        env.close()

def load_state_files(env, steps, namefile = None):
    # Load run information for symple cases
    env.suffix = (namefile)
    state = env.loadstate()[0][0:steps, :, :]
    cons = env.loadstate()[1][0:steps, :]
    tcomp = env.loadstate()[2][0:steps]

    return state, cons, tcomp

def plot_planets_trajectory(ax, state, name_planets, labelsize = 20, steps = 30):
    n_planets = np.shape(state)[1]
    for j in range(n_planets):
        x = state[0:steps, j, 2]
        y = state[0:steps, j, 3]
        m = state[0, j, 1]
        size_marker = np.log(m)/30

        ax.plot(x[1:], y[1:], marker = None, 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[j%len(colors)], \
                    alpha = 0.1)
        
        ax.scatter(x[1:], y[1:], s = size_marker, \
                    c = colors[j%len(colors)], \
                    marker = markers[j//len(colors)],\
                    label = name_planets[j])
        
    ax.set_xlabel('x (m)', fontsize = labelsize)
    ax.set_xlabel('y (m)', fontsize = labelsize)
    
def plot_actions_taken(ax, x_axis, y_axis, label = None):
    colors = colors2[0]
    ax.plot(x_axis, y_axis, color = colors, marker = '.', linestyle = ':', alpha = 0.5)

def plot_evolution(ax, x_axis, y_axis, label = None, color = None, 
                   colorindex = None, linestyle = None):
    if colorindex != None:
        color = colors[colorindex+3] # start in the blues
    ax.plot(x_axis, y_axis, color = color, linestyle = linestyle, label = label)


# if __name__ == '__main__':
#     cases = 9
#     action = 5
#     steps = 1000
#     runs_trajectory(cases, action, steps)
#     plot_runs_trajectory(cases, action, steps)
