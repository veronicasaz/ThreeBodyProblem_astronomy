import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch

from TrainingFunctions import DQN, load_reward, plot_reward
# from Cluster.cluster_2D.envs.SympleIntegration_env import IntegrateEnv
from Cluster.cluster_2D.envs.HermiteIntegration_env import IntegrateEnv_Hermite


# colors = ['orange', 'green', 'blue', 'red', 'grey', 'black']
colors = ['darkgoldenrod', 'mediumseagreen', 'coral', 'indianred', \
        'navy', 'deepskyblue', 'steelblue', 'mediumslateblue']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def run_trajectory(seed = 123, action = 'RL', env = None, 
                   name_suffix = None, steps = None,
                   reward_f = None):
    """
    Run one initialization with RL or with an integrator
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
    state, info = env.reset(seed = seed, steps = steps, typereward = reward_type)

    reward = np.zeros(steps)
    i = 0
    if action == 'RL':
         # Load trained policy network
        n_actions = env.action_space.n
        n_observations = len(state)
        model = DQN(n_observations, n_actions, settings = env.settings) # we do not specify ``weights``, i.e. create untrained model
        model.load_state_dict(torch.load(env.settings['Training']['savemodel'] +'model_weights.pth'))
        model.eval()
        
        steps_taken = list()
        steps_taken.append(0) # initial conditions
        
        while i < steps-1:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state).max(1)[1].view(1, 1)
            steps_taken.append(action.item())
            state, y, terminated, info = env.step(action.item())
            reward[i] = env.reward
            i += 1
        env.close()
        np.save(env.settings['Integration']['savefile'] + 'RL_steps_taken', np.array(steps_taken))
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
    # Load run information for symple cases
    env.suffix = (namefile)
    state = env.loadstate()[0][0:steps, :, :]
    cons = env.loadstate()[1][0:steps, :]
    tcomp = env.loadstate()[2][0:steps]

    return state, cons, tcomp

def plot_planets_trajectory(ax, state, name_planets, labelsize = 15, steps = 30):
    n_planets = np.shape(state)[1]
    for j in range(n_planets):
        x = state[0:steps, j, 2]
        y = state[0:steps, j, 3]
        m = state[0, j, 1]
        size_marker = np.log(m)/30

        ax.scatter(x[0], y[0], s = 20*size_marker,\
                   c = colors[j%len(colors)], \
                    label = name_planets[j])
        ax.plot(x[1:], y[1:], marker = None, 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[j%len(colors)], \
                    alpha = 0.1)
        
        ax.scatter(x[1:], y[1:], s = size_marker, \
                    c = colors[j%len(colors)])
                    # marker = markers[j//len(colors)],)
        
    ax.legend()
    ax.set_xlabel('x (m)', fontsize = labelsize)
    ax.set_ylabel('y (m)', fontsize = labelsize)
    
def plot_planets_distance(ax, x_axis, state, name_planets, labelsize = 15, steps = 30):
    n_planets = np.shape(state)[1]
    Dist = []
    Labels = []
    for i in range(n_planets):
        r1 = state[0:steps, i, 2:5]
        m = state[0, i, 1]
        for j in range(i+1, n_planets):
            r2 = state[0:steps, j, 2:5]
            Dist.append(np.linalg.norm(r2-r1, axis = 1))
            Labels.append('%i-%i'%(i, j))
        
        size_marker = np.log(m)/30
    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i])
    ax.legend()


def plot_actions_taken(ax, x_axis, y_axis, label = None):
    colors = colors2[0]
    ax.plot(x_axis, y_axis, color = colors, marker = '.', linestyle = ':', alpha = 0.5)

def plot_evolution(ax, x_axis, y_axis, label = None, color = None, 
                   colorindex = None, linestyle = None):
    if colorindex != None:
        color = colors[(colorindex+3)%len(colors)] # start in the blues
    ax.plot(x_axis, y_axis, color = color, linestyle = linestyle, label = label)


# if __name__ == '__main__':
#     cases = 9
#     action = 5
#     steps = 1000
#     runs_trajectory(cases, action, steps)
#     plot_runs_trajectory(cases, action, steps)
