import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json

import torch
import torchvision.models as models
import gym

from Cluster.cluster_2D.envs.SympleIntegration_env import IntegrateEnv
from TrainingFunctions import DQN, load_reward, plot_reward

def run_withRL(a):
    # env = gym.make('cluster_2D:SympleInt-v0')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, info = a.reset()

    # Load trained policy network
    n_actions = a.action_space.n
    n_observations = len(state)
    model = DQN(n_observations, n_actions, settings = a.settings) # we do not specify ``weights``, i.e. create untrained model
    model.load_state_dict(torch.load(a.settings['Training']['savemodel'] +'model_weights.pth'))
    model.eval()

    # Load environment
    a.suffix = ('_case_withRL')
    
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    i = 0
    while i < (a.settings['Integration']['max_steps']):
        action = model(state).max(1)[1].view(1, 1)
        state, reward_p, terminated, info = a.step(action.item())
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        i += 1
    a.close()

def run_many_symple_cases(a, values):
    cases = len(values)
    a = IntegrateEnv()
    for j in range(cases):
        print("Case %i"%j)
        print(values[j][0])
        print(a.actions[values[j][0]])
        a.suffix = ('_case%i'%j)
        value = values[j]
        terminated = False
        i = 0
        a.reset()
        while i < (a.settings['Integration']['max_steps']):
            x, y, terminated, zz = a.step(value[i%len(value)])
            i += 1
        a.close()
        # a.plot_orbit()

def evaluate_many_symple_cases(values, \
                        plot_tstep = True, \
                        plot_grid = True,
                        plot_errorvstime = True):
    cases = len(values)
    a = IntegrateEnv()

    # Load run information for symple cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        a.suffix = ('_case%i'%i)
        state.append(a.loadstate()[0])
        cons.append(a.loadstate()[1])
        tcomp.append(a.loadstate()[2])
        name.append('_case%i'%i)
    
    # Load RL run
    cases += 1
    a.suffix = ('_case_withRL')
    state.append(a.loadstate()[0])
    cons.append(a.loadstate()[1])
    tcomp.append(a.loadstate()[2])
    name.append('_case_withRL')

    bodies = len(state[0][0,:,0])
    steps = len(cons[0][:,0])
    # # Calculate errors in cartesian
    # baseline = state[4] # Take highest order as a baseline
    # E_x = np.zeros((steps, bodies))
    # for i in range(cases):
    #     E_x[:, i] = state[i]

    # Calculate the energy errors
    # baseline = cons[4] # Take highest order as a baseline
    E_E = np.zeros((steps, cases))
    E_M = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    for i in range(cases):
        # E_E[:, i] = abs((cons[i][:, 1] - cons[i][0, 1])/ cons[i][0, 1]) # absolute relative energy error
        # E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:])/ cons[i][0, 2:], axis = 1) # relative angular momentum error
        E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
        E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
        T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times


    # plot
    colors = ['red', 'green', 'blue', 'orange', 'grey', 'black']
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]
    markers = ['o', 'x', '.', '^', 's']
    # labels = ['Order 1', 'Order 2', 'Order 3', 'Order 5', 'Order 10', 'Mixed order']
    n_order_cases = len(a.settings['Integration']['order'])
    n_tstep_cases = len(a.settings['Integration']['t_step'])

    if plot_tstep == True:
        fig, ax = plt.subplots(3, 1, layout = 'constrained', figsize = (10, 10))
        x_axis = np.arange(0, steps, 1)

        # plot symple
        for i in range(cases-2): # plot all combinations with constant dt and order
            ax[0].plot(x_axis, E_E[:, i], color = colors[i//n_tstep_cases], linestyle = lines[i%n_tstep_cases], label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
            ax[1].plot(x_axis, E_M[:, i], color = colors[i//n_tstep_cases], linestyle = lines[i%n_tstep_cases], label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
            ax[2].plot(x_axis, T_c[:, i], color = colors[i//n_tstep_cases], linestyle = lines[i%n_tstep_cases], label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
        
        # plot Random choice
        ax[0].plot(x_axis, E_E[:, -2], color = "darkblue", linestyle = '--', label = 'Random')
        ax[1].plot(x_axis, E_M[:, -2], color = "darkblue", linestyle = '--', label = 'Random')
        ax[2].plot(x_axis, T_c[:, -2], color = "darkblue", linestyle = '--', label = 'Random')

        # plot RL
        ax[0].plot(x_axis, E_E[:, -1], color = "lightgreen", linestyle = '-', label = 'RL')
        ax[1].plot(x_axis, E_M[:, -1], color = "lightgreen", linestyle = '-', label = 'RL')
        ax[2].plot(x_axis, T_c[:, -1], color = "lightgreen", linestyle = '-', label = 'RL')



        labelsize = 20
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('Energy error',  fontsize = labelsize)
        ax[1].set_ylabel('Angular momentum error', fontsize = labelsize)
        ax[1].set_xlabel('Step', fontsize = labelsize)
        for pl in range(len(ax)):
            ax[pl].set_yscale('log')
            ax[pl].tick_params(axis='both', which='major', labelsize=labelsize)
        plt.savefig('./SympleIntegration_runs/Symple_comparison.png', dpi = 100)
        plt.show()
    
    if plot_grid == True:
        fig, ax = plt.subplots(2, 1, layout = 'constrained', figsize = (10, 10))
        print(a.actions)
        print(np.array(a.actions)[:,0])
        print(E_E[-1,:])

        # for i in range(cases): #Excluding the case with the mixed actions
        sc = ax[0].scatter(np.array(a.actions)[:,0], np.array(a.actions)[:,1], marker = 'o',\
                           s = 500, c = E_E[-1,:-1], cmap = 'RdYlBu', \
                            norm=matplotlib.colors.LogNorm())
        plt.colorbar(sc, ax = ax[0])

        sm = ax[1].scatter(np.array(a.actions)[:,0], np.array(a.actions)[:,1], marker = 'o',\
                           s = 500, c = E_M[-1,:-1], cmap = 'RdYlBu', \
                            norm=matplotlib.colors.LogNorm())
        plt.colorbar(sm, ax = ax[1])

        labelsize = 20
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('t_step',  fontsize = labelsize)
        ax[0].set_title("Final Energy error", fontsize = labelsize)
        ax[1].set_ylabel('t_step', fontsize = labelsize)
        ax[1].set_xlabel('Order', fontsize = labelsize)
        ax[1].set_title("Final Angular momentum error", fontsize = labelsize)
        for pl in range(len(ax)):
            ax[pl].set_yscale('log')
            ax[pl].tick_params(axis='both', which='major', labelsize=labelsize)
        plt.savefig('./SympleIntegration_runs/Symple_comparison2.png', dpi = 100)
        plt.show()

    if plot_errorvstime == True:
        fig, ax = plt.subplots(2, 1, layout = 'constrained', figsize = (10, 10))

        for i in range(cases-1): #Excluding the case with the mixed actions
            ax[0].scatter(E_E[-1, i], T_c[-1, i], marker = markers[i//n_tstep_cases], color = colors[i%n_tstep_cases], s = 100, label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
            ax[1].scatter(E_M[-1, i], T_c[-1, i], marker = markers[i//n_tstep_cases], color = colors[i%n_tstep_cases], s = 100, label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))

        # sm = ax[1].scatter(np.array(a.actions)[:,0], np.array(a.actions)[:,1], marker = 'o',\
        #                 s = 500, c = E_M[-1,:-1], cmap = 'RdYlBu', \
        #                     norm=matplotlib.colors.LogNorm())
        labelsize = 20
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('Final Energy Error',  fontsize = labelsize)
        # ax[0].set_title("Final Energy error", fontsize = labelsize)
        ax[1].set_xlabel('Final Angular momentum Error', fontsize = labelsize)
        # ax[1].set_title("Final Angular momentum error", fontsize = labelsize)
        for pl in range(len(ax)):
            ax[pl].set_yscale('log')
            ax[pl].set_xscale('log')
            ax[pl].set_ylabel('t_comp (s)',  fontsize = labelsize)

            ax[pl].tick_params(axis='both', which='major', labelsize=labelsize)
        plt.savefig('./SympleIntegration_runs/Symple_comparison3.png', dpi = 100)
        plt.show()

def test_1_case(value):
    a = IntegrateEnv()
    terminated = False
    i = 0
    a.reset()
    print(value)

    while terminated == False and i < (a.settings['Integration']['max_steps']):
        x, y, terminated, zz = a.step(value[i%len(value)])
        i += 1
    a.close()
    a.plot_orbit()

    
if __name__ == '__main__':

    ##########################################
    # Run all possibilities with Symple
    # a = IntegrateEnv()
    # steps = 30 # large value just in case
    # values = list()
    # for i in range(len(a.actions)):
    #     values.append([i]*steps)

    # values.append(np.random.randint(0, len(a.actions), size = steps))
    # run_many_symple_cases(values)
    # evaluate_many_symple_cases(values, \
    #                         plot_tstep = True, \
    #                         plot_grid = False,\
    #                         plot_errorvstime = False)

    # test_1_case(values[5])

    ##########################################
    # Run all possibilities with Symple vs RL
    a = IntegrateEnv()
    steps = 30 # large value just in case
    values = list()
    for i in range(len(a.actions)):
        values.append([i]*steps)

    values.append(np.random.randint(0, len(a.actions), size = steps))
    print("Symple simulations", len(values))
    # run_many_symple_cases(a, values)

    # RL
    # run_withRL(a)
    reward = load_reward(a)
    plot_reward(a, reward)

    # evaluate_many_symple_cases(values, \
    #                         plot_tstep = True, \
    #                         plot_grid = False,\
    #                         plot_errorvstime = False)




