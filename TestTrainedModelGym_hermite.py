"""
TestTrainedModelGym_hermite: tests and plots for the RL algorithm

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json
import seaborn as sns

import torch
import torchvision.models as models
import gym

from Cluster.cluster_2D.envs.HermiteIntegration_env import IntegrateEnv_Hermite
from Cluster.cluster_2D.envs.FlexibleIntegration_env import IntegrateEnv_multiple
from TrainingFunctions import DQN, load_reward, plot_reward
from PlotsSimulation import load_state_files, plot_planets_trajectory, \
                            run_trajectory, plot_evolution,\
                            plot_actions_taken, plot_planets_distance


colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']


def evaluate_many_hermite_cases(values, \
                        plot_tstep = True, \
                        plot_grid = True,
                        plot_errorvstime = True,
                        subfolder = None):
    """
    evaluate_many_hermite_cases: plot many things from trained data
    INPUTS:
        values: action vector for each of the cases stored in a list
        plot_tstep: plot step vs energy error, angular momentum and computation time
        plot_grid: plot energy error vs angular momentum
        plot_errorvstime: plot computation time vs energy error and vs angular momentum
        subfolder: save in a subfolder if needed
    """
    cases = len(values)
    a = IntegrateEnv_Hermite()
    a.subfolder = subfolder

    # Load run information for many cases
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

    # Calculate the energy errors
    E_E = np.zeros((steps, cases))
    E_M = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    for i in range(cases):
        E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
        E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
        T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times

    # plot
    colors = ['red', 'green', 'blue', 'orange', 'grey', 'black']
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]
    markers = ['o', 'x', '.', '^', 's']
    n_order_cases = len(a.settings['Integration']['order'])
    n_tstep_cases = len(a.settings['Integration']['t_step'])

    if plot_tstep == True:
        fig, ax = plt.subplots(3, 1, layout = 'constrained', figsize = (10, 10))
        x_axis = np.arange(0, steps, 1)

        # plot 
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
    
    if plot_grid == True: # TODO: does not include RL
        fig, ax = plt.subplots(2, 1, layout = 'constrained', figsize = (10, 10))

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

        for i in range(cases-2): #Excluding the case with the mixed actions
            ax[0].scatter(E_E[-1, i], T_c[-1, i], marker = markers[i//n_tstep_cases], color = colors[i%n_tstep_cases], s = 100, label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
            ax[1].scatter(E_M[-1, i], T_c[-1, i], marker = markers[i//n_tstep_cases], color = colors[i%n_tstep_cases], s = 100, label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
        
        # plot Random actions
        ax[0].scatter(E_E[-1, -2], T_c[-1, -2], marker = markers[(i+1)//n_tstep_cases], color = colors[(i+1)%n_tstep_cases], s = 100, label = 'Random choice')
        ax[1].scatter(E_M[-1, -2], T_c[-1, -2], marker = markers[(i+1)//n_tstep_cases], color = colors[(i+1)%n_tstep_cases], s = 100, label = 'Random choice')
        
        # plot RL
        ax[0].scatter(E_E[-1, -1], T_c[-1, -1], marker = markers[(i+2)//n_tstep_cases], color = colors[(i+2)%n_tstep_cases], s = 100, label = 'RL')
        ax[1].scatter(E_M[-1, -1], T_c[-1, -1], marker = markers[(i+2)//n_tstep_cases], color = colors[(i+2)%n_tstep_cases], s = 100, label = 'RL')

        labelsize = 20
        ax[0].legend(loc='upper right')
        ax[0].set_xlabel('Final Energy Error',  fontsize = labelsize)
        ax[1].set_xlabel('Final Angular momentum Error', fontsize = labelsize)
        for pl in range(len(ax)):
            ax[pl].set_yscale('log')
            ax[pl].set_xscale('log')
            ax[pl].set_ylabel('t_comp (s)',  fontsize = labelsize)

            ax[pl].tick_params(axis='both', which='major', labelsize=labelsize)
        plt.savefig('./SympleIntegration_runs/Symple_comparison3.png', dpi = 100)
        plt.show()

def test_1_case(value):
    """
    test_1_case: run 1 case to test environment
    INPUTS:
        value: list of actions
    """
    a = IntegrateEnv_Hermite()
    terminated = False
    i = 0
    a.reset()
    print(value)

    while terminated == False and i < (a.settings['Integration']['max_steps']):
        x, y, terminated, zz = a.step(value[i%len(value)])
        i += 1
    a.close()
    a.plot_orbit()

    
def plot_steps(values, env = None, steps = None, RL = False, reward_plot = False, save_path = None):
    """
    plot_steps: multiple subplots
    INPUTS:
        values: list of actions for each case
        env: environment that has been run
        steps: number of steps taken
        RL: include reinforcement learning plots
        reward_plot: include plot with the reward value
        save_path: path to save the figure

    PLOTS:
        plot 1: cartesian coordinates (orbit) of restrictive option
        plot 2:cartesian coordinates (orbit) of RL algorithm
        plot 3: steps vs pairwise distance between bodies
        plot 4: actions taken by the RL algorithm
        plot 5: steps vs reward function
        plot 6: steps vs energy error
    """
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    #############################################
    # Load run information for symple cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = '_Case_%.2E'%(env_hermite.actions[i]))
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_%i'%i)
    
    if RL == True:
        # Load RL run
        cases += 1
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = '_Case_RL')
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_RL')
        steps_taken = np.load(env.settings['Integration']['savefile'] +'RL_steps_taken.npy', allow_pickle=True)

    Reward = np.load(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward.npy')

    if steps == None:
        steps = len(cons[0][:,0])

    #############################################
    # Calculate the energy errors
    E_E = np.zeros((steps, cases))
    E_M = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    for i in range(cases):
        E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
        E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
        T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times

    #############################################
    # plot
    fig = plt.figure(figsize = (10,10))
    labelsize = 24
    linewidth = 2.5

    if RL == True: 
        plots_v = 9
    else: 
        plots_v = 7
    if reward_plot == True:
        plots_v += 2

    if RL == True:
        fig = plt.figure(figsize = (12, 18))
    else:
        fig = plt.figure(figsize = (12, 18))
    gs1 = matplotlib.gridspec.GridSpec(plots_v, 2, figure = fig, left=0.13, wspace=0.3, 
                                       hspace = 1.7, right = 0.99,
                                        top = 0.97, bottom = 0.14)
    ax1 = fig.add_subplot(gs1[0:3, 0]) # cartesian symple
    ax2 = fig.add_subplot(gs1[0:3, 1]) # cartesian another step
    ax3 = fig.add_subplot(gs1[3:5, :]) # pairwise distance
    ax4 = fig.add_subplot(gs1[9:, :]) # energy error 

    if RL == True:
        ax7 = fig.add_subplot(gs1[5:7, :]) # actions
    if reward_plot == True:
        ax6 = fig.add_subplot(gs1[7:9, :]) # reward
        ax6.set_yscale('symlog', linthresh = 1e1)

    x_axis = np.arange(0, steps, 1)

    # Plot cartesian
    name_planets = np.arange(np.shape(state)[1]).astype(str)    
    
    # Do it for RL
    plot_planets_trajectory(ax1, state[0]/1.496e11, name_planets, steps = steps, labelsize = labelsize)
    distance = plot_planets_distance(ax3, x_axis, state[0]/1.496e11, name_planets, steps = steps, labelsize = labelsize)
    
    if RL == True:
        plot_planets_trajectory(ax2, state[-1]/1.496e11, name_planets, steps = steps, labelsize = labelsize)
        ax2.set_title(r"RL-variable $\mu$", fontsize = labelsize+2)
    else:
        plot_planets_trajectory(ax2, state[-1], name_planets, steps = steps)
        ax2.set_title(r"$\mu = %.2E$"%(env_hermite.actions[-1]), fontsize = labelsize+2)

    # Plot energy errors
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]

    for i in range(cases): # plot all combinations with constant dt and order
        if RL == True and i == cases-1:
            label = 'RL'
            linestyle = '-'
        else:
            label = r"$%i: \mu$ =%.2E"%(i, env.actions[i])
            linestyle = lines[i%len(lines)]
        plot_evolution(ax4, x_axis, E_E[:, i], label = label, colorindex = i, linestyle = linestyle, linewidth = linewidth)
        # plot_evolution(ax5, x_axis, T_c[:, i], label = label, colorindex = i, linestyle = linestyle, linewidth = linewidth)
    
        if reward_plot == True:
            plot_evolution(ax6, x_axis, Reward[i, :], label = label, colorindex = i, linestyle = linestyle, linewidth = linewidth)
    
    if RL == True:
    # plot RL and actions
        label = 'RL'
        plot_actions_taken(ax7, x_axis, steps_taken)
        ax7.set_ylim([-1, cases])
        ax7.set_ylabel('Action taken', fontsize = labelsize)
        ax7.set_yticks(np.arange(0, cases-1))

    # Draw vertical lines for close encounters
    X_vert = []
    for D in distance:
        index_x = np.where(D < 1)[0]
        X_vert.append(index_x)
    def flatten(xss):
        return [x for xs in xss for x in xs]
    X_vert = flatten(X_vert)
    
    prev = 0
    for X_D in X_vert:
        print(prev, X_D)
        for ax_vert in [ax3, ax6, ax7]:
            ax_vert.axvline(x=X_D, ymin=-1.2, ymax=1, c= 'k', lw =2, zorder = 0, \
                    clip_on = False, alpha = 0.2, linestyle = '--')
        for ax_vert in [ax4]:
            ax_vert.axvline(x=X_D, ymin=0, ymax=1, c= 'k', lw =2, zorder = 0,\
                        clip_on = False, alpha = 0.2, linestyle = '--')
        prev = X_D
                

    ax3.set_ylabel(r'$\vert\vert \vec r_i - \vec r_j\vert\vert$ (au)', fontsize = labelsize)
    ax4.set_ylabel('Energy error',  fontsize = labelsize)
    ax4.set_xlabel('Step', fontsize = labelsize)

    ax1.set_title(r"$\mu$ = %.2E"%(env_hermite.actions[0]), fontsize = labelsize+2)
    ax4.set_yscale('log')
    ax6.set_ylabel('Reward',  fontsize = labelsize)

    for ax_i in [ax1, ax2, ax3, ax4,  ax6, ax7]:
        ax_i.tick_params(axis='both', which='major', labelsize=labelsize-2)
    
    handles, labels = ax4.get_legend_handles_labels()
    ax4.scatter(0,0 , color = 'white', label = ' ')
    ax4.scatter(0,0 , color = 'white', label = ' ')
    print(handles, labels)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                       fancybox = True, ncol = 3, fontsize = labelsize-2)
    
    if save_path != None:
        plt.savefig(save_path)
    else:
        if RL == True:
            plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_RLsteps.png', dpi = 100)
        else:
            plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_steps.png', dpi = 100)
    # plt.show()
    

def plot_rewards(values, initializations, reward_functions, env = None, steps = None):
    """
    plot_rewards: plot reward values for different functions
    INPUTS: 
        values: list of actions for the different cases
        initializations: number of initializations used to create the data
        reward_funcitons: list of rewards chosen and the weights
        env: environment used
        steps: steps taken
    """
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    if steps == None:
        steps = len(cons[0][:,0])

    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(3, 4, width_ratios=[1,1,1,0.1],
                                    left=0.08, wspace=0.1, hspace = 0.5, right = 0.93,
                                    top = 0.99, bottom = 0.01)
    labelsize = 10
    axbar = plt.subplot(gs1[:,-1]  )
    cm = plt.cm.get_cmap('RdYlBu')

    for reward_case in range(len(reward_functions)):
        #############################################
        # Load run information for cases
        state = list()
        cons = list()
        tcomp = list()
        name = list()
        for i in range(initializations):
            for j in range(cases):
                state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = 'Seed_%i_Case_%.2E_Rf_%i'%(i, env_hermite.actions[j], reward_case))
                state.append(state_i)
                cons.append(cons_i)
                tcomp.append(tcomp_i)
                name.append('Seed_%i_Case_%.2E'%(i, env_hermite.actions[j]))
        
        Reward = np.load(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_multiple_%i'%reward_case +'.npy')

        #############################################
        # Calculate the energy errors
        E_E = np.zeros((initializations*cases, steps))
        T_c = np.zeros((initializations*cases, steps))
        action = np.zeros((initializations*cases, steps))
        for i in range(initializations):
            for j in range(cases):
                E_E[i*cases+j, :] = abs(cons[i*cases + j][:, 1]) # absolute relative energy error
                T_c[i*cases+j, :] = np.cumsum(tcomp[i*cases + j]) # add individual computation times
                action[i*cases+j, :] = np.ones(steps) * env_hermite.actions[j]

        E_E = E_E.flatten()
        Reward = Reward.flatten()
        action = action.flatten()
        T_c = T_c.flatten()

        #############################################
        # plot
        Reward_name = 'Type %i, W1 = %.2f, W2 = %.2f, W3 = %.2f'%(reward_functions[reward_case][0],\
                                                                  reward_functions[reward_case][1],\
                                                                  reward_functions[reward_case][2],\
                                                                  reward_functions[reward_case][3])
        ax = fig.add_subplot(gs1[reward_case//3, reward_case%3]) # cartesian symple
        ax.set_title(Reward_name, fontsize = labelsize-3)

        x = E_E
        y = Reward
        z = action
        sc = ax.scatter(x, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())

        ax.set_xlabel('Energy error',  fontsize = labelsize)
        ax.set_ylabel('Reward',  fontsize = labelsize)
        ax.set_xscale('log')
        if reward_case == 5 or reward_case == 6:
            ax.set_yscale('symlog')
    
    fig.colorbar(sc, axbar)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards.png', dpi = 100, layout = 'tight' )
    plt.show()
    
def plot_rewards2(values, initializations, reward_functions, env = None, steps = None):
    """
    plot_rewards2: same as plot_rewards but different distribution of subplots
    INPUTS: 
        values: list of actions for the different cases
        initializations: number of initializations used to create the data
        reward_funcitons: list of rewards chosen and the weights
        env: environment used
        steps: steps taken
    """
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    if steps == None:
        steps = len(cons[0][:,0])

    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(4, 3, width_ratios=[1,1,0.05],
                                    left=0.08, wspace=0.2, hspace = 0.5, right = 0.92,
                                    top = 0.97, bottom = 0.05)

    labelsize = 12
    axbar = plt.subplot(gs1[:,-1]  )
    cm = plt.cm.get_cmap('RdYlBu')

    for ploti, reward_case in enumerate(np.arange(0, len(reward_functions))):
        #############################################
        # Load run information for cases
        state = list()
        cons = list()
        tcomp = list()
        name = list()

        E_E = np.zeros((initializations*cases, steps))
        Delta_EE = np.zeros((initializations*cases, steps))
        T_c = np.zeros((initializations*cases, steps))
        action = np.zeros((initializations*cases, steps))
        for i in range(initializations):
            for j in range(cases):
                state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = 'Seed_%i_Case_%.2E_Rf_%i'%(i, env_hermite.actions[j], reward_case))
                state.append(state_i)
                cons.append(cons_i)
                tcomp.append(tcomp_i)
                name.append('Seed_%i_Case_%.2E'%(i, env_hermite.actions[j]))

                E_E[i*cases+j, :] = abs(cons_i[:, 1]) # absolute relative energy error
                Delta_EE[i*cases+j, :-1] = -(np.log10(abs(cons_i[1:, 1])) - np.log10(abs(cons_i[:-1, 1])))  # relative energy error minus previous step
                T_c[i*cases+j, :] = np.cumsum(tcomp_i) # add individual computation times
                action[i*cases+j, :] = np.ones(steps) * env_hermite.actions[j]

        Reward = np.load(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_multiple_%i'%reward_case +'.npy')
        
        E_E = E_E.flatten()
        Delta_EE = Delta_EE.flatten()
        Reward = Reward.flatten()
        action = action.flatten()
        T_c = T_c.flatten()

        #############################################
        # plot
        Reward_name = 'Type %i, W1 = %.2f, W2 = %.2f, W3 = %.2f'%(reward_functions[reward_case][0],\
                                                                  reward_functions[reward_case][1],\
                                                                  reward_functions[reward_case][2],\
                                                                  reward_functions[reward_case][3])
        ax = fig.add_subplot(gs1[ploti, 0]) 
        ax2 = fig.add_subplot(gs1[ploti, 1]) 
        ax.set_title(Reward_name, fontsize = labelsize)

        x = E_E
        x2 = Delta_EE
        y = Reward
        z = action
        sc = ax.scatter(x, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())
        ax2.scatter(x2, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())

        ax.set_xlabel(r'$\Delta E_i$',  fontsize = labelsize-1)
        
        ax2.set_xlabel(r'$\log(\vert \Delta E_{i-1}\vert )- log(\vert \Delta E_{i}\vert )$',  fontsize = labelsize-1)
        ax.set_ylabel(r'Reward ($R$)',  fontsize = labelsize-1)
        ax.set_xscale('log')
    
    cbar = fig.colorbar(sc, axbar)
    cbar.set_label(r'Time step parameter ($\mu$)', fontsize = labelsize)
    cbar.ax.tick_params(labelsize=labelsize)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards.png', dpi = 100, layout = 'tight' )
    plt.show()

def plot_one_reward(values, initializations, reward_function, reward_case, env = None, steps = None):
    """
    plot_one_reward: plot reward values for different runs 
    INPUTS: 
        values: list of actions for the different cases
        initializations: number of initializations used to create the data
        reward_functions: list of rewards chosen and the weights
        reward_case: reward function and weights chosen
        env: environment used
        steps: steps taken
    """
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    if steps == None:
        steps = len(cons[0][:,0])

    fig = plt.figure(figsize = (8,11))
    gs1 = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1,0.08],
                                    left=0.13, wspace=0.25, hspace = 0.2, right = 0.88,
                                    top = 0.95, bottom = 0.06)
    labelsize = 17
    axbar = plt.subplot(gs1[:,-1]  )
    cm = plt.cm.get_cmap('RdYlBu')

    #############################################
    # Load run information for cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()

    E_E = np.zeros((initializations*cases, steps))
    Delta_EE = np.zeros((initializations*cases, steps))
    T_c = np.zeros((initializations*cases, steps))
    action = np.zeros((initializations*cases, steps))
    for i in range(initializations):
        for j in range(cases):
            state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = 'Seed_%i_Case_%.2E_Rf_%i'%(i, env_hermite.actions[j], reward_case))
            state.append(state_i)
            cons.append(cons_i)
            tcomp.append(tcomp_i)
            name.append('Seed_%i_Case_%.2E'%(i, env_hermite.actions[j]))
            
            E_E[i*cases+j, :] = abs(cons_i[:, 1]) # absolute relative energy error
            Delta_EE[i*cases+j, :-1] = -(np.log10(abs(cons_i[1:, 1])) - np.log10(abs(cons_i[:-1, 1])))  # relative energy error minus previous step
            T_c[i*cases+j, :] = np.cumsum(tcomp_i) # add individual computation times
            action[i*cases+j, :] = np.ones(steps) * env_hermite.actions[j]

    Reward = np.load(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_multiple_%i'%reward_case +'.npy')
    
    E_E = E_E.flatten()
    Delta_EE = Delta_EE.flatten()
    Reward = Reward.flatten()
    action = action.flatten()
    T_c = T_c.flatten()

    index = np.where(Reward == 0)[0]
    E_E = np.delete(E_E, index)
    Delta_EE = np.delete(Delta_EE, index)
    Reward = np.delete(Reward, index)
    T_c = np.delete(T_c, index)
    action = np.delete(action, index)

    #############################################
    # plot
    Reward_name = 'Type %i, W1 = %.2f, W2 = %.2f, W3 = %.2f'%(reward_functions[reward_case][0],\
                                                                reward_functions[reward_case][1],\
                                                                reward_functions[reward_case][2],\
                                                                reward_functions[reward_case][3])
    ax = fig.add_subplot(gs1[0, 0]) 
    ax2 = fig.add_subplot(gs1[1, 0]) 

    x = E_E
    x2 = Delta_EE
    y = Reward
    z = action
    sc = ax.scatter(x, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())
    ax2.scatter(x2, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())
    ax.set_xlabel(r'$\Delta E_i$',  fontsize = labelsize)
    ax2.set_xlabel(r'$-(\log(\vert \Delta E_i\vert )- log(\vert \Delta E_{i-1}\vert ))$',  fontsize = labelsize)
    ax.set_ylabel(r'Reward ($R$)',  fontsize = labelsize)
    ax2.set_ylabel(r'Reward ($R$)',  fontsize = labelsize)
    ax.set_title(Reward_name, fontsize = labelsize, y = 1.05)
    
    ax.set_xscale('log')
    ax2.set_xscale('symlog', linthresh = 1e-3)
    ax2.set_yscale('symlog', linthresh = 1e0)

    ax.tick_params(axis='x', labelsize=labelsize-3)
    ax.tick_params(axis='y', labelsize=labelsize-3)
    ax2.tick_params(axis='x', labelsize=labelsize-3)
    ax2.tick_params(axis='y', labelsize=labelsize-3)
    
    cbar = fig.colorbar(sc, axbar)
    cbar.set_label(r'Time step parameter ($\mu$)', fontsize = labelsize)
    cbar.ax.tick_params(labelsize=labelsize-3)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards_1.png', dpi = 100)
    plt.show()


def plot_rewards_random_action(values, initializations, reward_functions, env = None, steps = None):
    """
    plot_rewards_random_action: plot rewards for a simulation in which random actions have been taken for each step
    INPUTS: 
        values: list of actions for the different cases
        initializations: number of initializations used to create the data
        reward_functions: list of rewards chosen and the weights
        env: environment used
        steps: steps taken
    """
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    if steps == None:
        steps = len(cons[0][:,0])

    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(4, 3, width_ratios=[1,1,0.1],
                                    left=0.08, wspace=0.1, hspace = 0.5, right = 0.93,
                                    top = 0.95, bottom = 0.01)
    labelsize = 10
    axbar = plt.subplot(gs1[:,-1]  )
    cm = plt.cm.get_cmap('RdYlBu')

    for ploti, reward_case in enumerate(np.arange(0, len(reward_functions))):
        #############################################
        # Load run information for cases
        state = list()
        cons = list()
        tcomp = list()
        name = list()

        E_E = np.zeros((initializations, steps))
        Delta_EE = np.zeros((initializations, steps))
        T_c = np.zeros((initializations, steps))
        action = np.zeros((initializations, steps))
        for i in range(initializations):
            state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = 'Seed_%i_Rf_%i'%(i, reward_case))
            state.append(state_i)
            cons.append(cons_i)
            tcomp.append(tcomp_i)

            E_E[i, :] = abs(cons_i[:, 1]) # absolute relative energy error
            Delta_EE[i, :-1] = -(np.log10(abs(cons_i[1:, 1])) - np.log10(abs(cons_i[:-1, 1])))  # relative energy error minus previous step
            T_c[i, :] = np.cumsum(tcomp_i) # add individual computation times
            for j in range(len(cons_i[:,0])):
                action[i, j] = env.actions[int(cons_i[j, 0])]

        Reward = np.load(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_multiple_%i'%reward_case +'.npy')
        
        E_E = E_E.flatten()
        Delta_EE = Delta_EE.flatten()
        Reward = Reward.flatten()
        action = action.flatten()
        T_c = T_c.flatten()

        #############################################
        # plot
        Reward_name = 'Type %i, W1 = %.2f, W2 = %.2f, W3 = %.2f'%(reward_functions[reward_case][0],\
                                                                  reward_functions[reward_case][1],\
                                                                  reward_functions[reward_case][2],\
                                                                  reward_functions[reward_case][3])
        ax = fig.add_subplot(gs1[ploti, 0]) 
        ax2 = fig.add_subplot(gs1[ploti, 1]) 
        ax.set_title(Reward_name, fontsize = labelsize-3)

        x = E_E
        x2 = Delta_EE
        y = Reward
        z = action
        sc = ax.scatter(x, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())
        ax2.scatter(x2, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())

        ax.set_xlabel('Energy error',  fontsize = labelsize)
        ax2.set_xlabel('Energy error step difference',  fontsize = labelsize)
        ax.set_ylabel('Reward',  fontsize = labelsize)
        ax.set_xscale('log')
        ax2.set_xscale('log')
    
    fig.colorbar(sc, axbar)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards.png', dpi = 100, layout = 'tight' )
    plt.show()

def plot_runs_trajectory(cases, action, steps, seeds, env = None):
    """
    plot_runs_trajectory: plot orbit with different initialization seeds
    INPUTS:
        cases: number of cases studied
        action: action taken for the given case
        steps: steps taken
        seeds: array with the seed for each initialization
        env: environment used
    """
    if env == None:
        env = IntegrateEnv_Hermite()

    # Load run information for Hermite cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile =  '_traj_action_%i_initialization_%i'%(action, i))
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_%i'%i)

    # plot
    title_size = 20
    label_size = 18

    name_planets = (np.arange(np.shape(state)[1])+1).astype(str)
    fig, axes = plt.subplots(nrows=int(cases//2), ncols= 2, layout = 'constrained', figsize = (8, 12))

    for i, ax in enumerate(axes.flat):
        if i == 0:
            legend_on = True
        else:
            legend_on = False
        plot_planets_trajectory(ax, state[i]/1.496e11, name_planets, labelsize = label_size, steps = steps, legend_on = legend_on)
        ax.set_title("Seed = %i"%(seeds[i]), fontsize = label_size+2)
        
        #     ax.legend(fontsize = label_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size-4, labelrotation=0)

    plt.axis('equal')
    plt.savefig('HermiteIntegration_runs/' + env.subfolder + 'plot_state_action%i'%action, dpi = 100)
    plt.show()
    
def plot_steps_reward(values, rewards, env = None, steps = None):
    """
    plot_steps_reward: plot reward for different reward functions
    INPUTS:
        values: list of the actions taken for each case
        rewards: reward functions and weights in a list of lists
        env: environment used
        steps: steps taken
    """
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    fig = plt.figure(figsize = (10,14))
    gs1 = matplotlib.gridspec.GridSpec(len(rewards)+ 1, 1, figure = fig, left=0.08, wspace=0.3, 
                                       hspace = 0.5, right = 0.99,
                                        top = 0.97, bottom = 0.11)

    for z in range(len(rewards)):
        #############################################
        # Load run information for symple cases
        state = list()
        cons = list()
        tcomp = list()
        name = list()
        for i in range(cases):
            state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = '_Case_%.2E_%i'%(env_hermite.actions[i], z))
            state.append(state_i)
            cons.append(cons_i)
            tcomp.append(tcomp_i)
            name.append('Case_%i'%i)
        
        Reward = np.load(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_'+str(z)+'.npy')

        if steps == None:
            steps = len(cons[0][:,0])

        #############################################
        # Calculate the energy errors
        E_E = np.zeros((steps, cases))
        E_M = np.zeros((steps, cases))
        T_c = np.zeros((steps, cases))
        for i in range(cases):
            E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
            E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
            T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times

        x_axis = np.arange(0, steps, 1)

        labelsize = 12

        ax1 = fig.add_subplot(gs1[z+1, 0]) # cartesian symple
        ax1.set_title('Type %i, W1 = %.2f, W2 = %.2f, W3 = %.2f'%(rewards[z][0],\
                                                                rewards[z][1],\
                                                                rewards[z][2],\
                                                                rewards[z][3]))
        if z == 0:
            ax0 = fig.add_subplot(gs1[0, 0])
            name_planets = np.arange(np.shape(state)[1]).astype(str) 
            plot_planets_distance(ax0, x_axis, state[0], name_planets, steps = steps)
            ax0.set_yscale('log')
            ax0.set_ylabel('Pair-wise distance', fontsize = labelsize)
            ax0.tick_params(axis='both', which='major', labelsize=labelsize)
            
        # Plot energy errors
        lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]

        for i in range(cases): # plot all combinations with constant dt and order
            label = "Tstep parameter %.2E"%(env.actions[i])
            linestyle = lines[i%len(lines)]
            plot_evolution(ax1, x_axis[1:-1], Reward[i, 1:-1], label = label, colorindex = i, linestyle = linestyle)
        
        if z == len(rewards)-1:
            ax1.set_xlabel('Step', fontsize = labelsize)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.52), \
                       fancybox = True, ncol = cases//2+cases%2, fontsize = labelsize)

        ax1.set_ylabel(r'$\vert\vert fr_i - r_j\vert\vert$')
        ax1.set_ylabel('Reward',  fontsize = labelsize)
        ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_steps_rewards.png', dpi = 100)
    plt.show()
    
def plot_int_comparison(I, steps, ENV):
    """
    plot_int_comparison: compare different integrators
    INPUTS:
        I: list of integrators
        steps: steps taken
        ENV: list of environments used for each integration
    """
    fig = plt.figure(figsize = (10,14))
    gs1 = matplotlib.gridspec.GridSpec(len(I)+2, 1, figure = fig, left=0.13, wspace=0.3, 
                                       hspace = 0.5, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    ENERGY = []
    AX = []
    for z in range(len(I)):
        #############################################
        # Load run information for symple cases
        state = list()
        cons = list()
        tcomp = list()
        name = list()
        state_i, cons_i, tcomp_i = load_state_files(ENV[z], steps, namefile = '_'+I[z])
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        env = ENV[z]
        steps_taken = np.load(env.settings['Integration']['savefile'] +'RL_steps_taken'+I[z]+'.npy', allow_pickle=True)

        
        if steps == None:
            steps = len(cons[0][:,0])

        #############################################
        cases = 1
        # Calculate the energy errors
        E_E = np.zeros((steps, cases))
        E_M = np.zeros((steps, cases))
        T_c = np.zeros((steps, cases))
        for i in range(cases):
            E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
            E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
            T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times

        ENERGY.append(E_E[:, 0])
        x_axis = np.arange(0, steps, 1)
        labelsize = 16

        if z == 0:
            ax0 = fig.add_subplot(gs1[0, 0])
            name_planets = np.arange(np.shape(state)[1]).astype(str) 
            distance = plot_planets_distance(ax0, x_axis, state[0]/1.496e11, name_planets, steps = steps, labelsize = labelsize)
            ax0.set_yscale('log')
            ax0.set_ylabel('Pair-wise \n distance (au)', fontsize = labelsize+1)
            ax0.tick_params(axis='both', which='major', labelsize=labelsize)
            AX.append(ax0)
            
        ax1 = fig.add_subplot(gs1[z+1, 0]) # cartesian symple
        ax1.set_title(I[z], fontsize = labelsize+3)
        plot_actions_taken(ax1, x_axis, steps_taken)
        ax1.set_ylim([-1, 6])
        ax1.set_ylabel('Action taken', fontsize = labelsize+1)
        ax1.set_yticks(np.arange(1, 6))
        ax1.tick_params(axis='both', which='major', labelsize=labelsize)
        AX.append(ax1)

        # Plot energy errors
        lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]

    ax2 = fig.add_subplot(gs1[-1, 0]) 
    label = "Tstep parameter %.2E"%(env.actions[i])
    linestyle = lines[i%len(lines)]
    for z in range(len(ENV)):
        plot_evolution(ax2, x_axis, ENERGY[z], label = I[z], colorindex = z, linestyle = linestyle, linewidth = 2.5)
        
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), \
                       fancybox = True, ncol = 4, fontsize = labelsize+3)
    ax2.set_ylabel('Energy error',  fontsize = labelsize+1)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.set_yscale('log')
    ax2.set_xlabel("Steps",fontsize = labelsize+1)
    AX.append(ax2)

    #  Draw vertical lines for close encounters
    X_vert = []
    for D in distance:
        index_x = np.where(D < 1)[0]
        X_vert.append(index_x)
    def flatten(xss):
        return [x for xs in xss for x in xs]
    X_vert = flatten(X_vert)
    for ax_vert in AX[0:-1]:
        for X_D in X_vert:
            ax_vert.axvline(x=X_D, ymin=-1.2, ymax=1, c= 'k', lw =2, zorder = 0, \
                        clip_on = False, alpha = 0.2, linestyle = '--')
    for ax_vert in [AX[-1]]:
        for X_D in X_vert:
            ax_vert.axvline(x=X_D, ymin=0, ymax=1, c= 'k', lw =2, zorder = 0,\
                         clip_on = False, alpha = 0.2, linestyle = '--')
    
    plt.savefig('./MixedIntegration_runs/'+ 'Comparison_steps.png', dpi = 100)
    plt.show()

def plot_EvsTcomp(values, initializations, steps = None, env = None):
    """
    plot_EvsTcomp: plot energy error vs computation time for different cases
    INPUTS:
        values: list of the actions taken for each case
        initializations: number of initializations used to generate the data
        steps: steps taken
        env: environment used
    """
    if env == None:
        env = IntegrateEnv_Hermite()

    cases = len(values)

    E_final = np.zeros((initializations, cases+1))
    Tcomp_final = np.zeros((initializations, cases+1))

    for z in range(initializations):
        cases = len(values)

        #############################################
        # Load run information for symple cases
        state = list()
        cons = list()
        tcomp = list()
        name = list()
        for k in range(len(values)):
            state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile ='_Case_%.2E_%i'%(env_hermite.actions[k], z))
            state.append(state_i)
            cons.append(cons_i)
            tcomp.append(tcomp_i)
            name.append('Case_%i'%k)

        cases += 1
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = '_Case_RL_%i'%z)
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_RL')

        if steps == None:
            steps = len(cons[0][:,0])

        #############################################
        # Calculate the energy errors
        E_E = np.zeros((steps, cases))
        E_M = np.zeros((steps, cases))
        T_c = np.zeros((steps, cases))
        for i in range(cases):
            E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
            E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error
            T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times

        E_final[z, :] = E_E[-1, :]
        Tcomp_final[z, :] = T_c[-1, :]


    fig = plt.figure(figsize = (6,6))
    gs1 = matplotlib.gridspec.GridSpec(2, 2, figure = fig, width_ratios = (3, 1), height_ratios = (1, 3), \
                                       left=0.15, wspace=0.3, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    
    msize = 50
    alphavalue = 0.5
    alphavalue2 = 0.9
    X = [Tcomp_final[:, 0], Tcomp_final[:, -2],Tcomp_final[:, -1]]
    Y = [E_final[:, 0], E_final[:, -2], E_final[:, -1]]
    labels = [r'$\mu = $%.1E'%env.actions[0], r'$\mu = $%.1E'%env.actions[-1],"RL", r'$\mu = $%.1E'%env.actions[2]]
    markers = ['o', 'x', 's', 'o']

    ax1 = fig.add_subplot(gs1[1, 0]) 
    ax2 = fig.add_subplot(gs1[0, 0])
    ax3 = fig.add_subplot(gs1[1, 1])

    binsx = np.logspace(np.log10(5e-1),np.log10(0.8e1), 50)
    binsy = np.logspace(np.log10(1e-14),np.log10(1e1), 50)

    order = [1,2,0]
    alpha = [0.5, 0.5, 0.9]
    for i in range(len(X)):
        ax1.scatter(X[i], Y[i], color = colors[i], alpha = alphavalue, marker = markers[i],\
                s = msize, label = labels[i], zorder =order[i])
        ax2.hist(X[i], bins = binsx, color = colors[i],  alpha = alpha[i], edgecolor=colors[i], \
                 linewidth=1.2, zorder =order[i])
        ax2.set_yscale('log')
        ax3.hist(Y[i], bins = binsy, color = colors[i], alpha = alpha[i], orientation='horizontal',\
                 edgecolor=colors[i], linewidth=1.2, zorder =order[i])
    
    labelsize = 12
    ax1.legend(fontsize = labelsize)
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax3.set_yscale('log')
    ax1.set_xlim([5e-1, 3])
    ax2.set_xlim([5e-1, 3])
    ax1.set_ylim([1e-14, 1e1])
    ax3.set_ylim([1e-14, 1e1])
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_steps_rewards.png', dpi = 100)
    plt.show()

def plot_EvsTcomp_integrators(I, initializations, steps = None, env = None):
    """
    plot_EvsTcomp_integrators: plot energy error vs computation time for different integrators
    INPUTS:
        I: list of integrators
        initializations: number of initializations used to generate the data
        steps: steps taken
        env: environment used
    """
    if env == None:
        env = IntegrateEnv_Hermite()

    cases = len(I)

    E_final = np.zeros((initializations, cases))
    Tcomp_final = np.zeros((initializations, cases))
    E_final2 = np.zeros((initializations, cases))
    Tcomp_final2 = np.zeros((initializations, cases))

    for z in range(initializations):
        cases = len(I)

        #############################################
        # Load run information for symple cases
        state = list()
        cons = list()
        tcomp = list()

        state_2 = list()
        cons_2 = list()
        tcomp_2 = list()

        for k in range(cases):
            state_i, cons_i, tcomp_i = load_state_files(ENV[k], steps, namefile ='Accurate_%s_%i'%(I[k], z))
            state.append(state_i)
            cons.append(cons_i)
            tcomp.append(tcomp_i)

            state_i, cons_i, tcomp_i = load_state_files(ENV[k], steps, namefile ='Inaccurate_%s_%i'%(I[k], z))
            state_2.append(state_i)
            cons_2.append(cons_i)
            tcomp_2.append(tcomp_i)

        if steps == None:
            steps = len(cons[0][:,0])

        #############################################
        # Calculate the energy errors
        E_E = np.zeros((steps, cases))
        T_c = np.zeros((steps, cases))
        E_E2 = np.zeros((steps, cases))
        T_c2 = np.zeros((steps, cases))
        for i in range(cases):
            E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
            T_c[:, i] = np.cumsum(tcomp[i]) # add individual computation times
            E_E2[:, i] = abs(cons_2[i][:, 1]) # absolute relative energy error
            T_c2[:, i] = np.cumsum(tcomp_2[i]) # add individual computation times

        E_final[z, :] = E_E[-1, :]
        Tcomp_final[z, :] = T_c[-1, :]
        E_final2[z, :] = E_E2[-1, :]
        Tcomp_final2[z, :] = T_c2[-1, :]

    fig = plt.figure(figsize = (6, 5))
    gs1 = matplotlib.gridspec.GridSpec(1, 1, figure = fig,\
                                       left=0.15, wspace=0.3, 
                                       hspace = 0.2, right = 0.97,
                                        top = 0.97, bottom = 0.11)
    
    msize = 50
    alphavalue = 0.5
    alphavalue2 = 0.9
    X = [Tcomp_final[:, 0], Tcomp_final[:, 1],Tcomp_final[:, 2]]
    Y = [E_final[:, 0], E_final[:, 1], E_final[:, 2]]
    X2 = [Tcomp_final2[:, 0], Tcomp_final2[:, 1],Tcomp_final2[:, 2]]
    Y2 = [E_final2[:, 0], E_final2[:, 1], E_final2[:, 2]]
    markers = ['o', 'x', 's', 'o']

    ax1 = fig.add_subplot(gs1[0, 0]) 
    labels1 = [r'%s: $\mu$ = %.1E'%(I[0], ENV[0].actions[0]),\
                r'%s: $\mu$ = %.1E'%(I[1], ENV[1].actions[0]),\
                r'%s: $\Delta t$ = %.1E'%(I[2], ENV[2].actions[0])]
    
    labels2 = [r'%s: $\mu$ = %.1E'%(I[0], ENV[0].actions[5]),\
                r'%s: $\mu$ = %.1E'%(I[1], ENV[1].actions[5]),\
                r'%s: $\Delta t$ = %.1E'%(I[2], ENV[2].actions[5])]

    for i in range(len(X)):
        ax1.scatter(X[i], Y[i], color = colors[i], marker = markers[0],\
                s = msize, label = labels1[i], alpha = alphavalue)
        ax1.scatter(X2[i], Y2[i], color = colors[i], marker = markers[1],\
                s = msize, label = labels2[i], alpha = alphavalue)
    
    labelsize = 12
    ax1.legend(fontsize = labelsize)
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.set_xlim([6e-1, 3])
    ax1.set_ylim([1e-13, 1e5])
    
    plt.savefig('./MixedIntegration_runs/Comparison_integr.png', dpi = 100)
    plt.show()
    
if __name__ == '__main__':
    experiment = 4 # number of the experiment to be run
    seed = 1
    
    if experiment == 0: 
        # plot trajectory with hermite for a fixed action, many initializations
        cases = 500
        action = 0
        steps = 100
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "1_Initializations/"
        env_hermite.subfolder = name_subfolder

        env_hermite.settings['Integration']['check_step'] = 1e-1
        seeds = np.arange(cases)
        def runs_trajectory_initializations(cases, action, steps):
            env_hermite.save_state_to_file = True # eliminate
            for j in range(cases):
                if j >9: # eliminate
                    env_hermite.save_state_to_file = False
                print("Case %i"%j)
                name_suff = ('_traj_action_%i_initialization_%i'%(action, j))
                reward = run_trajectory(seed = seeds[j], action = action, env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
                
        runs_trajectory_initializations(cases, action, steps)
        cases = 6 # only plot x
        plot_runs_trajectory(cases, action, steps, seeds, env_hermite)
        
    elif experiment == 1: 
        # plot reward function comparison for different actions and initializations
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "2_RewardStudy/"
        env_hermite.subfolder = name_subfolder

        cases = len(env_hermite.actions)
        steps = 100
        initializations = 10
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        seed = np.arange(initializations)
        reward_functions = [
                # [1, 1.0, 1.0, 1.0],
                # [2, 10.0, 100.0, 4.0],
                # [3, 1.0, 100.0, 5.0],
                # [4, 1.0, 100.0, 1.0],
                [2, 10.0, 200.0, 4.0]

            ]
        
        def runs_trajectory_initialization_cases(cases, steps, reward_functions):    
            for z in range(len(reward_functions)):
                Reward = np.zeros((cases*initializations, steps))
                for i in range(initializations):
                    for j in range(cases):
                        print(z, i, j)
                        name_suff = ('Seed_%i_Case_%.2E_Rf_%i'%(seed[i], env_hermite.actions[j], z))
                        reward = run_trajectory(seed = seed[i], action = j, env = env_hermite,\
                                    name_suffix = name_suff, steps = steps, reward_f = reward_functions[z])
                        Reward[j+(cases*i), :] = reward
                np.save(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_multiple_%i'%z, Reward)
            
        def runs_trajectory_initialization_cases_random_action(cases, steps, reward_functions):    
            for z in range(len(reward_functions)):
            # for z in [5, 6]:
                Reward = np.zeros((initializations, steps))
                for i in range(initializations):
                    print(z, i)
                    name_suff = ('Seed_%i_Rf_%i'%(seed[i], z))
                    reward = run_trajectory(seed = seed[i], action = 'random', env = env_hermite,\
                                name_suffix = name_suff, steps = steps, reward_f = reward_functions[z])
                    Reward[i, :] = reward
                np.save(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_multiple_%i'%z, Reward)
# 
        runs_trajectory_initialization_cases(cases, steps, reward_functions)
        plot_rewards2(values, initializations, reward_functions[0:4], steps = steps, env = env_hermite)
        plot_one_reward(values, initializations, reward_functions, 0,  steps = steps, env = env_hermite)
        
        runs_trajectory_initialization_cases_random_action(cases, steps, reward_functions)
        plot_rewards_random_action(values, initializations, reward_functions, steps = steps, env = env_hermite)

    elif experiment == 2:
        # Plot training results
        a = IntegrateEnv_Hermite()
        reward, EnergyError, HuberLoss = load_reward(a)
        plot_reward(a, reward, EnergyError, HuberLoss)
        
    elif experiment == 3:
        # plot trajectory with hermite for all combinations of actions, one initialization
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "0_AllActions/"
        env_hermite.subfolder = name_subfolder

        cases = len(env_hermite.actions)
        steps = 100
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        seed = 0
        def runs_trajectory_cases(cases, steps):
            Reward = np.zeros((cases, steps))
            for j in range(cases):
                name_suff = ('_Case_%.2E'%(env_hermite.actions[j]))
                reward = run_trajectory(seed = seed, action = j, env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
                Reward[j, :] = reward
            np.save(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward', Reward)
            return Reward
                
        runs_trajectory_cases(cases, steps)
        plot_steps(values, steps = steps, env = env_hermite, RL = False, reward_plot = True)

    elif experiment == 4:
        ##########################################
        # Run all possibilities with Hermite vs RL
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "3_AllActionsRL/"
        env_hermite.subfolder = name_subfolder

        cases = len(env_hermite.actions)
        steps = 300
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        def runs_trajectory_cases(cases, steps):
            Reward = np.zeros((cases+1, steps)) # RL
            for j in range(cases):
                name_suff = ('_Case_%.2E'%(env_hermite.actions[j]))
                reward = run_trajectory(seed = seed, action = j, env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
                Reward[j, :] = reward
            
            name_suff = '_Case_RL'
            reward = run_trajectory(seed = seed, action = 'RL', env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
            Reward[-1, :] = reward
            np.save(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward', Reward)
            return Reward
                
        runs_trajectory_cases(cases, steps)
        plot_steps(values, steps = steps, env = env_hermite, RL = True, reward_plot = True)
        plt.show()

    elif experiment == 5:
        ##########################################
        # See reward for different reward functions
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "4_AllActionsRL_reward/"
        env_hermite.subfolder = name_subfolder

        reward_functions = [
                [1, 1.0, 1.0, 1.0],
                [2, 10.0, 10.0, 4.0],
                [2, 100.0, 10.0, 4.0],
                [3, 1.0, 1.0, 4.0],
                [4, 1.0, 100.0, 1.0],
            ]

        cases = len(env_hermite.actions)
        steps = 100
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        def runs_trajectory_cases(cases, steps):
            for z in range(len(reward_functions)):
                Reward = np.zeros((cases, steps))
                for j in range(cases):
                    name_suff = ('_Case_%.2E_%i'%(env_hermite.actions[j], z))
                    reward = run_trajectory(seed = seed, action = j, env = env_hermite,\
                                name_suffix = name_suff, steps = steps, reward_f = reward_functions[z])
                    Reward[j, :] = reward
                
                np.save(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward_'+ str(z), Reward)
                
        runs_trajectory_cases(cases, steps)
        plot_steps_reward(values, reward_functions, steps = steps, env = env_hermite, RL = True, reward_plot = True)

    elif experiment == 6:
        ##########################################
        # Plot final energy error and computing time
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "5_EvsTcomp/"
        env_hermite.subfolder = name_subfolder

        cases = len(env_hermite.actions)
        steps = 100
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        initializations = 500
        seed = np.arange(initializations)
        def runs_trajectory_cases(cases, steps):
            for i in range(initializations):
                for j in range(cases):
                    name_suff = ('_Case_%.2E_%i'%(env_hermite.actions[j], i))
                    reward = run_trajectory(seed = seed[i], action = j, env = env_hermite,\
                                name_suffix = name_suff, steps = steps)
                    
                name_suff = '_Case_RL_%i'%i
                reward = run_trajectory(seed = seed[i], action = 'RL', env = env_hermite,\
                                name_suffix = name_suff, steps = steps)
                
        runs_trajectory_cases(cases, steps)
        plot_EvsTcomp(values, initializations, steps = steps, env = env_hermite, RL = True, reward_plot = True)

    elif experiment == 7:
        ################################################
        # Compare for different integrators
        I = ['Hermite', 'Huayno', 'Symple']

        ENV = list()
        for i in range(len(I)):
            env = IntegrateEnv_multiple(integrator = I[i])
            name_subfolder = I[i]+'/'
            env.subfolder = name_subfolder

            ENV.append(env)

        model_path = env.settings['Training']['savemodel'] + 'model_weights' +str(2100)+ '.pth'
        
        def runs_trajectory_cases(steps):
                for i in range(len(I)):
                    name_suff = '_'+I[i] 
                    
                    reward = run_trajectory(seed = 0, action = 'RL', env = ENV[i],\
                                    name_suffix = name_suff, steps = steps, model_path= model_path, \
                                    steps_suffix= I[i])
                    
        def runs_trajectory_fixed(steps, initializations, seed):
            for j in range(initializations):
                for i in range(len(I)):
                    name_suff = ('Accurate_%s_%i'%(I[i], j))
                    print(I[i], j, ENV[i].actions)
                    reward = run_trajectory(seed = seed[j], action = 0, env = ENV[i],\
                                    name_suffix = name_suff, steps = steps, model_path= model_path, \
                                    steps_suffix= I[i])
                    name_suff = ('Inaccurate_%s_%i'%(I[i], j))
                    reward = run_trajectory(seed = seed[j], action = 5, env = ENV[i],\
                                    name_suffix = name_suff, steps = steps, model_path= model_path, \
                                    steps_suffix= I[i])
                    ENV[i].close()

        seed = 0
        steps = 300
        runs_trajectory_cases(steps)
        plot_int_comparison(I, steps, ENV)

        steps = 100
        initializations = 100
        seed = np.arange(initializations)
        runs_trajectory_fixed(steps, initializations, seed)
        plot_EvsTcomp_integrators(I, initializations, steps = steps, env = ENV, RL = True, reward_plot = True)


    elif experiment == 8:
        ##########################################
        # Run all possibilities with for different trained nets
        def runs_trajectory_cases(cases, steps, case):
            Reward = np.zeros((cases+1, steps)) # RL
            for j in range(cases):
                name_suff = ('_Case_%.2E'%(env_hermite.actions[j]))
                reward = run_trajectory(seed = seed, action = j, env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
                Reward[j, :] = reward
            
            name_suff = '_Case_RL'
            model_path = env_hermite.settings['Training']['savemodel'] + 'model_weights' +str(case)+ '.pth'
            reward = run_trajectory(seed = seed, action = 'RL', env = env_hermite,\
                               name_suffix = name_suff, steps = steps, model_path= model_path)
            Reward[-1, :] = reward
            np.save(env_hermite.settings['Integration']['savefile'] + env_hermite.subfolder + '_reward', Reward)
            return Reward
        
        CASES  = [800, 1000, 1500, 2000]
        for C in CASES:
            env_hermite = IntegrateEnv_Hermite()
            name_subfolder = "7_automatic_steps/"+str(C)+'/'
            env_hermite.subfolder = name_subfolder

            cases = len(env_hermite.actions)
            steps = 300
            values = np.arange(cases)
            env_hermite.settings['Integration']['check_step'] = 1e-1
            seed = 6
                
            runs_trajectory_cases(cases, steps, C)
            save_path_fig = './HermiteIntegration_runs/'+ "7_automatic_steps/"+'Hermite_comparison_RLsteps_s'+str(seed) +'_'+str(C)+'.png'
            plot_steps(values, steps = steps, env = env_hermite, RL = True, reward_plot = True, 
                       save_path = save_path_fig)




