import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json

import torch
import torchvision.models as models
import gym

from Cluster.cluster_2D.envs.HermiteIntegration_env import IntegrateEnv_Hermite
from TrainingFunctions import DQN, load_reward, plot_reward
from PlotsSimulation import load_state_files, plot_planets_trajectory, \
                            run_trajectory, plot_evolution,\
                            plot_actions_taken, plot_planets_distance


def evaluate_many_hermite_cases(values, \
                        plot_tstep = True, \
                        plot_grid = True,
                        plot_errorvstime = True,
                        subfolder = None):
    cases = len(values)
    a = IntegrateEnv_Hermite()
    a.subfolder = subfolder

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
    
    if plot_grid == True: # TODO: does not include RL
        fig, ax = plt.subplots(2, 1, layout = 'constrained', figsize = (10, 10))

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

        for i in range(cases-2): #Excluding the case with the mixed actions
            ax[0].scatter(E_E[-1, i], T_c[-1, i], marker = markers[i//n_tstep_cases], color = colors[i%n_tstep_cases], s = 100, label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
            ax[1].scatter(E_M[-1, i], T_c[-1, i], marker = markers[i//n_tstep_cases], color = colors[i%n_tstep_cases], s = 100, label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
        
        # plot Random actions
        ax[0].scatter(E_E[-1, -2], T_c[-1, -2], marker = markers[(i+1)//n_tstep_cases], color = colors[(i+1)%n_tstep_cases], s = 100, label = 'Random choice')
        ax[1].scatter(E_M[-1, -2], T_c[-1, -2], marker = markers[(i+1)//n_tstep_cases], color = colors[(i+1)%n_tstep_cases], s = 100, label = 'Random choice')
        
        # plot RL
        ax[0].scatter(E_E[-1, -1], T_c[-1, -1], marker = markers[(i+2)//n_tstep_cases], color = colors[(i+2)%n_tstep_cases], s = 100, label = 'RL')
        ax[1].scatter(E_M[-1, -1], T_c[-1, -1], marker = markers[(i+2)//n_tstep_cases], color = colors[(i+2)%n_tstep_cases], s = 100, label = 'RL')

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

    
def plot_steps(values, env = None, steps = None, RL = False, reward_plot = False):
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
    labelsize = 18
    linewidth = 2.5

    if RL == True: 
        plots_v = 11
    else: 
        plots_v = 9
    if reward_plot == True:
        plots_v += 2

    if RL == True:
        fig = plt.figure(figsize = (12, 16))
    else:
        fig = plt.figure(figsize = (12, 18))
    gs1 = matplotlib.gridspec.GridSpec(plots_v, 2, figure = fig, left=0.08, wspace=0.3, 
                                       hspace = 1.7, right = 0.99,
                                        top = 0.98, bottom = 0.1)
    ax1 = fig.add_subplot(gs1[0:3, 0]) # cartesian symple
    ax2 = fig.add_subplot(gs1[0:3, 1]) # cartesian another step
    ax3 = fig.add_subplot(gs1[3:5:, :]) # pairwise distance
    ax4 = fig.add_subplot(gs1[-4:-2, :]) # energy error 
    ax5 = fig.add_subplot(gs1[-2:, :]) # computation time

    if RL == True:
        ax7 = fig.add_subplot(gs1[5:7, :]) # actions
    if reward_plot == True:
        ax6 = fig.add_subplot(gs1[-6:-4, :]) # reward
        ax6.set_yscale('symlog')

    x_axis = np.arange(0, steps, 1)

    # Plot cartesian
    name_planets = np.arange(np.shape(state)[1]).astype(str)    
    
    # Do it for RL
    plot_planets_trajectory(ax1, state[0], name_planets, steps = steps)
    plot_planets_distance(ax3, x_axis, state[0], name_planets, steps = steps, labelsize = labelsize-2)
    
    if RL == True:
        plot_planets_trajectory(ax2, state[-1], name_planets, steps = steps)
        ax2.set_title(r"RL-variable $\mu$")
    else:
        plot_planets_trajectory(ax2, state[-1], name_planets, steps = steps)
        ax2.set_title(r"$\mu = %.2E$"%(env_hermite.actions[-1]))

    # Plot energy errors
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]

    for i in range(cases): # plot all combinations with constant dt and order
        # label = 'O%i, t_step = %1.1E'%(env.actions[values[i][0]][0], env.actions[values[i][0]][1])
        if RL == True and i == cases-1:
            label = 'RL'
            linestyle = '-'
        else:
            label = r"$%i: \mu$ =%.2E"%(i, env.actions[i])
            linestyle = lines[i%len(lines)]
        plot_evolution(ax4, x_axis, E_E[:, i], label = label, colorindex = i, linestyle = linestyle, linewidth = linewidth)
        plot_evolution(ax5, x_axis, T_c[:, i], label = label, colorindex = i, linestyle = linestyle, linewidth = linewidth)
    
        if reward_plot == True:
            plot_evolution(ax6, x_axis, Reward[i, :], label = label, colorindex = i, linestyle = linestyle, linewidth = linewidth)
    

    if RL == True:
    # plot RL and actions
        label = 'RL'
        plot_actions_taken(ax7, x_axis, steps_taken)
        ax7.set_ylim([-1, cases])
        ax7.set_ylabel('Action taken', fontsize = labelsize)
        ax7.set_yticks(np.arange(0, cases-1))
        
    ax3.set_ylabel(r'$\vert\vert \vec r_i - \vec r_j\vert\vert$', fontsize = labelsize)
    ax4.set_ylabel('Energy error',  fontsize = labelsize)
    ax5.set_xlabel('Step', fontsize = labelsize)
    ax5.set_ylabel('Comp. time (s)', fontsize = labelsize)

    ax1.set_title("Tstep_param = %.2E"%(env_hermite.actions[0]))


    ax4.set_yscale('log')
    ax6.set_ylabel('Reward',  fontsize = labelsize)


    for ax_i in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax_i.tick_params(axis='both', which='major', labelsize=labelsize-2)
    
    ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), \
                       fancybox = True, ncol = 4, fontsize = labelsize-2)
    
    if RL == True:
        plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_RLsteps.png', dpi = 100)
    else:
        plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_steps.png', dpi = 100)
    plt.show()
    

def plot_rewards(values, initializations, reward_functions, env = None, steps = None):
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
        # ax.tricontour(x, y, z)

        ax.set_xlabel('Energy error',  fontsize = labelsize)
        ax.set_ylabel('Reward',  fontsize = labelsize)
        ax.set_xscale('log')
        if reward_case == 5 or reward_case == 6:
            ax.set_yscale('symlog')
        # ax.set_yscale('symlog', linthresh = 5)
        # ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    fig.colorbar(sc, axbar)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards.png', dpi = 100, layout = 'tight' )
    plt.show()
    
def plot_rewards2(values, initializations, reward_functions, env = None, steps = None):
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    if steps == None:
        steps = len(cons[0][:,0])

    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(4, 3, width_ratios=[1,1,0.05],
                                    left=0.08, wspace=0.2, hspace = 0.5, right = 0.93,
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
                Delta_EE[i*cases+j, :-1] = (np.log10(abs(cons_i[1:, 1])) - np.log10(abs(cons_i[:-1, 1])))  # relative energy error minus previous step
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
        # ax.tricontour(x, y, z)

        ax.set_xlabel(r'Energy error ($\Delta E_i$)',  fontsize = labelsize-1)
        ax2.set_xlabel(r'Energy error step difference ($\Delta E_i- \Delta E_{i-1}$)',  fontsize = labelsize-1)
        ax.set_ylabel(r'Reward ($R$)',  fontsize = labelsize-1)
        ax.set_xscale('log')
        # ax2.set_xscale('log')
        # if reward_case == 5 or reward_case == 6:
        #     ax.set_yscale('symlog')
        # ax.set_yscale('symlog', linthresh = 5)
        # ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    cbar = fig.colorbar(sc, axbar)
    cbar.set_label(r'Time step parameter ($\mu$)', fontsize = labelsize)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards.png', dpi = 100, layout = 'tight' )
    plt.show()

def plot_one_reward(values, initializations, reward_function, reward_case, env = None, steps = None):
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
            Delta_EE[i*cases+j, :-1] = (np.log10(abs(cons_i[1:, 1])) - np.log10(abs(cons_i[:-1, 1])))  # relative energy error minus previous step
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
    ax = fig.add_subplot(gs1[0, 0]) 
    ax2 = fig.add_subplot(gs1[1, 0]) 
    # ax.set_title(Reward_name, fontsize = labelsize)

    x = E_E
    x2 = Delta_EE
    y = Reward
    z = action
    sc = ax.scatter(x, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())
    ax2.scatter(x2, y, c = z, cmap = cm, s = 10, norm=matplotlib.colors.LogNorm())
    # ax.tricontour(x, y, z)

    # ax.set_xlabel(r'Energy error ($\Delta E_i$)',  fontsize = labelsize)
    # ax2.set_xlabel(r'Energy error step difference ($\log(\vert Delta E_i\vert )- log(\vert \Delta E_{i-1}\vert )$)',  fontsize = labelsize)
    ax.set_xlabel(r'$\Delta E_i$',  fontsize = labelsize)
    ax2.set_xlabel(r'$\log(\vert Delta E_i\vert )- log(\vert \Delta E_{i-1}\vert )$',  fontsize = labelsize)
    ax.set_ylabel(r'Reward ($R$)',  fontsize = labelsize)
    ax2.set_ylabel(r'Reward ($R$)',  fontsize = labelsize)
    ax.set_title(Reward_name, fontsize = labelsize, y = 1.05)
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=labelsize-3)
    ax.tick_params(axis='y', labelsize=labelsize-3)
    ax2.tick_params(axis='x', labelsize=labelsize-3)
    ax2.tick_params(axis='y', labelsize=labelsize-3)
    ax.set_ylim([-5.0, 10.0])
    ax2.set_xlim([-1.0, 1.0])
    ax2.set_ylim([-10.0, 10.0])
    # ax2.set_xscale('log')
    # if reward_case == 5 or reward_case == 6:
    #     ax.set_yscale('symlog')
    # ax.set_yscale('symlog', linthresh = 5)
    # ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    cbar = fig.colorbar(sc, axbar)
    cbar.set_label(r'Time step parameter ($\mu$)', fontsize = labelsize)
    cbar.ax.tick_params(labelsize=labelsize-3)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards_1.png', dpi = 100)
    plt.show()


def plot_rewards_random_action(values, initializations, reward_functions, env = None, steps = None):
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
        # ax.tricontour(x, y, z)

        ax.set_xlabel('Energy error',  fontsize = labelsize)
        ax2.set_xlabel('Energy error step difference',  fontsize = labelsize)
        ax.set_ylabel('Reward',  fontsize = labelsize)
        ax.set_xscale('log')
        ax2.set_xscale('log')
        # if reward_case == 5 or reward_case == 6:
        #     ax.set_yscale('symlog')
        # ax.set_yscale('symlog', linthresh = 5)
        # ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    fig.colorbar(sc, axbar)

    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_rewards.png', dpi = 100, layout = 'tight' )
    plt.show()

def plot_runs_trajectory(cases, action, steps, seeds, env = None):
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

    # name_planets = env.names
    name_planets = (np.arange(np.shape(state)[1])+1).astype(str)
    fig, axes = plt.subplots(nrows=int(cases//2), ncols= 2, layout = 'constrained', figsize = (8, 12))

    for i, ax in enumerate(axes.flat):
        if i == 0:
            legend_on = True
        else:
            legend_on = False
        plot_planets_trajectory(ax, state[i], name_planets, labelsize = label_size, steps = steps, legend_on = legend_on)
        ax.set_title("Seed = %i"%(seeds[i]), fontsize = label_size+2)
        
        #     ax.legend(fontsize = label_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size-4, labelrotation=0)

    plt.axis('equal')
    plt.savefig('HermiteIntegration_runs/' + env.subfolder + 'plot_state_action%i'%action, dpi = 100)
    plt.show()
    
def plot_steps_reward(values, rewards, env = None, steps = None, RL = False, reward_plot = False):
    cases = len(values)
    if env == None:
        env = IntegrateEnv_Hermite()

    fig = plt.figure(figsize = (10,11))
    gs1 = matplotlib.gridspec.GridSpec(len(rewards)+ 1, 1, figure = fig, left=0.08, wspace=0.3, 
                                       hspace = 0.7, right = 0.99,
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
            
        ax1.set_yscale('symlog')

        # Plot energy errors
        lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]

        for i in range(cases): # plot all combinations with constant dt and order
            # label = 'O%i, t_step = %1.1E'%(env.actions[values[i][0]][0], env.actions[values[i][0]][1])
            label = "Tstep parameter %.2E"%(env.actions[i])
            linestyle = lines[i%len(lines)]
            # plot_evolution(ax4, x_axis, E_E[:, i], label = label, colorindex = i, linestyle = linestyle)
            plot_evolution(ax1, x_axis[1:-1], Reward[i, 1:-1], label = label, colorindex = i, linestyle = linestyle)
        
        if z == len(rewards)-1:
            ax1.set_xlabel('Step', fontsize = labelsize)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.52), \
                       fancybox = True, ncol = cases//2+cases%2, fontsize = labelsize)

        ax1.set_ylabel(r'$\vert\vert fr_i - r_j\vert\vert$')
            
        # ax[1].set_ylabel('Angular momentum error', fontsize = labelsize)
        # ax1.set_xlabel('Step', fontsize = labelsize)
        ax1.set_ylabel('Reward',  fontsize = labelsize)
        ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig('./HermiteIntegration_runs/'+ env.subfolder+'Hermite_comparison_steps_rewards.png', dpi = 100)
    plt.show()
    

if __name__ == '__main__':
    experiment = 4
    
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
                
        # runs_trajectory_initializations(cases, action, steps)
        cases = 6 # only plot x
        plot_runs_trajectory(cases, action, steps, seeds, env_hermite)
        
    elif experiment == 1: 
        # plot reward function comparison for different actions and initializations
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "2_RewardStudy/"
        env_hermite.subfolder = name_subfolder

        cases = len(env_hermite.actions)
        steps = 50
        initializations = 10
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        seed = np.arange(initializations)
        reward_functions = [
                [1, 1.0, 1.0, 1.0],
                [2, 1.0, 100.0, 10.0],
                [2, 1.0, 10.0, 4.0],
                [3, 1.0, 100.0, 5.0],
                [4, 1.0, 100.0, 1.0]
            ]
        
        def runs_trajectory_initialization_cases(cases, steps, reward_functions):    
            for z in range(len(reward_functions)):
            # for z in [5, 6]:
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

        runs_trajectory_initialization_cases(cases, steps, reward_functions)
        plot_rewards2(values, initializations, reward_functions[0:4], steps = steps, env = env_hermite)
        plot_one_reward(values, initializations, reward_functions, 4,  steps = steps, env = env_hermite)
        
        # runs_trajectory_initialization_cases_random_action(cases, steps, reward_functions)
        # plot_rewards_random_action(values, initializations, reward_functions, steps = steps, env = env_hermite)

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
        # Run all possibilities with Symple vs RL
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "3_AllActionsRL/"
        env_hermite.subfolder = name_subfolder


        cases = len(env_hermite.actions)
        steps = 100
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        seed = 7
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


    elif experiment == 5:
        ##########################################
        # See reward for different reward functions
        env_hermite = IntegrateEnv_Hermite()
        name_subfolder = "4_AllActionsRL_reward/"
        env_hermite.subfolder = name_subfolder

        reward_functions = [
                [1, 1.0, 1.0, 1.0],
                [2, 1.0, 100.0, 10.0],
                [2, 1.0, 10.0, 4.0],
                [3, 1.0, 100.0, 5.0],
                # [4, 1.0, 100.0, 5.0],
                [4, 1.0, 100.0, 1.0]
                # [4, 1.0, 0.0, 0.0]
            ]

        cases = len(env_hermite.actions)
        steps = 100
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        seed = 6
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





