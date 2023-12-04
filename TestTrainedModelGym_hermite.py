import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json

import torch
import torchvision.models as models
import gym

from Cluster.cluster_2D.envs.HermiteIntegration_env import IntegrateEnv
from TrainingFunctions import DQN, load_reward, plot_reward
from PlotsSimulation import load_state_files, plot_planets_trajectory, \
                            run_trajectory, plot_evolution,\
                            plot_actions_taken


def evaluate_many_hermite_cases(values, \
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

def plot_steps(values, env = None, steps = None, RL = False):
    cases = len(values)
    if env == None:
        env = IntegrateEnv()

    #############################################
    # Load run information for symple cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = 'Case_%.2E'%(env_hermite.actions[i]))
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_%i'%i)
    
    if RL == True:
        # Load RL run
        cases += 1
        state_i, cons_i, tcomp_i = load_state_files(env, steps, namefile = 'Case_RL')
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_RL')
        steps_taken = np.load(env.settings['Integration']['savefile'] +'RL_steps_taken.npy', allow_pickle=True)

    # print(cons[-1][:, 1])
    # print("=========================")
    # a = np.load("./SympleIntegration_runs/state_consCase_RL.npy")
    # print(a[0:steps, 1])
    # a = np.load("./SympleIntegration_runs/state_consCase_8.npy")
    # print(a[0:steps, 1])

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

    # Take difference with respect to the best one
    # for i in range(cases):
    #     T_c[:, i] /= T_c[:, 0]

    #############################################
    # plot
    fig = plt.figure(figsize = (10,10))
    if RL == True: 
        gs1 = matplotlib.gridspec.GridSpec(7, 2, left=0.08, wspace=0.1, hspace = 0.5,  right = 0.99)
        ax1 = fig.add_subplot(gs1[0:2, 0]) # cartesian symple
        ax2 = fig.add_subplot(gs1[0:2, 1]) # cartesian RL
        ax3 = fig.add_subplot(gs1[2, :]) # actions
        ax4 = fig.add_subplot(gs1[3:5, :]) # energy error 
        ax5 = fig.add_subplot(gs1[5:, :]) # computation time
    else:
        gs1 = matplotlib.gridspec.GridSpec(6, 2, left=0.08, wspace=0.1, hspace = 0.5, right = 0.99)
        ax1 = fig.add_subplot(gs1[0:2, 0]) # cartesian symple
        # ax2 = fig.add_subplot(gs1[0:2, 1]) # cartesian RL
        ax4 = fig.add_subplot(gs1[2:4, :]) # energy error 
        ax5 = fig.add_subplot(gs1[4:, :]) # computation time

    x_axis = np.arange(0, steps, 1)

    # Plot cartesian
    name_planets = np.arange(np.shape(state)[1]).astype(str)
    print(name_planets)
    
    
    # Do it for RL
    plot_planets_trajectory(ax1, state[0], name_planets, steps = steps)
    ax1.set_title("Tstep_param = %.2E"%(env_hermite.actions[0]))
    
    if RL == True:
        plot_planets_trajectory(ax2, state[-1], name_planets, steps = steps)


    # Plot energy errors
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]

    for i in range(cases): # plot all combinations with constant dt and order
        # label = 'O%i, t_step = %1.1E'%(env.actions[values[i][0]][0], env.actions[values[i][0]][1])
        if RL == True and i == cases-1:
            label = 'RL'
            linestyle = '-'
        else:
            label = "Tstep parameter %.2E"%(env.actions[i])
            linestyle = lines[i%cases]
        plot_evolution(ax4, x_axis, E_E[:, i], label = label, colorindex = i, linestyle = linestyle)
        plot_evolution(ax5, x_axis, T_c[:, i], label = label, colorindex = i, linestyle = linestyle)
    
    labelsize = 10

    if RL == True:
    # plot RL and actions
        label = 'RL'
        plot_actions_taken(ax3, x_axis, steps_taken)
        ax3.set_ylabel('Action taken', fontsize = labelsize)
        ax2.set_xlabel('x (au)', fontsize = labelsize)
        ax2.set_ylabel('y (au)', fontsize = labelsize)
    
    ax1.set_xlabel('x (au)', fontsize = labelsize)
    ax1.set_ylabel('y (au)', fontsize = labelsize)
    ax4.legend(loc='upper right')
    ax4.set_ylabel('Energy error',  fontsize = labelsize)
    # ax[1].set_ylabel('Angular momentum error', fontsize = labelsize)
    ax5.set_xlabel('Step', fontsize = labelsize)
    ax5.set_ylabel('Comp. time (s)', fontsize = labelsize)
    ax4.set_yscale('log')
    # ax5.set_yscale('log')
    ax4.tick_params(axis='both', which='major', labelsize=labelsize)
    ax5.tick_params(axis='both', which='major', labelsize=labelsize)
    
    if RL == True:
        plt.savefig('./HermiteIntegration_runs/Hermite_comparison_RLsteps.png', dpi = 100)
    else:
        plt.savefig('./HermiteIntegration_runs/Hermite_comparison_steps.png', dpi = 100)
    plt.show()
    


def plot_runs_trajectory(cases, action, steps, seeds):
    env = IntegrateEnv()

    # Load run information for Hermite cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        state_i, cons_i, tcomp_i = load_state_files(env, namefile = '_traj_action_%i_initialization_%i'%(action, i))
        state.append(state_i)
        cons.append(cons_i)
        tcomp.append(tcomp_i)
        name.append('Case_%i'%i)

    # plot
    label_size = 20
    name_planets = env.names
    fig, axes = plt.subplots(nrows=int(cases//3), ncols= 3, layout = 'constrained', figsize = (10, 10))

    for i, ax in enumerate(axes.flat):
        plot_planets_trajectory(ax, state[i], name_planets, labelsize = label_size, steps = steps)
        if i == 0:
            ax.legend(fontsize = 10)
            ax.title("Seed = %i"%seeds[i], fontsize = label_size)

    plt.axis('equal')
    plt.savefig('HermiteIntegration_runs/plot_state_action%i'%action)
    plt.show()
    
if __name__ == '__main__':
    experiment = 0

    
    if experiment == 0: 
        # plot trajectory with hermite for all combinations of actions, one initialization
        env_hermite = IntegrateEnv()
        cases = len(env_hermite.actions)
        steps = 10
        values = np.arange(cases)
        env_hermite.settings['Integration']['check_step'] = 1e-1
        seed = 1
        def runs_trajectory_cases(cases, steps):
            for j in range(cases):
                print("Case %.2E"%env_hermite.actions[j])
                name_suff = ('Case_%.2E'%(env_hermite.actions[j]))
                run_trajectory(seed = seed, action = j, env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
                
        runs_trajectory_cases(cases, steps)
        plot_steps(values, steps = steps, env = env_hermite, RL = False)

    elif experiment == 1: 
        # plot trajectory with hermite for a fixed action, many initializations
        cases = 9
        action = 5
        steps = 500
        hermite = IntegrateEnv()
        hermite.settings['Integration']['check_step'] = 1e-1
        seeds = np.arange(cases)
        def runs_trajectory_initializations(cases, action, steps):
            for j in range(cases):
                print("Case %i"%j)
                name_suff = ('HermiteInitializations/_traj_action_%i_initialization_%i'%(action, j))
                run_trajectory(seed = seeds[j], action = action, env = env_hermite,\
                               name_suffix = name_suff, steps = steps)
                
        runs_trajectory_initializations(cases, action, steps)
        plot_runs_trajectory(cases, action, steps, seeds)

    elif experiment == 2:
        # Plot training results
        a = IntegrateEnv()
        reward, EnergyError = load_reward(a)
        plot_reward(a, reward, EnergyError)

    elif experiment == 3:
        ##########################################
        # Run all possibilities with Symple vs RL
        env_hermite = IntegrateEnv()
        steps = 10 
        values = list()
        seed = 0
        
        print(len(env_hermite.actions))
        for i in range(len(env_hermite.actions)):
            values.append([i]*steps)
        # values.append(np.random.randint(0, len(env_symple.actions), size = steps))

        ###############################################
        # If already run, this can be commented:
        cases = len(values)
        print(cases)
        for j in range(cases):
            print("Hermite simulations", len(values))
            run_trajectory(seed = seed, action = values[j],
                    env = env_hermite, name_suffix = "Case_%i"%j, 
                    steps = steps)

        steps_taken = run_trajectory(seed = seed, action = 'RL',
                    env = env_hermite, name_suffix = 'Case_RL', 
                    steps = steps)
        
        ###############################################
        plot_steps(values, steps = steps, env = env_hermite, RL = True)





