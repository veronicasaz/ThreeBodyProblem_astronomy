"""
PlotsSimulation: plotting functions

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from PlotsFunctions import plot_planets_trajectory, plot_evolution

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def calculate_errors(states, cons, tcomp):
    cases = len(states)
    steps = np.shape(cons[0][:, 0])[0]

    # Calculate the energy errors
    R = np.zeros((steps, cases))
    E = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    Action = np.zeros((steps, cases))
    for i in range(cases):
        R[1:, i] = cons[i][1:steps, 1]
        E[1:, i] = abs(cons[i][1:steps, 2]) # absolute relative energy error
        T_c[1:, i] = np.cumsum(tcomp[i][1:steps]) # add individual computation times
        Action[1:, i] = cons[i][1:steps, 0]

    return E, T_c, R, Action


def plot_initializations(state, cons, tcomp, names, save_path, seed):
    # Setup plot
    title_size = 20
    label_size = 18
    rows = 3
    columns = 2
    fig, axes = plt.subplots(nrows=rows, ncols= columns, \
                             layout = 'constrained', figsize = (8, 12))
    
    name_planets = (np.arange(np.shape(state)[1])+1).astype(str)
    steps = np.shape(cons)[1]
    print(steps)
    for i, ax in enumerate(axes.flat):
        if i == 0:
            legend_on = True
        else:
            legend_on = False
        plot_planets_trajectory(ax, state[i]/1.496e11, name_planets, labelsize = label_size, steps = steps, legend_on = legend_on)
        ax.set_title("Seed = %i"%(seed[i]), fontsize = title_size)
        
        ax.tick_params(axis='both', which='major', labelsize=label_size-4, labelrotation=0)

    plt.axis('equal')
    plt.savefig(save_path, dpi = 100)
    plt.show()


###########################
# TODOOO ##################
###########################
def plot_reward_comparison(env, STATES, CONS, TCOMP, Titles, save_path, R):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[0, :])
    ax3 = fig.add_subplot(gs1[1, :])
    ax4 = fig.add_subplot(gs1[2, :])
    ax5 = fig.add_subplot(gs1[3, :])
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax5, x_axis, R[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-3)

    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_yscale('log')

    ax5.set_xlabel('Step', fontsize = label_size)

    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Energy Error Local', fontsize = label_size)
    ax4.set_ylabel('Computation time (s)', fontsize = label_size)
    ax5.set_ylabel('Reward', fontsize = label_size)
    

    plt.savefig(save_path, dpi = 150)
    plt.show()

###########################
# TODOOO ##################
###########################



###########################
# TODOOO ##################
###########################
def plot_comparison_end(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    legend = True

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    x_axis = np.arange(1, len(T_comp), 1)

    ax1 = fig.add_subplot(gs1[0, 0])
    ax12 = fig.add_subplot(gs1[0, 1])
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    ax4 = fig.add_subplot(gs1[3, :])
    for case in range(len(STATES)):
        print(T_comp[-1, case], Energy_error[-1, case])
        ax1.scatter(T_comp[-1, case], Energy_error[-1, case], label = Titles[case][1:], \
                    color = colors[(case+2)%len(colors)])
        ax12.scatter(T_comp[-1, case], Energy_error_local[-1, case], label = Titles[case][1:], \
                    color = colors[(case+2)%len(colors)])
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
    
    for ax in [ax1, ax12, ax2, ax3, ax4]:
        ax.set_yscale('log')

    ax1.set_xscale('log')
    ax12.set_xscale('log')

    ax1.legend()

    ax4.set_xlabel('Step', fontsize = label_size)

    ax1.set_ylabel('Energy Error', fontsize = label_size)
    ax12.set_ylabel('Energy Error Local', fontsize = label_size)
    ax1.set_xlabel('Computation time (s)', fontsize = label_size)
    ax12.set_xlabel('Computation time (s)', fontsize = label_size)

    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Energy Error Local', fontsize = label_size)
    ax4.set_ylabel('Computation time (s)', fontsize = label_size)
    
    ax2.legend(fontsize = label_size -3)

    plt.savefig(save_path, dpi = 150)
    plt.show()


###########################
# TODOOO ##################
###########################
def plot_energy_vs_tcomp(env, STATES, cons, tcomp, Titles, initializations, save_path, plot_traj_index = [0,1,2]):
    """
    plot_EvsTcomp: plot energy error vs computation time for different cases
    INPUTS:
        values: list of the actions taken for each case
        initializations: number of initializations used to generate the data
        steps: steps taken
        env: environment used
    """
    cases = len(STATES)

    fig = plt.figure(figsize = (6,6))
    gs1 = matplotlib.gridspec.GridSpec(2, 2, figure = fig, width_ratios = (3, 1), height_ratios = (1, 3), \
                                       left=0.15, wspace=0.3, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    
    msize = 50
    alphavalue = 0.5
    alphavalue2 = 0.9
    
    ax1 = fig.add_subplot(gs1[1, 0]) 
    ax2 = fig.add_subplot(gs1[0, 0])
    ax3 = fig.add_subplot(gs1[1, 1])

    markers = ['o', 'x', 's', 'o', 'x']
    order = [1,2,0, 3, 4, 5, 6]
    alpha = [0.5, 0.5, 0.9, 0.8, 0.9]


    # Calculate the energy errors
    E_T = np.zeros((initializations, len(plot_traj_index)))
    E_B = np.zeros((initializations, len(plot_traj_index)))
    T_c = np.zeros((initializations, len(plot_traj_index)))
    Labels = np.zeros((initializations, len(plot_traj_index)))
    nsteps_perepisode = np.zeros((initializations, len(plot_traj_index)))
    

    for act in range(len(plot_traj_index)):
        for i in range(initializations):
            nsteps_perepisode = len(cons[plot_traj_index[act]*initializations +i][:,0])
            E_B[i, act] = abs(cons[plot_traj_index[act]*initializations + i][2, -1]) # absolute relative energy error
            E_T[i, act] = abs(cons[plot_traj_index[act]*initializations + i][3, -1]) # absolute relative energy error

            T_c[i, act] = np.sum(tcomp[plot_traj_index[act]*initializations + i])/nsteps_perepisode # add individual computation times
            Labels[i, act] = plot_traj_index[act]

    def minimum(a, b):
        if a <= b:
            return a
        else:
            return b
        
    def maximum(a, b):
        if a >= b:
            return a
        else:
            return b
        
    min_x = 10
    min_y = 10
    max_x = 0
    max_y = 0 #random initial values
    for i in range(len(plot_traj_index)):  
        X = T_c[:, i]
        Y = E_B[:, i]
        ax1.scatter(X, Y, color = colors[i], alpha = alphavalue, marker = markers[i],\
                s = msize, label = Labels[0, i], zorder =order[i])
        min_x = minimum(min_x, min(X))
        min_y = minimum(min_y, min(Y))
        max_x = maximum(max_x, max(X))
        max_y = maximum(max_y, max(Y))
    binsx = np.logspace(np.log10(min_x),np.log10(max_x), 50)
    binsy = np.logspace(np.log10(min_y),np.log10(max_y), 50)  

    for i in range(len(plot_traj_index)):  
        ax2.hist(X, bins = binsx, color = colors[i],  alpha = alpha[i], edgecolor=colors[i], \
                 linewidth=1.2, zorder =order[i])
        ax2.set_yscale('log')
        ax3.hist(Y, bins = binsy, color = colors[i], alpha = alpha[i], orientation='horizontal',\
                 edgecolor=colors[i], linewidth=1.2, zorder =order[i])
    
    labelsize = 12
    ax1.legend(fontsize = labelsize)
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax3.set_yscale('log')
    # ax1.set_xlim([5e-1, 3])
    # ax2.set_xlim([5e-1, 3])
    # ax1.set_ylim([1e-14, 1e1])
    # ax3.set_ylim([1e-14, 1e1])
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig(save_path+'tcomp_vs_Eerror_evaluate.png', dpi = 100)
    plt.show()