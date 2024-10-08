"""
Plots_TestTrained: plotting functions from the trained models

Author: Veronica Saz Ulibarrena
Last modified: 31-May-2024
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter

import random
from scipy.signal import savgol_filter
from Plots_TestEnvironment import calculate_errors
from PlotsFunctions import plot_planets_trajectory, plot_evolution, \
    plot_planets_distance, plot_actions_taken

# colors = [ 'steelblue', 'deepskyblue', 'mediumseagreen', 'coral',  \
#         'navy', 'mediumslateblue', 'darkgoldenrod','black', 'blue', 'green']
colors = [ 'steelblue','darkgoldenrod' , 'mediumseagreen', 'coral',  \
        'navy', 'mediumslateblue', 'deepskyblue','black', 'blue', 'green']

def plot_durations(episode_rewards, episode, show_result=False):
    """
    plot_durations: plot training progress
    INPUTS:
        episode_rewards: reward for each episode
        episode: episode number
        show_result: if True, shows plot
    https://www.linkedin.com/advice/0/how-do-you-evaluate-performance-robustness-your-reinforcement:
    cumulative reward, the average reward per episode, the number of steps per episode, or the success rate over time.
    """
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    x = np.arange(len(rewards_t.numpy()))
    plt.scatter(x, rewards_t.numpy())
    plt.yscale('symlog', linthresh = 1e-4)
    if episode %50 == 0:
        plt.savefig('./SympleIntegration_training/reward_progress_%i'%episode)


def plot_reward(a, reward, Eerror, HuberLoss):
    """
    plot_reward: plot training parameters such as reward
    INPUTS:
        a: environment
        reward: rewards
        Eerror: energy error
        HuberLoss: huber loss
    """
    episodes = len(reward)-1 #TODO: why is it saving an empty array at the end?
    x_episodes = np.arange(episodes)

    nsteps_perepisode = np.zeros(episodes)
    
    cumul_reward_perepisode = np.zeros(episodes)
    avg_reward_perepisode = np.zeros(episodes)
    
    avg_energy_perepisode = np.zeros(episodes)
    last_energy_perepisode = np.zeros(episodes)
    reward_flat = list()
    energyerror_flat = list()
    episode_end_list = list()

    huberloss_flat = list()
    episode_flat = list()
    energyerror_flat_total = list()

    for i in range(episodes):
        energyerror_flat = energyerror_flat + Eerror[i][1:]
        reward_flat = reward_flat + reward[i][1:]
        episode_flat = episode_flat + [i]*len(reward[i][1:])

        nsteps_perepisode[i] = len(reward[i])
        cumul_reward_perepisode[i] = sum(reward[i][1:])
        avg_reward_perepisode[i] =  sum(reward[i][1:]) / nsteps_perepisode[i]
        avg_energy_perepisode[i] = abs(np.array(Eerror[i][-1])) / nsteps_perepisode[i]
        last_energy_perepisode[i] = abs(Eerror[i][-1])
        
    x_all = np.arange(len(reward_flat))
    
    f, ax = plt.subplots(4, 1, figsize = (8,6))
    plt.subplots_adjust(left=0.07, right=0.97, top=0.95, \
                        bottom=0.1, hspace = 1.2)
    fontsize = 18

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2

    pts = 11
    # ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    y = cumul_reward_perepisode
    ax[0].plot(x_episodes, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)    
    ax[0].plot(x_episodes, yy, color = 'black')
    ax[0].set_title(r'Cumul $R$/episode', fontsize = fontsize)

    # y = avg_reward_perepisode
    y = avg_reward_perepisode
    ax[1].plot(x_episodes,y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[1].plot(x_episodes, yy, color = 'black')
    ax[1].set_title(r'Avg $R$/episode', fontsize = fontsize)
    # ax[1].set_yscale('symlog', linthresh = 1e1)

    y = avg_energy_perepisode
    ax[2].plot(x_episodes, y, color = colors[0], alpha = 1)
    # y = energyerror_flat
    # ax[2].plot(x_all, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[2].plot(x_episodes, yy, color = 'black')
    ax[2].set_title(r'Slope $\Delta E/episode$', fontsize = fontsize)
    ax[2].set_yscale('log')

    y = last_energy_perepisode
    ax[3].plot(x_episodes, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[3].plot(x_episodes, yy, color = 'black')
    ax[3].set_title(r'Final $\Delta E/episode$', fontsize = fontsize)
    ax[3].set_yscale('log')

    for ax_i in ax: 
        ax_i.tick_params(axis='x', labelsize=fontsize-5)
        ax_i.tick_params(axis='y', labelsize=fontsize-5)

    ax[-1].set_xlabel('Episode', fontsize = fontsize-2)
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path + 'cumulative_reward.png', dpi = 100)
    plt.show()

def plot_test_reward(a, test_reward, trainingTime):
    """
    plot_test_reward: plot training parameters taken from the test dataset
    INPUTS:
        a: environment
        test_reward: array with each row for each episode and columns: 
            [Reward, Energy error, Computation time]
    """
    fig, ax = plt.subplots(3, 1, figsize = (10,7))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, \
                        bottom=0.1, hspace = 0.6)
    fontsize = 16

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2
    
    # pts = 11
    # ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    episodes = len(test_reward) 
    # episodes = 1000
    x_episodes = np.arange(episodes-1)
    
    REWARD_avg = []
    REWARD_std = []
    REWARD_min = []
    REWARD_max = []
    EERROR_avg = []
    EERROR_std = []
    EERROR_min = []
    EERROR_max = []
    test_cases = len(np.array(test_reward[0]).reshape((-1, 3)))
    TCOMP_avg = []
    TCOMP_std = []
    TCOMP_min = []
    TCOMP_max = []

    REWARD_all = np.zeros((episodes-1, test_cases))
    EERROR_all = np.zeros((episodes-1, test_cases))
    TCOMP_all = np.zeros((episodes-1, test_cases))
   
    EERROR_jump_avg = []
    EERROR_jump_std= []
    for e in range(episodes-1):
        # print(e)
        reshaped = np.array(test_reward[e]).reshape((-1, 3))
        REWARD_avg.append(np.mean(reshaped[:, 0]))
        REWARD_std.append(np.std(reshaped[:, 0]))
        # REWARD_min.append(min(reshaped[:, 0]))
        # REWARD_max.append(max(reshaped[:, 0]))

        # print(e, test_reward[e])
        EERROR_avg.append(np.mean(np.log10(abs(reshaped[:, 1]))))
        EERROR_std.append(np.std(np.log10(abs(reshaped[:, 1]))))
        # EERROR_min.append(min(np.log10(abs(reshaped[:, 1]))))
        # EERROR_max.append(max(np.log10(abs(reshaped[:, 1]))))
        
        TCOMP_avg.append(np.mean(reshaped[:, 2]))
        TCOMP_std.append(np.std(reshaped[:, 2]))
        # TCOMP_min.append(min(reshaped[:, 2]))
        # TCOMP_max.append(max(reshaped[:, 2]))

        REWARD_all[e, :] = np.array(np.log10(abs(reshaped[:, 0])))
        EERROR_all[e, :] = np.array(np.log10(abs(reshaped[:, 1])))
        TCOMP_all[e, :] = np.array(np.log10(abs(reshaped[:, 2])))

    y = [REWARD_avg, EERROR_avg, TCOMP_avg]
    err = [REWARD_std, EERROR_std, TCOMP_std]
    # err_minmax = [[REWARD_min, REWARD_max], [EERROR_min, EERROR_max], [TCOMP_min, TCOMP_max]]
    for plot in range(3):
        # ax[plot].errorbar(x_episodes, y[plot], e[plot], color = colors[0], \
        #                   alpha = 1, fmt='o')
        y[plot] = np.array(y[plot])
        err[plot] = np.array(err[plot])
        ax[plot].plot(x_episodes, y[plot], color= colors[0], \
                          alpha = 0.2, marker = '.')
        ax[plot].plot(x_episodes, y[plot] + err[plot], color = colors[1], \
                          alpha = 0.2, marker = '.')
        ax[plot].plot(x_episodes, y[plot] - err[plot], color = colors[1], \
                          alpha = 0.2, marker = '.')
        
        # ax[plot].plot(x_episodes, y[plot] + np.array(err_minmax[plot][1]), color = colors[2], \
        #                   alpha = 0.2, marker = '^')
        # ax[plot].plot(x_episodes, y[plot] - np.array(err_minmax[plot][0]), color = colors[2], \
        #                   alpha = 0.2, marker = '^')
        
        
    # ax[0].plot(x_episodes, REWARD_all)
    # ax[1].plot(x_episodes, EERROR_all)
    # ax[2].plot(x_episodes, TCOMP_all)
        
    # ax[-1].plot(x_episodes, trainingTime[:-1], color = colors[0], marker = '.')
    fig.suptitle('Training time: %.2f min'%(trainingTime[-1]/60), y = 0.99, x = 0.23, fontsize = fontsize -3)
    
    def maxN(elements, n):
        a = sorted(elements, reverse=True)[:n]
        index = np.zeros(n)
        for i in range(n):
            print(np.where(elements == a[i])[0])
            index[i] = np.where(elements == a[i])[0][0]
        return index, a
    index, value = maxN(y[0], 5) 
    print(index)
    for i in range(len(index)):
        ax[0].plot([index[i], index[i]], \
                   [min(np.array(REWARD_avg)-np.array(REWARD_std)),\
                    value[i]], linestyle = '-', marker = 'x', linewidth = 2, color = 'red')
    
    for ax_i in ax: 
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize-3)
        ax_i.tick_params(axis='y', labelsize=fontsize-3)

    ax[-1].set_xlabel('Episode', fontsize = fontsize)
    ax[0].set_title('R', fontsize = fontsize)
    # ax[0].set_yscale('symlog', linthresh = 1e1)
    # ax[0].set_ylim((-1e3, 5))
    ax[1].set_title(r'$log_{10}(\vert \Delta E\vert)$', fontsize = fontsize)
    # ax[2].set_title(r'$log_{10}(\vert \Delta E\vert) - log_{10}(\vert \Delta E_{prev}\vert)$', fontsize = fontsize)
    ax[2].set_title(r'$T_{comp}$ (s)', fontsize = fontsize)
    # ax[3].set_title(r'$T_{training}$ (s)', fontsize = fontsize)
    ax[2].set_yscale('symlog', linthresh = 1e-1)
    ax[2].set_yscale('log')

    # For hermite 1
    # ax[0].set_ylim([-10, 2])
    # ax[1].set_ylim([-12, -2])
    # ax[2].set_ylim([-15, -0.5])
    # ax[2].set_ylim([0.0001, 0.003])

    # For hermite 2
    # ax[0].set_ylim([-10, 10])
    # ax[1].set_ylim([-10, 0])
    # ax[2].set_ylim([-15, -0.5])
    # ax[3].set_ylim([0.0001, 0.003])


    # For symple 2
    # ax[0].set_ylim([-40, 5])
    # ax[1].set_ylim([-12, 5])
    # ax[2].set_ylim([-30, -0.8])
    # ax[3].set_ylim([0.0001, 0.05])
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path + 'test_reward.png', dpi = 100)
    plt.show()

def plot_test_reward_multiple(a, TESTREWARD, TRAININGTIME):
    """
    plot_test_reward: plot training parameters taken from the test dataset
    INPUTS:
        a: environment
        test_reward: array with each row for each episode and columns: 
            [Reward, Energy error, Computation time]
    """
    f, ax = plt.subplots(4, 1, figsize = (10,7))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, \
                        bottom=0.1, hspace = 0.6)
    fontsize = 18

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2
    
    # pts = 11
    # ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    for CASE in range(len(TESTREWARD)):
        test_reward = TESTREWARD[CASE]
        trainingTime = TRAININGTIME[CASE]
        episodes = len(test_reward) 
        x_episodes = np.arange(episodes)
        
        REWARD_avg = []
        REWARD_std = []
        EERROR_avg = []
        EERROR_std = []
        TCOMP_avg = []
        TCOMP_std = []
        EERROR_jump_avg = []
        EERROR_jump_std= []
        print(np.shape(test_reward))
        for e in range(episodes):
            reshaped = np.array(test_reward[e]).reshape((-1, 3))
            REWARD_avg.append(np.mean(reshaped[:, 0]))
            REWARD_std.append(np.std(reshaped[:, 0]))

            EERROR_avg.append(np.mean(np.log10(abs(reshaped[:, 1]))))
            EERROR_std.append(np.std(np.log10(abs(reshaped[:, 1]))))

            TCOMP_avg.append(np.mean(reshaped[:, 2]))
            TCOMP_std.append(np.std(reshaped[:, 2]))

        y = [REWARD_avg, EERROR_avg, TCOMP_avg]
        e = [REWARD_std, EERROR_std, TCOMP_std]
        for plot in range(3):
            # ax[plot].errorbar(x_episodes, y[plot], e[plot], color = colors[0], \
            #                   alpha = 1, fmt='o')
            y[plot] = np.array(y[plot])
            e[plot] = np.array(e[plot])
            ax[plot].plot(x_episodes, y[plot], color= colors[CASE+3], \
                            alpha = 1, marker = '.', label = 'Seed %i'%CASE)
            # ax[plot].plot(x_episodes, y[plot] + e[plot], color = colors[0], \
            #                 alpha = 0.2, marker = '.')
            # ax[plot].plot(x_episodes, y[plot] - e[plot], color = colors[0], \
            #                 alpha = 0.2, marker = '.')
            
            
        ax[-1].plot(x_episodes, trainingTime, color = colors[CASE+3], marker = '.')
        ax[0].legend()
    

    ##########################################
    # Format plots
    ##########################################
    def maxN(elements, n):
        a = sorted(elements, reverse=True)[:n]
        index = np.zeros(n)
        for i in range(n):
            print(np.where(elements == a[i])[0])
            index[i] = np.where(elements == a[i])[0][0]
        return index, a
    index, value = maxN(y[0], 5) 
    for i in range(len(index)):
        ax[0].plot([index[i], index[i]], \
                   [min(np.array(REWARD_avg)-np.array(REWARD_std)),\
                    value[i]], linestyle = '-', marker = 'x', linewidth = 2, color = 'red')
    
    for ax_i in ax: 
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize-3)
        ax_i.tick_params(axis='y', labelsize=fontsize-3)

    ax[-1].set_xlabel('Episode', fontsize = fontsize)
    ax[0].set_title('R', fontsize = fontsize)
    ax[1].set_title(r'$log_{10}(\vert \Delta E\vert)$', fontsize = fontsize)
    # ax[2].set_title(r'$log_{10}(\vert \Delta E\vert) - log_{10}(\vert \Delta E_{prev}\vert)$', fontsize = fontsize)
    ax[2].set_title(r'$T_{comp}$ (s)', fontsize = fontsize)
    ax[2].set_yscale('symlog', linthresh = 1e-1)
    ax[2].set_yscale('log')

    # For hermite 1
    # ax[0].set_ylim([-10, 2])
    ax[1].set_ylim([-12, -2])
    ax[2].set_ylim([-15, -0.5])
    ax[2].set_ylim([0.0001, 0.003])

    # For hermite 2
    # ax[0].set_ylim([-10, 4])
    # ax[1].set_ylim([-10, 0])
    # ax[2].set_ylim([-15, -0.5])
    # ax[3].set_ylim([0.0001, 0.003])


    # For symple 2
    # ax[0].set_ylim([-30, 5])
    # ax[1].set_ylim([-12, 5])
    # ax[2].set_ylim([-30, -0.8])
    # ax[3].set_ylim([0.0001, 0.05])
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path + 'test_reward.png', dpi = 100)
    plt.show()

def plot_balance(a, reward, Eerror, tcomp):
    """
    plot_balance: plot training parameters such as reward
    INPUTS:
        a: environment
        reward: rewards
        Eerror: energy error
        tcomp: computation time
    """
    episodes = len(reward)-1
    x_episodes = np.arange(episodes)

    cumul_reward_perepisode = np.zeros(episodes)
    # avg_energy_perepisode = np.zeros(episodes)
    last_energy_perepisode = np.zeros(episodes)
    sum_tcomp_perepisode = np.zeros(episodes)
    nsteps_perpeisode = np.zeros(episodes)

    for i in range(episodes):
        nsteps_perpeisode[i] = len(reward[i])
        cumul_reward_perepisode[i] = sum(reward[i])
        sum_tcomp_perepisode[i] = sum(tcomp[i]) /nsteps_perpeisode[i]
        last_energy_perepisode[i] = abs(Eerror[i][-1])/nsteps_perpeisode[i]

    
    f, ax = plt.subplots(2, 1, figsize = (10,6))
    plt.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.15, hspace = 0.3)
    fontsize = 18
    
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax[0].scatter(sum_tcomp_perepisode[1:], last_energy_perepisode[1:], c = x_episodes[1:], cmap =cm)
    ax[0].set_yscale('log')
    plt.colorbar(sc)
    ax[0].set_ylabel(r'Episode $\Delta E $(s)', fontsize = fontsize)
    ax[0].set_xlabel('Episode avg computation time (s)', fontsize = fontsize)

    sc2 = ax[1].scatter(sum_tcomp_perepisode[1:], cumul_reward_perepisode[1:], c = x_episodes[1:], cmap =cm)
    # ax[1].set_yscale('log')
    # plt.colorbar(sc2)
    ax[1].set_ylabel(r'Reward', fontsize = fontsize)
    ax[1].set_xlabel('Episode avg computation time (s)', fontsize = fontsize)
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path +'tcomp_vs_energy_training.png', dpi = 100)
    plt.show()

def plot_trajs(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    """
    plot_trajs: plot with trajectories, evolution of energy error, reward, actions for a comparison of cases
    INPUTS:
        env: environment for the runs
        states: state information containing particle evolution
        cons: information containing energy errors, rewards and actions taken
        tcomp: information containing the computation times
        titles: label  of each case
        save_path: path to save the figure
        plot_traj_index: indexes of the trajectories to plot on the cartesian plots
    """
    steps = len(TCOMP[0])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    # Setup plot
    fig = plt.figure(figsize = (12,18))
    label_size = 20
    linewidth = 2.5
    linestyle = ['--', '-', '-', '-', '-', '-', '-', '-', '-']
    gs1 = matplotlib.gridspec.GridSpec(11, 2, 
                                    left=0.13, wspace=0.3, 
                                    hspace = 1.7, right = 0.94,
                                    top = 0.97, bottom = 0.14)
    
    ax1 = fig.add_subplot(gs1[0:3, 0]) # cartesian one
    ax2 = fig.add_subplot(gs1[0:3, 1]) # cartesian RL
    ax3 = fig.add_subplot(gs1[3:5, :]) # pairwise distance
    ax4 = fig.add_subplot(gs1[5:7, :]) # actions
    ax5 = fig.add_subplot(gs1[7:9, :]) # reward
    ax6 = fig.add_subplot(gs1[9:, :]) # energy error 

    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    axis = [ax2, ax1]
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, 1] # plot best and worst
    for case_i, case in enumerate(plot_traj_index): 
        plot_planets_trajectory(axis[case_i], STATES[case], name_bodies, \
                labelsize=label_size, steps = env.settings['Integration']['max_steps'], \
                legend_on = True)
        
        axis[case_i].set_title(Titles[case], fontsize = label_size + 2)
        axis[case_i].set_xlabel('x (au)', fontsize = label_size)
        axis[case_i].set_ylabel('y (au)', fontsize = label_size)

    # Plot distance
    x_axis = np.arange(0, steps) * env.settings['Integration']['check_step']
    distance = plot_planets_distance(ax3, x_axis, STATES[1]/1.496e11, name_bodies,\
                                        steps = steps, labelsize = label_size)

    # Plot RL 
    n_actions = env.settings['RL']['number_actions']
    y_axis = np.zeros(n_actions)
    for i in range(len(y_axis)):
        y_axis[i] = env.actions[i]
    ax4.set_ylabel('Action taken', fontsize = label_size)

    # Simple
    plot_actions_taken(ax4, x_axis, action[:, 0])
    ax4.set_ylim([-1, n_actions+1])
    labels = np.arange(n_actions)[::3]
    ax4.set_yticks(np.arange(n_actions)[::3])
    LABELS = []
    for label in labels:
        LABELS.append('%i: %.0E'%(label, env.actions[label]))
    ax4.set_yticklabels(LABELS)
    # Dual axis
    # ax42 = ax4.twinx()
    # ax42.set_yticks(env.actions[::3])
    # ax42.set_yticklabels(y_axis[::3])
    # ax42.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))

    # Plot energy error
    for case in range(len(STATES)):
        if case == 0:
            label = 'RL'
        else:
            label = r"$%i: \mu$ =%.2E"%(case, env.actions[case-1])
        plot_evolution(ax5, x_axis, R[:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case], linewidth = linewidth)
        plot_evolution(ax6, x_axis, Energy_error[:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case], linewidth = linewidth)
        
    ax6.set_xlabel('Time (yr)', fontsize = label_size)

    ax3.set_ylabel(r'$\vert\vert \vec r_i - \vec r_j\vert\vert$ (au)', fontsize = label_size)
    ax5.set_ylabel(r'Reward', fontsize = label_size)
    ax6.set_ylabel(r'Energy error ', fontsize = label_size)

    ax4.set_xlabel('Step', fontsize = label_size)
    ax3.set_yscale('log')
    ax5.set_yscale('symlog', linthresh = 1e-1)
    ax6.set_yscale('log')

    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    ax3.legend(fontsize = label_size -3)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    
    for ax_i in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)


    # Draw vertical lines for close encounters
    X_vert = []
    for D in distance:
        index_x = np.where(D < 1)[0]
        X_vert.append(index_x)
    def flatten(xss):
        return [x for xs in xss for x in xs]
    X_vert = flatten(X_vert) 
    
    for X_D in X_vert:
        for ax_vert in [ax3, ax4, ax5]:
            ax_vert.axvline(x=X_D* env.settings['Integration']['check_step'], ymin=-1.2, ymax=1, c= 'k', lw =2, zorder = 0, \
                    clip_on = False, alpha = 0.2, linestyle = '--')
        for ax_vert in [ax6]:
            ax_vert.axvline(x=X_D* env.settings['Integration']['check_step'], ymin=0, ymax=1, c= 'k', lw =2, zorder = 0,\
                        clip_on = False, alpha = 0.2, linestyle = '--')

    plt.savefig(save_path, dpi = 150)
    plt.show()


def plot_trajs_RL(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    """
    plot_trajs_RL: plot with trajectories, evolution of energy error, reward, actions for a comparison of trained models
    INPUTS:
        env: environment for the runs
        states: state information containing particle evolution
        cons: information containing energy errors, rewards and actions taken
        tcomp: information containing the computation times
        titles: label  of each case
        save_path: path to save the figure
        plot_traj_index: indexes of the trajectories to plot on the cartesian plots
    """
    steps = len(TCOMP[0])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    # Setup plot
    fig = plt.figure(figsize = (12,18))
    label_size = 24
    linewidth = 2.5
    linestyle = ['--', '-', '-', '-', '-', '-', '-', '-', '-']
    gs1 = matplotlib.gridspec.GridSpec(11, 2, 
                                    left=0.13, wspace=0.3, 
                                    hspace = 1.7, right = 0.99,
                                    top = 0.97, bottom = 0.14)
    
    ax1 = fig.add_subplot(gs1[0:3, 0]) # cartesian one
    ax2 = fig.add_subplot(gs1[0:3, 1]) # cartesian RL
    ax3 = fig.add_subplot(gs1[3:5, :]) # pairwise distance
    ax4 = fig.add_subplot(gs1[5:7, :]) # actions
    ax5 = fig.add_subplot(gs1[7:9, :]) # reward
    ax6 = fig.add_subplot(gs1[9:, :]) # energy error 

    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    axis = [ax2, ax1]
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    for case_i, case in enumerate(plot_traj_index): 
        plot_planets_trajectory(axis[case_i], STATES[case], name_bodies, \
                labelsize=label_size, steps = env.settings['Integration']['max_steps'], \
                legend_on = True)
        
        axis[case_i].set_title(Titles[case], fontsize = label_size + 2)
        axis[case_i].set_xlabel('x (au)', fontsize = label_size)
        axis[case_i].set_ylabel('y (au)', fontsize = label_size)

    # Plot distance
    x_axis = np.arange(1, steps)
    distance = plot_planets_distance(ax3, x_axis, STATES[-1][1:,:]/1.496e11, name_bodies,\
                                        steps = steps, labelsize = label_size)

    

    # Plot energy error
    for case in range(len(STATES)):
        plot_evolution(ax5, x_axis, R[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = case, linewidth = linewidth)
        plot_evolution(ax6, x_axis, Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = case, linewidth = linewidth)
        plot_actions_taken(ax4, x_axis, action[1:, case], colorindex = case,  label = Titles[case])
        
    # Plot RL 
    ax4.set_ylim([-1, env.settings['RL']['number_actions']+1])
    ax4.set_ylabel('Action taken', fontsize = label_size)
    ax4.set_yticks(np.arange(0, env.settings['RL']['number_actions']))

    ax6.set_xlabel('Step', fontsize = label_size)

    ax3.set_ylabel(r'$\vert\vert \vec r_i - \vec r_j\vert\vert$ (au)', fontsize = label_size)
    ax5.set_ylabel(r'Reward', fontsize = label_size)
    ax6.set_ylabel(r'Energy error ', fontsize = label_size)

    ax4.set_xlabel('Step', fontsize = label_size)
    ax3.set_yscale('log')
    ax5.set_yscale('symlog', linthresh = 1e-1)
    ax6.set_yscale('log')

    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    ax3.legend(fontsize = label_size -3)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    
    for ax_i in [ax1, ax2, ax3, ax4,  ax5, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)


    # # Draw vertical lines for close encounters
    # X_vert = []
    # for D in distance:
    #     index_x = np.where(D < 1)[0]
    #     X_vert.append(index_x)
    # def flatten(xss):
    #     return [x for xs in xss for x in xs]
    # X_vert = flatten(X_vert)
    
    # for X_D in X_vert:
    #     for ax_vert in [ax3, ax4, ax5]:
    #         ax_vert.axvline(x=X_D, ymin=-1.2, ymax=1, c= 'k', lw =2, zorder = 0, \
    #                 clip_on = False, alpha = 0.2, linestyle = '--')
    #     for ax_vert in [ax6]:
    #         ax_vert.axvline(x=X_D, ymin=0, ymax=1, c= 'k', lw =2, zorder = 0,\
    #                     clip_on = False, alpha = 0.2, linestyle = '--')

    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_energy_vs_tcomp(env, STATES, CONS, TCOMP, Titles, seeds, save_path, RL_number = 1):
    """
    plot_energy_vs_tcomp: plot energy vs computation time at the end of the simulation for many initializations
    INPUTS:
        env: environment for the runs
        states: state information containing particle evolution
        cons: information containing energy errors, rewards and actions taken
        tcomp: information containing the computation times
        titles: label  of each case
        save_path: path to save the figure
    """
    steps = len(TCOMP[0])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    # # Group initializations per action type
    types = len(TCOMP)//len(seeds)
    X = []
    Y = []
    for i in range(types):
        X.append(Energy_error[-1, i*len(seeds):(i+1)*len(seeds)])
        Y.append(np.sum(T_comp[:, i*len(seeds):(i+1)*len(seeds)], axis = 0))

    
    fig = plt.figure(figsize = (8,8))
    gs1 = matplotlib.gridspec.GridSpec(2, 2, figure = fig, width_ratios = (3, 1), height_ratios = (1, 3), \
                                       left=0.18, wspace=0.4, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    
    msize = 30
    alphavalue = 0.5
    alphavalue2 = 0.9
    markers = ['o', 'x', '^', 's', 'o', 's', 'o', 'x', 's', 'o', 's', 'o', 'x', 's', 'o']
    # labels = [i[10] for i in Titles[0:RL_number]]
    labels = Titles
    # for i in range(types-RL_number):
    #     labels.append(r'$\mu$ = %.1E'%env.actions[i])
    # print(labels)

    ax1 = fig.add_subplot(gs1[1, 0]) 
    ax2 = fig.add_subplot(gs1[0, 0])
    ax3 = fig.add_subplot(gs1[1, 1])

    # For hermite
    # binsx = np.logspace(np.log10(2e-2),np.log10(0.4), 50)
    # binsy = np.logspace(np.log10(1e-14),np.log10(1e-1), 50)

    binsx = np.logspace(np.log10(2e-2),np.log10(10.0), 70)
    binsy = np.logspace(np.log10(1e-14),np.log10(1e4), 70)

    # order = [3, 1,0, 2]
    alpha =  [0.5, 0.5, 0.5]+[0.8]*RL_number
    # plot_index = [i for i in range(RL_number)] + [RL_number, (types-RL_number)//2+RL_number-2,(types-RL_number)//2+RL_number+1, (types)-1]
    plot_index = [i for i in range(RL_number)] + [RL_number, (types-RL_number)//2+RL_number-1, (types)-1]
    # plot_index = [i for i in range(RL_number)] + [RL_number, 7+RL_number, (types)-1]
    order = np.arange(len(plot_index))
    dots_to_plot = 100
    for i, index in enumerate(reversed((plot_index))):
        ax1.scatter(Y[index][0:dots_to_plot], X[index][0:dots_to_plot], color = colors[i], alpha = alpha[i], marker = markers[i],\
                s = msize, label = labels[index], zorder =int(order[i]))
        ax2.hist(Y[index][0:dots_to_plot], bins = binsx, color = colors[i],  alpha = alpha[i], edgecolor=colors[i], \
                 linewidth=2.0, zorder = int(order[i]), histtype='barstacked')
        ax3.hist(X[index][0:dots_to_plot], bins = binsy, color = colors[i], alpha = alpha[i], orientation='horizontal',\
                 edgecolor=colors[i], linewidth=2.0, zorder = int(order[i]), histtype='barstacked')
    
    labelsize = 17
    ax1.legend(fontsize = labelsize-3)
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # For hermite
    # ax1.set_xlim([8e-2, 1.0])
    # ax2.set_xlim([8e-2, 1.0])
    # ax1.set_ylim([1e-14, 1e0])
    # ax3.set_ylim([1e-14, 1e0])

    # For symple
    # ax1.set_xlim([2e-2, 0.4])
    # ax1.set_ylim([1e-14, 1e0])
    # ax2.set_xlim([2e-2, 0.4])
    # ax3.set_ylim([1e-14, 1e1])

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize-2)
    
    plt.savefig(save_path, dpi = 100)
    plt.show()

    
def plot_energy_vs_tcomp_integrators(env, STATES, CONS, TCOMP, Titles, seeds, save_path):
    """
    plot_energy_vs_tcomp_integrators: plot energy vs computation time at the end of the simulation for many initializations
    with different integrators
    INPUTS:
        env: environment for the runs
        states: state information containing particle evolution
        cons: information containing energy errors, rewards and actions taken
        tcomp: information containing the computation times
        titles: label  of each case
        save_path: path to save the figure
    """
    steps = len(TCOMP[0])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    # Group initializations per action type
    types = len(TCOMP)//len(seeds)
    X = []
    Y = []
    for i in range(types):
        energy = Energy_error[1:, i*len(seeds):(i+1)*len(seeds)]
        tcomp = T_comp[1:, i*len(seeds):(i+1)*len(seeds)]
        tc = np.zeros(len(seeds))
        en = np.zeros(len(seeds))
        for j in range(len(seeds)):
            index_last = np.nonzero(energy[:, j])[0][-1]
            en[j] = energy[index_last, j]
            tc[j] = np.sum(tcomp[0:index_last, j], axis = 0)       
        X.append(en)
        Y.append(tc)

    fig = plt.figure(figsize = (6,6))
    gs1 = matplotlib.gridspec.GridSpec(1, 1, figure = fig,
                                       left=0.18, wspace=0.3, 
                                       hspace = 0.2, right = 0.97,
                                        top = 0.97, bottom = 0.11)
    
    msize = 40
    alphavalue = 0.5
    alphavalue2 = 0.9
    labelsize = 17

    markers = ['o', 'x', 's', 'o']
    labels = []
    for i in range(types):
        labels.append(Titles[i*len(seeds)])

    ax1 = fig.add_subplot(gs1[:, :]) 

    order = [0, 1,2,3,4,5]
    for i in range(types):
        ax1.scatter(Y[i], X[i], color = colors[i//2], alpha = alphavalue, marker = markers[i%2],\
                s = msize, label = labels[i], zorder =order[i])
    
    ax1.legend(fontsize = labelsize-4, loc = 'upper right')
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=labelsize-2)

    
    ax1.set_xscale('log')
    ax1.set_xlim([0.08, 70.0])
    ax1.set_ylim([1e-14, 1e5])
    
    plt.savefig(save_path, dpi = 100)
    plt.show()

def plot_energy_vs_tcomp_compar(env, STATES, CONS, TCOMP, Titles, seeds, save_path, RL_number = 1):
    """
    plot_energy_vs_tcomp: plot energy and computation time at the end of the simulation for many initializations
    INPUTS:
        env: environment for the runs
        states: state information containing particle evolution
        cons: information containing energy errors, rewards and actions taken
        tcomp: information containing the computation times
        titles: label  of each case
        save_path: path to save the figure
    """
    steps = len(TCOMP[0])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    # # Group initializations per action type
    types = len(TCOMP)//len(seeds)
    X = []
    Y = []
    for i in range(types):
        X.append(Energy_error[-1, i*len(seeds):(i+1)*len(seeds)])
        Y.append(np.sum(T_comp[:, i*len(seeds):(i+1)*len(seeds)], axis = 0))

    
    fig = plt.figure(figsize = (8,8))
    gs1 = matplotlib.gridspec.GridSpec(2, 1, figure = fig, \
                                       left=0.18, wspace=0.4, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    
    msize = 80
    alphavalue = 0.5
    alphavalue2 = 0.9
    markers = ['o', 's', '^','x', 'o', 's', 'o', 'x', 's', 'o', 's', 'o', 'x', 's', 'o']
    # labels = [i[10] for i in Titles[0:RL_number]]
    labels = Titles
    # for i in range(types-RL_number):
    #     labels.append(r'$\mu$ = %.1E'%env.actions[i])
    # print(labels)

    ax1 = fig.add_subplot(gs1[0]) 
    ax2 = fig.add_subplot(gs1[1])

    plot_index = [i for i in range(RL_number)] + [RL_number, (types-RL_number)//2+RL_number-1, (types)-1]
    # plot_index = [i for i in range(RL_number)] + [RL_number, 7+RL_number, (types)-1]
    order = np.arange(len(plot_index))
    dots_to_plot = 100
    xaxis = np.arange(4)
    for j in range(len(xaxis)):
        for i, index in enumerate(reversed(plot_index)):
            if j == 0:
                label = Titles[index]
            else:
                label = ''
            ax1.scatter(xaxis[j], X[index][j], marker = markers[i], color = colors[i], label = label, s = msize)
            ax2.scatter(xaxis[j], Y[index][j], marker = markers[i], color = colors[i], s = msize)
    
    labelsize = 17
    ax1.legend(fontsize = labelsize-3)
    ax1.set_xlabel('Seed',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax2.set_ylabel('Total computation time (s)',  fontsize = labelsize)
    ax2.set_xlabel('Seed',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.grid(alpha = 0.5)
    ax2.grid(alpha = 0.5)

    # For hermite
    # ax1.set_xlim([8e-2, 1.0])
    # ax2.set_xlim([8e-2, 1.0])
    # ax1.set_ylim([1e-14, 1e0])
    # ax3.set_ylim([1e-14, 1e0])

    # For symple
    # ax1.set_xlim([2e-2, 0.4])
    # ax1.set_ylim([1e-14, 1e0])
    # ax2.set_xlim([2e-2, 0.4])
    # ax3.set_ylim([1e-14, 1e1])

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize-2)
    
    plt.savefig(save_path, dpi = 100)
    plt.show()

def plot_int_comparison(env, STATES, CONS, TCOMP, Titles, I, save_path):
    """
    plot_int_comparison: compare different integrators when applied to one initialization
    INPUTS:
        env: environment to use
        states: state information containing particle evolution
        cons: information containing energy errors, rewards and actions taken
        tcomp: information containing the computation times
        titles: label  of each case
        I: list of integrators
        save_path: path to save the figure
    """
    steps = len(TCOMP[0])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    
    fig = plt.figure(figsize = (10,14))
    gs1 = matplotlib.gridspec.GridSpec(len(I)+2, 1, figure = fig, left=0.13, wspace=0.3, 
                                       hspace = 0.5, right = 0.99,
                                        top = 0.97, bottom = 0.11)

    x_axis = np.arange(0, steps, 1)* env.settings['Integration']['check_step']
    labelsize = 16
    AX = []

    ax0 = fig.add_subplot(gs1[0, 0])
    name_planets = np.arange(np.shape(STATES)[1]).astype(str) 
    distance = plot_planets_distance(ax0, x_axis, STATES[0]/1.496e11, name_planets, steps = steps, labelsize = labelsize)
    ax0.set_yscale('log')
    ax0.set_ylabel('Pair-wise \n distance (au)', fontsize = labelsize+1)
    ax0.tick_params(axis='both', which='major', labelsize=labelsize)
    AX.append(ax0)

    ax2 = fig.add_subplot(gs1[-1, 0]) 
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]
    linestyle = '-'

    n_actions = env.settings['RL']['number_actions']
    for z in range(len(I)):
        ax1 = fig.add_subplot(gs1[z+1, 0]) # cartesian symple
        ax1.set_title(I[z], fontsize = labelsize+3)
        print(np.shape(x_axis), np.shape(action[:, z]))
        plot_actions_taken(ax1, x_axis, action[:, z])
        ax1.set_ylim([-1, n_actions +1])
        ax1.set_ylabel('Action taken', fontsize = labelsize+1)
        # ax1.set_yticks(np.arange(0,  env.settings['RL']['number_actions']+1))
        labels = np.arange(n_actions)[::3]
        ax1.set_yticks(np.arange(n_actions)[::3])
        LABELS = []
        for label in labels:
            LABELS.append('%i: %.0E'%(label, env.actions[label]))
        ax1.set_yticklabels(LABELS)
        ax1.tick_params(axis='both', which='major', labelsize=labelsize)
        AX.append(ax1)

        # Plot energy errors
        plot_evolution(ax2, x_axis, Energy_error[:, z], label = I[z], colorindex = z, linestyle = linestyle, linewidth = 2.5)
        
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), \
                       fancybox = True, ncol = 4, fontsize = labelsize+3)
    ax2.set_ylabel('Energy error',  fontsize = labelsize+1)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.set_yscale('log')
    ax2.set_xlabel("Time (yr)",fontsize = labelsize+1)
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
            ax_vert.axvline(x=X_D* env.settings['Integration']['check_step'], ymin=-1.2, ymax=1, c= 'k', lw =2, zorder = 0, \
                        clip_on = False, alpha = 0.2, linestyle = '--')
    for ax_vert in [AX[-1]]:
        for X_D in X_vert:
            ax_vert.axvline(x=X_D* env.settings['Integration']['check_step'], ymin=0, ymax=1, c= 'k', lw =2, zorder = 0,\
                         clip_on = False, alpha = 0.2, linestyle = '--')
    
    plt.savefig(save_path, dpi = 100)
    plt.show()