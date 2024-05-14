import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from scipy.signal import savgol_filter
from Plots_TestEnvironment import calculate_errors
from PlotsFunctions import plot_planets_trajectory, plot_evolution, \
    plot_planets_distance, plot_actions_taken

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

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
    
    f, ax = plt.subplots(4, 1, figsize = (10,6))
    plt.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.15, hspace = 0.3)
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
    ax[0].set_ylabel(r'Cumul $R$/episode', fontsize = fontsize)

    # y = avg_reward_perepisode
    y = avg_reward_perepisode
    ax[1].plot(x_episodes,y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[1].plot(x_episodes, yy, color = 'black')
    ax[1].set_ylabel(r'Avg $R$/episode', fontsize = fontsize)
    # ax[1].set_yscale('symlog', linthresh = 1e1)

    y = avg_energy_perepisode
    ax[2].plot(x_episodes, y, color = colors[0], alpha = 1)
    # y = energyerror_flat
    # ax[2].plot(x_all, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[2].plot(x_episodes, yy, color = 'black')
    ax[2].set_ylabel(r'Slope $\Delta E/episode$', fontsize = fontsize)
    ax[2].set_yscale('log')

    y = last_energy_perepisode
    ax[3].plot(x_episodes, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[3].plot(x_episodes, yy, color = 'black')
    ax[3].set_ylabel(r'Final $\Delta E/episode$', fontsize = fontsize)
    ax[3].set_yscale('log')

    for ax_i in ax: 
        ax_i.tick_params(axis='x', labelsize=fontsize-5)
        ax_i.tick_params(axis='y', labelsize=fontsize-5)

    ax[-1].set_xlabel('Episode', fontsize = fontsize)
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path + 'cumulative_reward.png', dpi = 100)
    plt.show()

def plot_test_reward(a, test_reward):
    f, ax = plt.subplots(3, 1, figsize = (10,6))
    plt.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.15, hspace = 0.3)
    fontsize = 18

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2
    
    # pts = 11
    # ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    episodes = len(test_reward)-1 #TODO: 
    x_episodes = np.arange(episodes)
    
    REWARD_avg = []
    REWARD_std = []
    EERROR_avg = []
    EERROR_std = []
    TCOMP_avg = []
    TCOMP_std = []
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
        # ax[plot].scatter(x_episodes, y[plot] + e[plot], c = colors[1], \
        #                   alpha = 0.5, marker = '.')
        ax[plot].plot(x_episodes, y[plot] + e[plot], color = colors[1], \
                          alpha = 0.2, marker = '.')
        ax[plot].plot(x_episodes, y[plot] - e[plot], color = colors[1], \
                          alpha = 0.2, marker = '.')
        # ax[plot].scatter(x_episodes, y[plot] - e[plot], c = colors[1], \
        #                   alpha = 0.5, marker = '.')
        ax[plot].plot(x_episodes, y[plot], color= colors[0], \
                          alpha = 1, marker = '.')

    for ax_i in ax: 
        ax_i.tick_params(axis='x', labelsize=fontsize-5)
        ax_i.tick_params(axis='y', labelsize=fontsize-5)

    ax[-1].set_xlabel('Episode', fontsize = fontsize)
    ax[0].set_ylabel('R', fontsize = fontsize)
    ax[1].set_ylabel(r'$log_{10}(\vert \Delta E\vert)$', fontsize = fontsize)
    ax[2].set_ylabel(r'$T comp$ (s)', fontsize = fontsize)
    ax[2].set_yscale('log')

    # ax[1].set_yscale('log')
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path + 'test_reward.png', dpi = 100)
    plt.show()

def plot_balance(a, reward, Eerror, tcomp):
    """
    plot_reward: plot training parameters such as reward
    INPUTS:
        a: environment
        reward: rewards
        Eerror: energy error
        HuberLoss: huber loss
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
    distance = plot_planets_distance(ax3, x_axis, STATES[0][1:,:]/1.496e11, name_bodies,\
                                        steps = steps, labelsize = label_size)

    # Plot RL 
    plot_actions_taken(ax4, x_axis, action[1:, 0])
    ax4.set_ylim([-1, env.settings['RL']['number_actions']+1])
    ax4.set_ylabel('Action taken', fontsize = label_size)
    ax4.set_yticks(np.arange(0, env.settings['RL']['number_actions']))

    # Plot energy error
    for case in range(len(STATES)):
        if case == 0:
            label = 'RL'
        else:
            label = r"$%i: \mu$ =%.2E"%(case, env.actions[case-1])
        plot_evolution(ax5, x_axis, R[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case], linewidth = linewidth)
        plot_evolution(ax6, x_axis, Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case], linewidth = linewidth)
        
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
            ax_vert.axvline(x=X_D, ymin=-1.2, ymax=1, c= 'k', lw =2, zorder = 0, \
                    clip_on = False, alpha = 0.2, linestyle = '--')
        for ax_vert in [ax6]:
            ax_vert.axvline(x=X_D, ymin=0, ymax=1, c= 'k', lw =2, zorder = 0,\
                        clip_on = False, alpha = 0.2, linestyle = '--')

    plt.savefig(save_path, dpi = 150)
    plt.show()



def plot_trajs_RL(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    """
    plot_trajs_RL: plot results obtained with different trained models
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
                       colorindex = case, linestyle = linestyle[case], linewidth = linewidth)
        plot_evolution(ax6, x_axis, Energy_error[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle[case], linewidth = linewidth)
        plot_actions_taken(ax4, x_axis, action[1:, case], colorindex = case, label = Titles[case])
        
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

def plot_energy_vs_tcomp(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    a = 1
