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

from env.ThreeBP_env import ThreeBodyProblem_env
from TrainRL import train_net
from TestEnvironment import run_trajectory, load_state_files
from Plots_TestTrained import plot_reward, plot_balance, plot_test_reward, plot_trajs

def create_testing_dataset(env, seeds):
    state = np.zeros((len(seeds), 6*3)) # r, v for 3 particles
    for i in range(len(seeds)):
        env.settings['InitialConditions']['seed'] = seeds[i]
        st, info = env.reset()
        state[i, 0:9] = env.particles.position.value_in(env.units_l).flatten()
        state[i, 9:] = env.particles.velocity.value_in(env.units_l/env.units_t).flatten()
        env.close()

    np.save(env.settings['Training']['savemodel'] + 'TestDataset', state)

def load_reward(a, suffix = ''):
    """
    load_reward: load rewards from file 
    INPUTS:
        a: environment
    OUTPUTS:
        score: rewards
        EnergyE: energy error
        HuberLoss: huber loss
    """
    score = []
    with open(a.settings['Training']['savemodel'] + suffix + "rewards.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            score_r = list()
            for j in y.split():
                score_r.append(float(j))
            score.append(score_r)

    EnergyE = []
    with open(a.settings['Training']['savemodel'] + suffix + "EnergyError.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            Energy_r = list()
            for j in y.split():
                Energy_r.append(float(j))
            EnergyE.append(Energy_r)

    tcomp = []
    with open(a.settings['Training']['savemodel'] + suffix + "Tcomp.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            tcomp_r = list()
            for j in y.split():
                tcomp_r.append(float(j))
            tcomp.append(tcomp_r)

    testReward = []
    with open(a.settings['Training']['savemodel'] + suffix + "TestReward.txt", "r") as f:
        # for line in f:
        for y in f.read().split('\n'):
            testreward_r = list()
            for j in y.split():
                testreward_r.append(float(j))
            testReward.append(testreward_r)

    HuberLoss = []

    return score, EnergyE, HuberLoss, tcomp, testReward

if __name__ == '__main__':
    experiment = 3 # number of the experiment to be run
    seed = 0

    if experiment == 0: # Create testing dataset
        env = ThreeBodyProblem_env()
        systems = 10
        seeds = np.arange(systems)
        create_testing_dataset(env, seeds)

    elif experiment == 1: # Train
        train_net()

    elif experiment == 2:
        # Plot training results
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] =  '2_Training/' 
        reward, EnergyError, HuberLoss, tcomp, testReward = load_reward(env, suffix = '')
        # plot_reward(env, reward, EnergyError, HuberLoss)
        # plot_balance(env, reward, EnergyError, tcomp)
        plot_test_reward(env, testReward)
        
    elif experiment == 3: 
        # Plot evolution for all actions, one initialization
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '3_AllActionsRL/'

        NAMES = []
        TITLES = []
        NAMES.append('_actionRL')
        TITLES.append(r"RL-variable $\mu$")
        env.settings['Integration']['suffix'] = NAMES[0]
        run_trajectory(env, action = 'RL')
        for act in range(env.settings['RL']['number_actions']):
            NAMES.append('_action'+ str(env.actions[act]))
            TITLES.append(r'%i: $\mu$ = %.1E'%(act, env.actions[act]))
            env.settings['Integration']['suffix'] = NAMES[act+1]
            # run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        for act in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL.png'
        plot_trajs(env, STATE, CONS, TCOMP, TITLES, save_path, plot_traj_index=[0,1])
        # save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
        #     'Action_comparison_RL2.png'
        # plot_distance_action(env, STATE, CONS, TCOMP, NAMES, save_path)
        # save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
        #     'Action_comparison_RL3.png'
        # plot_comparison_end(env, STATE, CONS, TCOMP, NAMES, save_path, plot_traj_index=[0,1])

    elif experiment == 4:
        def start_env():
            env = Cluster_env()
            env.settings['Integration']['subfolder'] = '6_run_many/'
            env.settings['Integration']['max_steps'] = 60
            env.settings['Integration']['savestate'] = True
            return env

        initializations = 10
        seeds = np.arange(initializations)
        NAMES = []


        for i in range(initializations):
            env = start_env()
            env.settings['InitialConditions']['seed'] = seeds[i]
            NAMES.append('_actionRL_%i'%i)
            print(NAMES)
            env.settings['Integration']['suffix'] = NAMES[i]
            # run_trajectory(env, action = 'RL')

        for act in range(env.settings['RL']['number_actions']):
            for i in range(initializations):
                print(act, i)
                env = start_env()
                env.settings['InitialConditions']['seed'] = seeds[i]
                name = '_action_%i_%i'%(act, i)
                NAMES.append(name)
                env.settings['Integration']['suffix'] = name
                # run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        for act in range(len(NAMES)):
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)

        env = start_env()
        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Energy_vs_tcomp.png'
        plot_energy_vs_tcomp(env, STATE, CONS, TCOMP, NAMES, initializations, save_path, plot_traj_index=[0, 1, 2, 3, 4])


