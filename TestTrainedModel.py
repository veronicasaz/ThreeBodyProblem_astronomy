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
from Plots_TestTrained import plot_reward, plot_balance, plot_test_reward,\
    plot_trajs,  plot_trajs_RL, plot_energy_vs_tcomp

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
    experiment = 1 # number of the experiment to be run
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
        
        seeds = np.arange(5)
        # Plot evolution for all actions, one initialization
        for i in range(len(seeds)):
            env = ThreeBodyProblem_env()
            env.settings['Integration']['subfolder'] = '3_AllActionsRL/'

            NAMES = []
            TITLES = []
            NAMES.append('_actionRL'+'_seed_'+str(seeds[i]))
            TITLES.append(r"RL-variable $\mu$")
            env.settings['Integration']['suffix'] = NAMES[0]
            env.settings['InitialConditions']['seed'] = seeds[i]
            env.settings['Integration']['max_steps'] = 300

            model_path = env.settings['Training']['savemodel'] +'model_weights' +str(2500) +'.pth'
            run_trajectory(env, action = 'RL', model_path = model_path)
            
            for act in range(env.settings['RL']['number_actions']):
                NAMES.append('_action_'+ str(env.actions[act])+'_seed_'+str(seeds[i]))
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
                str(seeds[i]) + 'Action_comparison_RL.png'
            plot_trajs(env, STATE, CONS, TCOMP, TITLES, save_path, plot_traj_index=[0,1])

    elif experiment == 4: 
        # Plot evolution for many RL models
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '4_ManyRLmodelsRL/'

        NAMES = []
        TITLES = []

        # RL_models = ['0', '140', '200', '340', '600', '700', '800']
        RL_models = ['1600', '1660', '1700', '1960']
        for act in range(len(RL_models)):
            NAMES.append('_actionRL_'+ str(RL_models[act]))
            TITLES.append(r"RL-variable $\mu$ " + RL_models[act])
            env.settings['Integration']['suffix'] = NAMES[act]
            env.settings['Integration']['max_steps'] = 300
            model_path = env.settings['Training']['savemodel'] +'model_weights' +str(RL_models[act]) +'.pth'
            run_trajectory(env, action = 'RL', model_path = model_path)

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
        plot_trajs_RL(env, STATE, CONS, TCOMP, TITLES, save_path, plot_traj_index='bestworst')

    elif experiment == 5:
        # Run final energy vs computation time for different cases
        initializations = 500
        seeds = np.arange(initializations)

        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '5_EvsTcomp/'

        NAMES = []
        TITLES = []

        # RL
        for ini in range(initializations):
            NAMES.append('_actionRL_seed%i'%seeds[ini])
            TITLES.append(r"RL-variable $\mu$, seed %i"%seeds[ini])
            env.settings['Integration']['suffix'] = NAMES[ini]
            env.settings['InitialConditions']['seed'] = seeds[ini]
            run_trajectory(env, action = 'RL')

        for act in range(env.settings['RL']['number_actions']):
            for ini in range(initializations):
                name = '_action%i_seed%i'%(act, seeds[ini])
                NAMES.append(name)
                TITLES.append(r'%i: $\mu$ = %.1E'%(act, env.actions[act]))
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[ini]
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
            'Energy_vs_tcomp.png'
        plot_energy_vs_tcomp(env, STATE, CONS, TCOMP, NAMES, seeds, save_path)


