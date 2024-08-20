"""
TestTrainedModel: tests and plots for the RL algorithm

Author: Veronica Saz Ulibarrena
Last modified: 31-May-2024
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
from Plots_TestTrained import plot_reward, plot_balance, \
    plot_test_reward,plot_test_reward_multiple,\
    plot_trajs,  plot_trajs_RL, plot_energy_vs_tcomp

def create_testing_dataset(env, seeds):
    """
    create_testing_dataset: create a dataset to test
    """
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
        suffix: suffix name of the file
    OUTPUTS:
        score: rewards
        EnergyE: energy error
        HuberLoss: huber loss
        tcomp: computation time
        testReward: reward of the test dataset
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

    trainingTime = []
    with open(a.settings['Training']['savemodel'] + suffix + "TrainingTime.txt", "r") as f:
        for j in f.read().split():
            trainingTime.append(float(j))

    HuberLoss = []

    return score, EnergyE, HuberLoss, tcomp, testReward, trainingTime

if __name__ == '__main__':
    experiment = 6 # number of the experiment to be run
    seed = 0

    if experiment == 0: # Create testing dataset
        env = ThreeBodyProblem_env()
        systems = 10
        seeds = np.arange(systems)
        create_testing_dataset(env, seeds)

    elif experiment == 1: # Train model 
        env = ThreeBodyProblem_env()
        env.settings['Training']['seed_weights'] = 3
        # train_net(env = env, suffix = "retrained/") # without pretraining


        # From pretrained model
        # model = '173_good'
        model = '1444_pretrained'
        # model = '357_pretrained'
        env = ThreeBodyProblem_env()
        model_path = env.settings['Training']['savemodel'] +'model_weights' +model +'.pth'
        # env.settings['Training']['max_episodes'] = 100
        env.settings['Training']['lr'] = 1e-5
        # train_net(model_path_pretrained=model_path, env = env, suffix = "currentTraining/")
        train_net(model_path_pretrained=model_path, env = env, suffix = "retrained/")

    elif experiment == 1.5: # Train model with different seeds
        env = ThreeBodyProblem_env()
        seeds_train = 4
        for i in range(seeds_train):
            env.settings['Training']['seed_weights'] = i
            # train_net(env = env, suffix = "seed_%i/"%i) # without pretraining

        env.settings['Integration']['subfolder'] =  '2_Training/'
        TESTREWARD = [] 
        TRAININGTIME = []
        # for i in range(seeds_train):
        for i in range(3):
            reward, EnergyError, HuberLoss, tcomp, testReward, trainingTime = load_reward(env, suffix = 'seed_%i/'%i)
            TESTREWARD.append(testReward)
            TRAININGTIME.append(trainingTime)

        plot_test_reward_multiple(env, TESTREWARD, TRAININGTIME)


    elif experiment == 2:
        # Plot training results
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] =  '2_Training/' 
        # reward, EnergyError, HuberLoss, tcomp, testReward, trainingTime = load_reward(env, suffix = 'retrained/')
        # reward, EnergyError, HuberLoss, tcomp, testReward, trainingTime = load_reward(env, suffix = '25_from24/1444_rl8!!/')
        reward, EnergyError, HuberLoss, tcomp, testReward, trainingTime = load_reward(env, suffix = '24_biggerNet/seed_1/')
        plot_reward(env, reward, EnergyError, HuberLoss)
        # plot_balance(env, reward, EnergyError, tcomp)
        # plot_test_reward(env, testReward, trainingTime)

    elif experiment == 3: # plot comparison of trianed model with baseline results
        # model = '1092_good'
        model = '11'
        
        seeds = np.arange(7)
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
            
            # index_to_plot = [0, 1, 6, 11, 16, 20]
            index_to_plot = [0, 1,3,5,  8, 10]

            # model_path = env.settings['Training']['savemodel'] +'model_weights' +model +'.pth'
            # model_path = env.settings['Training']['savemodel'] +'25_from24/1444_rl8/model_weights88.pth'
            model_path = env.settings['Training']['savemodel'] +'25_from24/1444_rl5e-8/model_weights184.pth'
            

            run_trajectory(env, action = 'RL', model_path = model_path)
            
            for act in range(env.settings['RL']['number_actions']):
                NAMES.append('_action_'+ str(env.actions[act])+'_seed_'+str(seeds[i]))
                TITLES.append(r'%i: $\mu$ = %.1E'%(act, env.actions[act]))
                env.settings['Integration']['suffix'] = NAMES[act+1]
                # run_trajectory(env, action = act)

            STATE = []
            CONS = []
            TCOMP = []
            for act in index_to_plot:
                env.settings['Integration']['suffix'] = NAMES[act]
                state, cons, tcomp = load_state_files(env)
                STATE.append(state)
                CONS.append(cons)
                TCOMP.append(tcomp)
            TITLES = [TITLES[x] for x in index_to_plot]
            print(TITLES)

            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                str(seeds[i]) + 'Action_comparison_RL.png'
            plot_trajs(env, STATE, CONS, TCOMP, TITLES, save_path, plot_traj_index=[0,1])

    elif experiment == 4: 
        # Plot evolution for many RL models
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '4_ManyRLmodelsRL/'
        env.settings['InitialConditions']['seed'] = 5

        NAMES = []
        TITLES = []

        # RL_models = ['2080', '2090', '2100', '2110', '2120']
        RL_models = ['80_good', '270_good', '490_good', '2090_good']
        for act in range(len(RL_models)):
            NAMES.append('_actionRL_'+ str(RL_models[act]))
            TITLES.append(r"RL-variable $\mu$ " + RL_models[act])
            env.settings['Integration']['suffix'] = NAMES[act]
            env.settings['Integration']['max_steps'] = 150
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
        initializations = 100
        # model=  '270_good'
        # model = '80_good'
        model = '357_pretrained'
        seeds = np.arange(initializations)

        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '5_EvsTcomp/'

        NAMES = []
        TITLES = []

        # RL
        model_path = env.settings['Training']['savemodel'] +'model_weights' + model +'.pth'
        for ini in range(initializations):
            NAMES.append('_actionRL_seed%i'%seeds[ini])
            TITLES.append(r"RL-variable $\mu$, seed %i"%seeds[ini])
            env.settings['Integration']['suffix'] = NAMES[ini]
            env.settings['InitialConditions']['seed'] = seeds[ini]
            env.settings['Integration']['max_steps'] = 300
            run_trajectory(env, action = 'RL', model_path = model_path)

        for act in range(env.settings['RL']['number_actions']):
            for ini in range(initializations):
                name = '_action%i_seed%i'%(act, seeds[ini])
                NAMES.append(name)
                TITLES.append(r'%i: $\mu$ = %.1E'%(act, env.actions[act]))
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[ini]
                env.settings['Integration']['max_steps'] = 300
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


    elif experiment == 6:
        # Run final energy vs computation time for different cases
        initializations = 100
        seeds = np.arange(initializations)

        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '5_EvsTcomp/'

        NAMES = []
        TITLES = []
        # MODELS = ['seed_1/model_weights1608', 'seed_1/model_weights1444','seed_2/model_weights357','seed_2/model_weights818']
        # MODELS = ['24_biggerNet/seed_1/model_weights1444']
        # MODELS = ['25_from24/1444_rl5e-8/model_weights184']
        MODELS = ['25_from24/1444_rl8!!/model_weights88']
                    # '25_from24/1444_rl7_differentsettings/model_weights20',
                    # 
                    # '25_from24/1444_rl6/model_weights4']
                #   '24_biggerNet/seed_2/model_weights357','24_biggerNet/seed_2/model_weights818']
        MODELS_title = ['1444-88', '1444', '1444-88', '1444-20', '1444-4']

        # RL
        for m_i in range(len(MODELS)):
            model_path = env.settings['Training']['savemodel'] + MODELS[m_i] +'.pth'
            for ini in range(initializations):
                print(m_i, ini)
                NAMES.append('_actionRL_%i_seed%i'%(m_i, seeds[ini]))
                env.settings['Integration']['suffix'] = NAMES[ini+ initializations*m_i]
                env.settings['InitialConditions']['seed'] = seeds[ini]
                env.settings['Integration']['max_steps'] = 300
                env.settings['Training']['display'] = False
                # run_trajectory(env, action = 'RL', model_path = model_path)
            TITLES.append(r"RL-"+MODELS_title[m_i])

        for act in range(env.settings['RL']['number_actions']):
            for ini in range(initializations):
                name = '_action%i_seed%i'%(act, seeds[ini])
                NAMES.append(name)
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[ini]
                env.settings['Integration']['max_steps'] = 300
                # run_trajectory(env, action = act)
            TITLES.append(r'$\mu$ = %.1E'%(env.actions[act]))

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
        plot_energy_vs_tcomp(env, STATE, CONS, TCOMP, TITLES, seeds, save_path, RL_number=len(MODELS))