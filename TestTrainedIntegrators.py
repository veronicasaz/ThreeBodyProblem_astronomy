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
from Plots_TestTrained import plot_test_reward,\
    plot_trajs,  plot_trajs_RL, plot_energy_vs_tcomp, \
    plot_energy_vs_tcomp_integrators, plot_int_comparison
from TestTrainedModel import load_reward


if __name__ == '__main__':
    experiment = 1 # number of the experiment to be run
    seed = 0

    ################################
    # Integrator comparison
    ################################
    if experiment == 0: # plot comparable actions

        # Run final energy vs computation time for different cases
        initializations = 100
        seeds = np.arange(initializations)

        NAMES = []
        TITLES = []
        I = ['Hermite', 'Huayno', 'Symple']

        for integrat in range(len(I)):
            env = ThreeBodyProblem_env()
            env.settings['Integration']['subfolder'] = '6_ComparableActions_integrators/'
            env.settings['Integration']['integrator'] = I[integrat]
            env._initialize_RL() # redo to adapt actions to integrator
            for ini in range(initializations):
                print(I[integrat], ini, env.actions)
                index_0 = 0
                name = '%s_seed%i_action%i'%(I[integrat], seeds[ini], index_0)
                NAMES.append(name)
                TITLES.append(r"%s: $\mu$ = %.1E"%(I[integrat], env.actions[index_0]))
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[ini]
                # run_trajectory(env, action = index_0)
            
            for ini in range(initializations):
                index_1 = len(env.actions)-1
                name = '%s_seed%i_action%i'%(I[integrat], seeds[ini], index_1)
                NAMES.append(name)
                TITLES.append(r"%s: $\mu$ = %.1E"%(I[integrat], env.actions[index_1]))
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[ini]
                # run_trajectory(env, action = index_1)


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
            'Energy_vs_tcomp_comparable_actions.png'
        plot_energy_vs_tcomp_integrators(env, STATE, CONS, TCOMP, TITLES, seeds, save_path)


    elif experiment == 1: # run actions integrators
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '6_Comparison_integrators/'

        # Run final energy vs computation time for different cases
        initializations = 100
        seeds = np.arange(initializations)

        NAMES = []
        TITLES = []
        I = ['Hermite', 'Huayno', 'Symple']

        for integrat in range(len(I)):
            env = ThreeBodyProblem_env()
            env.settings['Integration']['subfolder'] = '6_ComparableActions_integrators/'
            env.settings['Integration']['integrator'] = I[integrat]
            env._initialize_RL() # redo to adapt actions to integrator
            for ini in range(initializations):
                print(I[integrat], ini, env.actions)
                index_0 = 0
                name = '%s_seed%i_action%i'%(I[integrat], seeds[ini], index_0)
                NAMES.append(name)
                TITLES.append(r"%s: $\mu$ = %.1E"%(I[integrat], env.actions[index_0]))
                env.settings['Integration']['suffix'] = name
                env.settings['InitialConditions']['seed'] = seeds[ini]
                run_trajectory(env, action = index_0)

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
            'Evolution_integrators.png'
        plot_int_comparison(env, STATE, CONS, TCOMP, TITLES, I, save_path)


    ################################
    # Symple
    ################################
    elif experiment == 2: # Train symple
        env = ThreeBodyProblem_env()
        env.settings['Training']["savemodel"] = "./Training_Results_Symple/"
        env.settings['Integration']['integrator'] = 'Symple/'
        train_net()

    elif experiment == 3:
        # Plot training results
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] =  '2_Training/Symple/' 
        env.settings['Integration']['integrator'] = 'Symple/'
        reward, EnergyError, HuberLoss, tcomp, testReward = load_reward(env, suffix = '')
        plot_test_reward(env, testReward)
    
    elif experiment == 4: 
        model = 100
        
        seeds = np.arange(5)
        # Plot evolution for all actions, one initialization
        for i in range(len(seeds)):
            env = ThreeBodyProblem_env()
            env.settings['Integration']['subfolder'] = '7_AllActionsRL_Symple/'
            env.settings['Integration']['integrator'] = 'Symple/'

            NAMES = []
            TITLES = []
            NAMES.append('_actionRL'+'_seed_'+str(seeds[i]))
            TITLES.append(r"RL-variable $\mu$")
            env.settings['Integration']['suffix'] = NAMES[0]
            env.settings['InitialConditions']['seed'] = seeds[i]
            env.settings['Integration']['max_steps'] = 300

            model_path = env.settings['Training']['savemodel'] +'model_weights' +str(model) +'.pth'
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

    elif experiment == 5: 
        # Plot evolution for many RL models
        env = ThreeBodyProblem_env()
        env.settings['Integration']['subfolder'] = '8_ManyRLmodelsRL_Symple/'
        env.settings['Integration']['integrator'] = 'Symple/'

        NAMES = []
        TITLES = []

        RL_models = ['100', '1530', '700', '2200', '2500']
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

    elif experiment == 6:
        # Run final energy vs computation time for different cases
        initializations = 500
        seeds = np.arange(initializations)

        env = ThreeBodyProblem_env()
        
        env.settings['Integration']['subfolder'] = '9_EvsTcomp_Symple/'
        env.settings['Integration']['integrator'] = 'Symple'

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


