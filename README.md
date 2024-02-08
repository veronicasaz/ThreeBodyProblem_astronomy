# RL_integrator
Authors: Veronica Saz Ulibarrena, Simon Portegies Zwart

## Project description
Use of Reinforcement Learning for the choice of integration parameters that determine the time-step size for the integration of 
the gravitational N-body problem.

## Training and testing 
TestGym.py -> Train the RL algorithm 
TrainingFunctions.py -> Additional functions for training the RL algorithm

TestTrainedModelGym_hermite.py -> multiple experiments to visualize the results of the RL training
PlotsSimulation.py -> visualize results of the integration

## Environments
Cluster/cluster_2D/envs/HermiteIntegration_env.py: environment for the integration of the Nbody problem with Hermite integrator
Cluster/cluster_2D/envs/FlexibleIntegration_env.py: environment for the integration of the Nbody problem with different integrators

## Settings
settings_hermite.json: settings for the integration and training of the RL for the case with Hermite integrator
settings_multiple.json: settings for the integration and training of the RL for the case with different integrators

## Other
helpfunctions.py: additional functions to help loading files