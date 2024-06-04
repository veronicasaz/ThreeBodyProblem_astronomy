# RL_integrator
Authors: Veronica Saz Ulibarrena, Simon Portegies Zwart

## Project description
Use of Reinforcement Learning for the choice of integration parameters that determine the time-step size for the integration of 
the gravitational 3-body problem.

Full paper available at: LINK

## Environments
* `env/ThreeBP_env.py`: environment for the integration of the 3-body problem with different integrators. Hermite, Huayno, ph4, and Symple available. 

## Settings
* `settings_integration_3BP.json`: settings containing initial conditions, integration, and RL training parameters.

## Training and testing 
* `TestEnvironment.py`: experiments to analyze the problem, study the reward function and test the performance of the environment. 
* `TestTrainedModel.py`: Train the RL algorithm and evaluate the trained model. Perform experiments to understand the performance of the trained model compared to the baseline. 
* `TrainRL.py`: functions necessary for training. DeepQN implementation and more
* `TestTrainedIntegrators.py`: experiments to apply the trained model to other integrators and retrain for Symple integrator. 

## Plots
* `PlotsFunctions.py`: basic functions to plot trajectories, time evolution, and actions taken.
* `Plots_TestEnvironment.py`: plots and functions to plot the results of the experiments in * `TestEnvironment.py`.
* `Plots_TestTrained.py`: plots and functions to plot the results of the experiments in * `TestTrainedModel.py` and * `TestTrainedntegrators.py`.



## Other
* `helpfunctions.py`: additional functions to help loading files
