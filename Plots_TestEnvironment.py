"""
PlotsSimulation: plotting functions

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from TrainingFunctions import DQN
from PlotsFunctions import plot_planets_trajectory

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def plot_initializations(state, cons, tcomp, names, save_path, seed):
    # Setup plot
    title_size = 20
    label_size = 18
    fig = plt.figure(figsize = (10,17))
    rows = 3
    columns = 2
    fig, axes = plt.subplots(nrows=rows, ncols= columns, \
                             layout = 'constrained', figsize = (8, 12))
    
    name_planets = (np.arange(np.shape(state)[1])+1).astype(str)
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