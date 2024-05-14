"""
PlotsSimulation: plotting functions

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']


def plot_planets_trajectory(ax, state, name_planets, labelsize = 15, steps = 30, \
                            legend_on = True, axislabel_on = True, marker = 'o', axis = 'xy'):
    """
    plot_planets_trajectory: plot trajectory of three bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
        legend_on: True or False to display the legend
    """
    if axis == 'xy': 
        indexes = [2, 3]
    elif axis == 'xz':
        indexes = [2, 4]

    n_planets = np.shape(state)[1]
    for j in range(n_planets):
        x = state[0:steps, j, indexes[0]]/1.496e11
        y = state[0:steps, j, indexes[1]]/1.496e11
        m = state[0, j, 1]
        size_marker = np.log10(m)/10

        ax.scatter(x[0], y[0], s = 20*size_marker,\
                   c = colors[j%len(colors)], \
                    label = "Particle "+ name_planets[j])
        ax.plot(x[1:], y[1:], marker = None, 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[j%len(colors)], \
                    alpha = 0.1)
        
        ax.scatter(x[1:], y[1:], marker = marker, s = size_marker, \
                    c = colors[j%len(colors)])        
        
    if legend_on == True:
        ax.legend(fontsize = labelsize)
    if axislabel_on == True:
        ax.set_xlabel('x (au)', fontsize = labelsize)
        ax.set_ylabel('y (au)', fontsize = labelsize)


    
def plot_planets_distance(ax, x_axis, state, name_planets, labelsize = 12, 
                          steps = 30, legend = True):
    """
    plot_planets_distance: plot steps vs pairwise-distance of the bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
    """
    n_planets = np.shape(state)[1]
    Dist = []
    Labels = []
    for i in range(n_planets):
        r1 = state[0:steps, i, 2:5]
        m = state[0, i, 1]
        for j in range(i+1, n_planets):
            r2 = state[0:steps, j, 2:5]
            Dist.append(np.linalg.norm(r2-r1, axis = 1))
            Labels.append('Particle %i-%i'%(i, j))
        
        size_marker = np.log(m)/30
    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i], linewidth = 2.5)
    if legend == True:
        ax.legend(fontsize =labelsize, framealpha = 0.5)
    ax.set_yscale('log')
    return Dist


def plot_distance_to_one(ax, x_axis, state, labelsize = 12, 
                         legend = True):
    """
    plot_planets_distance: plot steps vs pairwise-distance of the bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
    """
    steps = len(x_axis)
    Dist = []
    Labels = []
    
    last_index = np.where(state[0, :, 8] == 1)[0][0]
    r0 = state[0:steps, last_index, 2:5]
    for i in range(0, last_index):
        r1 = state[0:steps, i, 2:5]
        m = state[0, i, 1]
        Dist.append(np.linalg.norm(r0-r1, axis = 1))
        Labels.append('Particle %i'%(i))
        
        size_marker = np.log(m)/30
    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i], linewidth = 2.5)
    if legend == True:
        ax.legend(fontsize =labelsize, framealpha = 0.5)
    # ax.set_yscale('log')

    return Dist

def plot_actions_taken(ax, x_axis, y_axis, colorindex = None, label = None):
    """
    plot_actions_taken: plot steps vs actions taken by the RL algorithm
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
    """
    if colorindex != None:
        color = colors[(colorindex+2)%len(colors)] # start in the blues
    else:
        color = colors[0]
    ax.plot(x_axis, y_axis, color = color, linestyle = '-', alpha = 0.5,
            marker = '.', markersize = 8, label = label)
    ax.grid(axis='y')

def plot_evolution(ax, x_axis, y_axis, label = None, color = None, 
                   colorindex = None, linestyle = None, linewidth = 1):
    """
    plot_evolution: plot steps vs another measurement
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
        label: label for the legend of each data line
        color: color selection for each line
        colorindex: index of the general color selection
        linestyle: type of line
        linewidth: matplotlib parameter for the width of the line
    """
    if colorindex != None:
        color = colors[(colorindex+2)%len(colors)] # start in the blues
    ax.plot(x_axis, y_axis, color = color, linestyle = linestyle, label = label, 
            linewidth = linewidth)
