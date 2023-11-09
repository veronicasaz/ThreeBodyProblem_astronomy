import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Cluster.cluster_2D.envs.SympleIntegration_env import IntegrateEnv


def runs_trajectory(cases, action, steps):
    a = IntegrateEnv()
    seeds = np.arange(cases)
    for j in range(cases):
        print("Case %i"%j)
        a.suffix = ('_traj_case%i'%j)
        i = 0
        a.reset(seed = seeds[j])
        while i < (steps):
            x, y, terminated, zz = a.step(action)
            i += 1
        a.close()

def plot_runs_trajectory(cases, steps):
    a = IntegrateEnv()
    a.reset()

    # Load run information for symple cases
    state = list()
    cons = list()
    tcomp = list()
    name = list()
    for i in range(cases):
        a.suffix = ('_traj_case%i'%i)
        state.append(a.loadstate()[0])
        cons.append(a.loadstate()[1])
        tcomp.append(a.loadstate()[2])
        name.append('_case%i'%i)

    # plot
    colors = ['orange', 'green', 'blue', 'red', 'grey', 'black']
    lines = ['-', '--', ':', '-.' ]
    markers = ['o', 'x', '.', '^', 's']
    label_size = 20

    n_planets = np.shape(state[0])[1]
    name_planets = a.names
    
    fig, axes = plt.subplots(nrows=int(cases//3), ncols= 3, layout = 'constrained', figsize = (10, 10))

    for i, ax in enumerate(axes.flat):
        for j in range(n_planets):
            x = state[i][0:steps, j, 2]
            y = state[i][0:steps, j, 3]
            m = state[i][0, j, 1]
            size_marker = np.log(m)* 7

            ax.plot(x[1:], y[1:], markersize = size_marker, \
                        linestyle = lines[j%len(lines)],\
                        color = colors[j%len(colors)], \
                        label = name_planets[j])
        ax.set_xlabel('x (m)', fontsize = label_size)
        ax.set_xlabel('y (m)', fontsize = label_size)
        if i == 0:
            ax.legend(fontsize = 10)

    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    cases = 9
    action = 5
    steps = 100
    # runs_trajectory(cases, action, steps)
    plot_runs_trajectory(cases, steps)