import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches   
from matplotlib.gridspec import GridSpec

from amuse.units import units, constants, nbody_system

def plot_state(bodies):
    v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
    plt.scatter(bodies.x.value_in(units.au),\
                bodies.y.value_in(units.au), \
                c=v.value_in(units.kms), alpha=0.5)
    plt.colorbar()
    plt.show()


def plot_trajectory(settings):
    state = np.load(settings['Integration']['savefile'] +'_state.npy')
    cons = np.load(settings['Integration']['savefile'] +'_cons.npy')
    t_comp = np.load(settings['Integration']['savefile'] +'_tcomp.npy')

    # Divide into different actions found
    index_0 = np.where(cons[:,0] == 0)[0]
    index_1 = np.where(cons[:,0] == 1)[0]

    markers = ['o', 'x'] # representing each action
    names = ['BHTree', 'Hermite']

    # Find border separating actions
    index_b = np.where(abs(cons[1:,0] - cons[:-1,0]) > 0)[0]
    index_b += 1

    # Time steps
    t = np.arange(0, (settings['Integration']['check_step']) * \
        settings['Integration']['t_step'] * settings['Integration']['max_steps'], \
        settings['Integration']['t_step'])

    fig = plt.figure()
    fig.suptitle("Trajectory")
    # gs = GridSpec(2,2, width_ratios= [1,2], height_ratios=[1,1])
    gs = GridSpec(2,2, width_ratios= [1,1], height_ratios=[1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 0])


    # Trajectory
    for i in range(settings['Integration']['bodies']):
        ax1.scatter(state[0, i, 2], state[0, i, 3], marker = 'o', s = 150)
        ax1.plot(state[:, i, 2], state[:, i, 3])

    ax1.set_xlabel(r"$x$ (m)")
    ax1.set_ylabel(r"$y$ (m)")

    # Energy error
    ax2.plot(t, (cons[:, 1]- cons[0, 1])/ cons[0, 1])
    ax2.set_xlabel("Integration time (yr)")
    ax2.set_ylabel("Relative Energy error")

    # Add distinction between integrators
    for i in range(len(index_b)):
        if i < 2:
            ax2.scatter(t[index_b[i]], (cons[index_b[i], 1]- cons[0, 1])/ cons[0, 1] , \
                    marker = markers[int(cons[index_b[i],0])],
                    color = 'black', label = names[int(cons[index_b[i],0])])
        else:
            ax2.scatter(t[index_b[i]], (cons[index_b[i], 1]- cons[0, 1])/ cons[0, 1] , \
                    marker = markers[int(cons[index_b[i],0])],
                    color = 'black')
        
    # plt.legend(handles=['o','x)
    ax2.legend()

    # Angular momentum error
    L0 = cons[0, 2:]
    L = np.linalg.norm(cons[:, 2:] - L0, axis = 1) / np.linalg.norm(L0)
    ax3.plot(t, L)
    ax3.set_xlabel("Integration time (yr)")
    ax3.set_ylabel("Relative Angular momentum error")

    # Add distinction between integrators
    for i in range(len(index_b)):
        if i < 2:
            ax3.scatter(t[index_b[i]], L[index_b[i]] , \
                    marker = markers[int(cons[index_b[i],0])],
                    color = 'black', label = names[int(cons[index_b[i],0])])
        else:
            ax3.scatter(t[index_b[i]], L[index_b[i]] , \
                    marker = markers[int(cons[index_b[i],0])],
                    color = 'black')
            
    # Computation time 
    steps = int(len(t)/len(t_comp))
    ax4.plot(t[::steps], t_comp)
    # Add distinction between integrators
    for i in range(len(index_b)):
        if i < 2:
            ax4.scatter(t[index_b[i]], t_comp[index_b[i]//steps] , \
                    marker = markers[int(cons[index_b[i],0])],
                    color = 'black', label = names[int(cons[index_b[i],0])])
        else:
            ax4.scatter(t[index_b[i]], t_comp[index_b[i]//steps] , \
                    marker = markers[int(cons[index_b[i],0])],
                    color = 'black')
    
    ax4.set_xlabel("Integration time (yr)")
    ax4.set_ylabel("Computation time (s)")

    # annotate_axes(fig)
    plt.show()


