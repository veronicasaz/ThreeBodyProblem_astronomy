import gym 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from setuptools import setup
import json

from amuse.units import units, constants, nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.community.hermite.interface import Hermite
from amuse.community.symple.interface import symple
from amuse.community.ph4.interface import ph4
from amuse.ext.solarsystem import new_solar_system
from amuse.ext.orbital_elements import  generate_binaries

from pyDOE import lhs

from plots import plot_state, plot_trajectory


"""
https://www.gymlibrary.dev/content/environment_creation/

Questions:
why do the inner planets go slower?
How does the time-scaling works? Not doing it with the correct units
"""
def orbital_period(G, a, Mtot):
    return 2*np.pi*(a**3/(constants.G*Mtot)).sqrt()

class IntegrateEnv(gym.Env):
    def __init__(self, render_mode = None, bodies = 4, suffix = ''):
        self.size = 4*bodies #TODO: fix observation space
        self.settings = load_json("./settings_symple.json")
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.size), \
                                                high=np.array([np.inf]*self.size), \
                                                dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self.settings['Integration']['order'])) # TODO: vector for timestep
        self.W = self.settings['Training']['weights']

        self.suffix = suffix # added info for saving files
    
        self._create_actions()
    # def _convert_units(self, converter, bodies, G):
    #     bodies.position = converter.to_nbody(bodies.position)
    #     print(bodies.position)

    def _create_actions(self):
        if self.settings['actions'] == 'mixed':
            t_step = self.settings['Integration']['t_step']
            order = self.settings['Integration']['order']
            self.actions = list()
            for i in range(len(order)):
                for j in range(len(t_step)):
                    self.actions.append((order[i], t_step[j]))
        elif self.settings['actions'] == 'order':
            self.actions = self.settings['Integration']['order']
        elif self.settings['actions'] == 't_step':
            self.actions = self.settings['Integration']['t_step']

    def _add_bodies(self, bodies):
        n_bodies = self.settings['Integration']['bodies']
        ranges = self.settings['Integration']['ranges']
        ranges_np = np.array(list(ranges.values()))

        K = lhs(len(ranges), samples = n_bodies) * (ranges_np[:, 1]- ranges_np[:, 0]) + ranges_np[:, 0] 

        for i in range(n_bodies):
            sun, particle = generate_binaries(
            bodies[bodies.name=='SUN'].mass,
            K[i, 0] | units.MSun,
            K[i, 1] | units.au,
            eccentricity = K[i, 2],
            inclination = K[i, 3],
            longitude_of_the_ascending_node = K[i, 5],
            argument_of_periapsis = K[i, 4],
            true_anomaly= K[i, 5],
            G = constants.G)
            particle.name = "Particle_%i"%i

            bodies.add_particles(particle)
        
        # remove extra solar system bodies
        bodies_to_be_removed = ['MERCURY', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']
        for b in range(len(bodies_to_be_removed)):
            bodies.remove_particle(bodies[bodies.name==bodies_to_be_removed[b]])
        return bodies


    def _initial_conditions(self, seed):
        # self.settings['Integration']['masses']

        bodies = new_solar_system()
        bodies = self._add_bodies(bodies)
        self.names = bodies.name
        self.n_bodies = len(bodies)

        self.converter = nbody_system.nbody_to_si(bodies.mass.sum(), 1 | units.au)
        
        if self.settings['Integration']['units'] == 'si':
            bodies.scale_to_standard(convert_nbody=self.converter)
            self.G = constants.G
            self.units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
            self.units_energy = units.m**2 * units.s**(-2)* units.kg
            self.units_time = units.yr
            self.units_t = units.s 
            self.units_l = units.m
            self.units_m = units.kg

        elif self.settings['Integration']['units'] == 'nbody':
            self.G = self.converter.to_nbody(constants.G)
            self.units_G = nbody_system.length**3 * nbody_system.mass**(-1) * nbody_system.time**(-2)
            self.units_energy = nbody_system.length**2 * nbody_system.time**(-2)* nbody_system.mass
            self.units_time = self.converter.to_nbody(1 | units.yr)
            self.units_t = nbody_system.time
            self.units_l = nbody_system.length
            self.units_m = nbody_system.mass

        bodies.move_to_center()

        # Plot initial state
        if self.settings['Integration']['plot'] == True:
            plot_state(bodies)
        return bodies

    def _get_info_initial(self):
        """
        get energy error, angular momentum error at current state
        """
        E_kin = self.gravity.particles.kinetic_energy().value_in(self.units_energy)
        E_pot = self.gravity.particles.potential_energy(G = self.G).value_in(self.units_energy)
        L = self.calculate_angular_m(self.gravity.particles)
        return E_kin + E_pot, L
    
    def _get_info(self):
        """
        get energy error, angular momentum error at current state
        """
        E_kin = self.gravity.particles.kinetic_energy().value_in(self.units_energy)
        E_pot = self.gravity.particles.potential_energy(G = self.G).value_in(self.units_energy)
        L = self.calculate_angular_m(self.gravity.particles)
        # Delta_E = E_kin + E_pot - self.E_0
        Delta_E = (E_kin + E_pot - self.E_0) / self.E_0
        # Delta_L = L - self.L_0
        Delta_L = (L - self.L_0) / self.L_0
        return Delta_E, Delta_L
    
    def _strip_units(self, particles): #TODO
        state = np.zeros(())
        return state
    
    def _initialize_integrator(self, integrator, timestep, tstep_param):
        if self.settings['Integration']['units'] == 'si':
            g = symple(self.converter, redirection ='none')
        else:
            g = symple(redirection ='none')
        g.initialize_code()
        # g.parameters.set_defaults()

        # g.parameters.epsilon_squared = (0.01| units.AU)**2
        g.parameters.integrator = integrator
        g.parameters.timestep_parameter = tstep_param
        g.parameters.timestep = timestep | self.units_time
        return g 
    
    def reset(self, options = None):
        seed = self.settings['Integration']['seed']
        # super().reset(seed=seed) # We need to seed self.np_random

        self.particles = self._initial_conditions(seed)

        # Initialize basic integrator and add particles
        if self.settings['actions'] == 'mixed':
            self.gravity = self._initialize_integrator(self.actions[0][0],\
                                                self.actions[0][1], 0.01)
        # elif self.settings['actions'] == 'order':
        #     self.gravity = self._initialize_integrator(self.actions[0][0],\
        #                                         self.actions[0][1], 0.01)
        # else:

        

        self.previous_int = self.settings['Integration']['order'][0]

        self.gravity.particles.add_particles(self.particles)
        self.gravity.commit_particles()

        self.channel = self.gravity.particles.new_channel_to(self.particles)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0, self.L_0 = self._get_info_initial()

        # state = self.gravity.particles
        state = [0, 0, 0, 0, 0] # Computation time, Error_0, L_0x3
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration
        self.t0_comp = time.time()

        # Initialize variables to save simulation
        if self.settings['Integration']['savestate'] == True:
            steps = self.settings['Integration']['max_steps']
            self.state = np.zeros((steps, self.n_bodies, 8)) # action, mass, rx3, vx3, 
            self.cons = np.zeros((steps, 5)) # action, E, Lx3, 
            self.comp_time = np.zeros((self.settings['Integration']['max_steps'], 1)) # computation time
            self._savestate(0, 0, self.particles, self.E_0, self.L_0) # save initial state

        return state, [self.E_0, self.L_0]
    
    def _calculate_reward(self, info, T, W):
        Delta_E, Delta_O = info
        return -(W[0]*Delta_E + W[1]*np.linalg.norm(Delta_O) + W[2]*T)
    
    def step(self, action):

        check_step = self.settings['Integration']['check_step'] # final time for step integration

        t0_step = time.time()
        print("Action", self.actions[action])

        # Apply action
        current_int = self.actions[action][0]
        t_step = self.actions[action][1]
        # if current_int != self.previous_int: # if it's different, change integrator
            # TODO: can this be done without reinitializing?
        self.gravity.parameters.integrator = current_int
        self.gravity.parameters.timestep = t_step | self.units_time
            # self.t_cumul = 0
        # elif current_int == self.previous_int and self.iteration > 0: # Do not accumulate for the first iter
        self.t_cumul += check_step
        # Integrate
        print("Iteration", self.iteration)
        # self.gravity.evolve_model(check_step | self.units_time)
        print(self.gravity.particles[3])
        self.gravity.evolve_model(self.t_cumul | self.units_time)
        self.channel.copy()
        print(self.gravity.particles[3])

        # for i, t in enumerate(times[1:]):
            # self.gravity.evolve_model(t)
            # self.channel.copy()
        if self.settings['Integration']['savestate'] == True:
            info = self._get_info()
            self._savestate(action, self.iteration, self.gravity.particles, info[0], info[1]) # save initial state
        self.previous_int = current_int
        
        # Get information for the reward
        T = time.time() - t0_step
        info = self._get_info()
        state = [T, info[0], info[1][0], info[1][1], info[1][2]] # TODO: change to particles positions and velocities
        reward = self._calculate_reward(info, T, self.W) # Use computation time for this step, including changing integrator

        self.iteration += 1
        self.comp_time[self.iteration-1] = T
        if self.settings['Integration']['savestate'] == True:
            np.save(self.settings['Integration']['savefile'] +'_tcomp' + self.suffix, self.comp_time)


        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info)

        if self.settings['Integration']['plot'] == True:
            plot_state(self.gravity.particles)

        # finish experiment if max number of iterations is reached
        if (self.iteration == self.settings['Integration']['max_steps']):
            terminated = True
            self.gravity.stop()
            # plot_trajectory(self.settings)
        else:
            terminated = False

        return state, reward, terminated, info
    
    def close(self): #TODO: needed?
        # if self.window is not None:
        #     pygame.display.quit()
        #     pygame.quit()
        # plot_trajectory(self.settings)
        self.gravity.stop()

    def calculate_angular_m(self, particles):
        """
        return angular momentum (units m, s, kg)
        """
        L = 0
        for i in range(len(particles)):
            r = particles[i].position.value_in(self.units_l)
            v = particles[i].velocity.value_in(self.units_l/self.units_t)
            m = particles[i].mass.value_in(self.units_m)
            L += np.cross(r, m*v)
        return L
    
    def calculate_energy(self): # not needed, verified with amuse
        ke = 0
        pe = 0
        for i in range(0, self.n_bodies):
            body1 = self.particles[i]
            ke += 0.5 * body1.mass.value_in(self.units_m) * np.linalg.norm(body1.velocity.value_in(self.units_l/self.units_t))**2
            for j in range(i+1, self.n_bodies):
                body2 = self.particles[j]
                pe -= self.G.value_in(self.units_G) * body1.mass.value_in(self.units_m) * body2.mass.value_in(self.units_m) / \
                    np.linalg.norm(body2.position.value_in(self.units_l) - body1.position.value_in(self.units_l))
        return ke, pe

    def _display_info(self, info):
        print("Iteration: %i/%i, E_E = %0.3E, E_L = %0.3E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0],\
                                 np.linalg.norm(info[1])))
        
    def _savestate(self, action, step, bodies, E, L):
        self.state[step, :, 0] = action
        self.state[step, :, 1] = bodies.mass.value_in(self.units_m)
        self.state[step, :, 2:5] = bodies.position.value_in(self.units_l)
        self.state[step, :, 5:] = bodies.velocity.value_in(self.units_l/self.units_t)
        self.cons[step, 0] = action
        self.cons[step, 1] = E
        self.cons[step, 2:] = L

        np.save(self.settings['Integration']['savefile'] +'_state'+ self.suffix, self.state)
        np.save(self.settings['Integration']['savefile'] +'_cons'+ self.suffix, self.cons)
    
    def loadstate(self):
        state = np.load(self.settings['Integration']['savefile'] +'_state'+ self.suffix+'.npy')
        cons = np.load(self.settings['Integration']['savefile'] +'_cons'+ self.suffix+'.npy')
        return state, cons

    def plot_orbit(self):
        state, cons = self.loadstate()

        n_bodies = np.shape(state)[1]

        for i in range(n_bodies):
            plt.plot(state[:, i, 2], state[:, i, 3], marker= 'o', label = self.names[i])
        plt.legend()
        plt.show()

        
def load_json(filepath):
    """
    load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data


# Test environment

def run_many_symple_cases(values):
    cases = len(values)
    a = IntegrateEnv()
    for j in range(cases):
        print("Case %i"%j)
        print(values[j][0])
        print(a.actions[values[j][0]])
        a.suffix = ('_case%i'%j)
        value = values[j]
        terminated = False
        i = 0
        a.reset()
        while terminated == False:
            x, y, terminated, zz = a.step(value[i%len(value)])
            i += 1
        a.close()
        # a.plot_orbit()

def evaluate_many_symple_cases(values, \
                        plot_tstep = True, \
                        plot_grid = True):
    cases = len(values)
    a = IntegrateEnv()

    # Load run information
    state = list()
    cons = list()
    for i in range(cases):
        a.suffix = ('_case%i'%i)
        state.append(a.loadstate()[0])
        cons.append(a.loadstate()[1])


    bodies = len(state[0][0,:,0])
    steps = len(cons[0][:,0])
    
    # # Calculate errors in cartesian
    # baseline = state[4] # Take highest order as a baseline
    # E_x = np.zeros((steps, bodies))
    # for i in range(cases):
    #     E_x[:, i] = state[i]

    # Calculate the energy errors
    # baseline = cons[4] # Take highest order as a baseline
    E_E = np.zeros((steps, cases))
    E_M = np.zeros((steps, cases))
    for i in range(cases):
        # E_E[:, i] = abs((cons[i][:, 1] - cons[i][0, 1])/ cons[i][0, 1]) # absolute relative energy error
        # E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:])/ cons[i][0, 2:], axis = 1) # relative angular momentum error
        E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
        E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error


    # plot
    colors = ['red', 'green', 'blue', 'orange', 'grey', 'yellow', 'black']
    lines = ['-', '--', ':', '-.', '-', '--', ':', '-.' ]
    # labels = ['Order 1', 'Order 2', 'Order 3', 'Order 5', 'Order 10', 'Mixed order']
    n_order_cases = len(a.settings['Integration']['order'])
    n_tstep_cases = len(a.settings['Integration']['t_step'])

    if plot_tstep == True:
        fig, ax = plt.subplots(2, 1, layout = 'constrained', figsize = (10, 10))
        x_axis = np.arange(0, steps, 1)

        for i in range(cases):
            ax[0].plot(x_axis, E_E[:, i], color = colors[i//n_tstep_cases], linestyle = lines[i%n_tstep_cases], label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))
            ax[1].plot(x_axis, E_M[:, i], color = colors[i//n_tstep_cases], linestyle = lines[i%n_tstep_cases], label = 'O%i, t_step = %1.1E'%(a.actions[values[i][0]][0], a.actions[values[i][0]][1]))

        labelsize = 20
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('Energy error',  fontsize = labelsize)
        ax[1].set_ylabel('Angular momentum error', fontsize = labelsize)
        ax[1].set_xlabel('Step', fontsize = labelsize)
        for pl in range(len(ax)):
            ax[pl].set_yscale('log')
            ax[pl].tick_params(axis='both', which='major', labelsize=labelsize)
        plt.show()
    
    if plot_grid == True:
        fig, ax = plt.subplots(2, 1, layout = 'constrained', figsize = (10, 10))
        print(a.actions)
        print(np.array(a.actions)[:,0])
        print(E_E[-1,:])

        # for i in range(cases): #Excluding the case with the mixed actions
        sc = ax[0].scatter(np.array(a.actions)[:,0], np.array(a.actions)[:,1], marker = 'o',\
                           s = 500, c = E_E[-1,:-1], cmap = 'RdYlBu', \
                            norm=matplotlib.colors.LogNorm())
        plt.colorbar(sc, ax = ax[0])

        sm = ax[1].scatter(np.array(a.actions)[:,0], np.array(a.actions)[:,1], marker = 'o',\
                           s = 500, c = E_M[-1,:-1], cmap = 'RdYlBu', \
                            norm=matplotlib.colors.LogNorm())
        plt.colorbar(sm, ax = ax[1])

        labelsize = 20
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('t_step',  fontsize = labelsize)
        ax[0].set_title("Final Energy error", fontsize = labelsize)
        ax[1].set_ylabel('t_step', fontsize = labelsize)
        ax[1].set_xlabel('Order', fontsize = labelsize)
        ax[0].set_title("Final Angular momentum error", fontsize = labelsize)
        for pl in range(len(ax)):
            ax[pl].set_yscale('log')
            ax[pl].tick_params(axis='both', which='major', labelsize=labelsize)
        plt.show()

def test_1_case():
    a = IntegrateEnv()
    terminated = False
    i = 0
    a.reset()
    value = values[0]
    print(value)

    while terminated == False:
        x, y, terminated, zz = a.step(value[i%len(value)])
        i += 1
    a.close()
    a.plot_orbit()


# Run all possibilities
a = IntegrateEnv()
steps = 30 # large value just in case
values = list()
for i in range(len(a.actions)):
    values.append([i]*steps)

# values = values[0:4]
values.append(np.random.randint(0, len(a.actions), size = steps))
# run_many_symple_cases(values)
evaluate_many_symple_cases(values, \
                           plot_tstep = False, \
                           plot_grid = True)

# test_1_case()




