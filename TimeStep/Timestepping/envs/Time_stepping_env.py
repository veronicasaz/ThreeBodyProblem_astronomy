import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup
import json

from amuse.units import units, constants, nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.community.hermite.interface import Hermite
from amuse.community.bhtree.interface import BHTree
from amuse.community.ph4.interface import ph4

from plots import plot_state, plot_trajectory


"""
https://www.gymlibrary.dev/content/environment_creation/
"""

class TimeSteppingEnv(gym.Env):
    def __init__(self, render_mode = None, bodies = 4, namesave = ""):
        self.size = 4*bodies
        self.settings = load_json("./settings_timestep.json")
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.size), \
                                                high=np.array([np.inf]*self.size), \
                                                dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self.settings['Integration']['tstep_param']))
        self.W = self.settings['Training']['weights']
        self.namesave = namesave
    
    # def _convert_units(self, converter, bodies, G):
    #     bodies.position = converter.to_nbody(bodies.position)
    #     print(bodies.position)

    def _initial_conditions(self, seed):
        #TODO: setup intial conditions
        self.n_stars = self.settings['Integration']['bodies']
        if self.settings['Integration']['masses'] == 'equal':
            m_stars = np.ones(self.n_stars)/ self.n_stars # sum of masses is 1
        elif self.settings['Integration']['masses'] == 'equal_sun':
            m_stars = np.ones(self.n_stars) # each particle is 1 MSun
        else:
            m_stars = np.ones(self.n_stars) * self.settings['Integration']['masses']

        r_cluster = self.settings['Integration']['r_cluster'] | units.au
        self.converter = nbody_system.nbody_to_si(m_stars.sum() | units.MSun, r_cluster)
        W0 = 3
        if self.settings['Integration']['units'] == 'si':
            np.random.seed(seed)
            bodies = new_king_model(self.n_stars, W0, self.converter)
            bodies.scale_to_standard(convert_nbody=self.converter)
            self.G = constants.G
            self.units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
            self.units_energy = units.m**2 * units.s**(-2)* units.kg
            self.units_time = units.yr
            self.units_t = units.s 
            self.units_l = units.m
            self.units_m = units.kg

        elif self.settings['Integration']['units'] == 'nbody':
            np.random.seed(seed)
            bodies = new_king_model(self.n_stars, W0)
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

    def _get_info(self):
        """
        get energy error, angular momentum error at current state
        """
        E_kin = self.particles.kinetic_energy().value_in(self.units_energy)
        E_pot = self.particles.potential_energy(G = self.G).value_in(self.units_energy)
        L = self.calculate_angular_m(self.gravity.particles)
        # Delta_E = E_kin + E_pot - self.E_0
        Delta_E = E_kin + E_pot
        # Delta_L = L - self.L_0
        Delta_L = L
        return Delta_E, Delta_L
    
    def _get_integrator(self):
        if self.settings['Integration']['integrators']== 'BHTree':             
            integrator = BHTree
        elif self.settings['Integration']['integrators'] == 'Hermite':             
            integrator = Hermite
        return integrator
    
    def _strip_units(self, particles): #TODO
        state = np.zeros(())
        return state
    
    def reset(self, options = None):
        seed = self.settings['Integration']['seed']
        # super().reset(seed=seed) # We need to seed self.np_random

        self.particles = self._initial_conditions(seed)

        # Initialize basic integrator and add particles
        # integrator = self._get_integrator()
        # self.gravity = integrator()
        # self.gravity.parameters.timestep_parameter = self.settings['Integration']['tstep_param'][0]
        self.gravity = ph4()
        self.gravity.parameters.timestep_parameter = self.settings['Integration']['tstep_param'][0]

        self.gravity.particles.add_particles(self.particles)
        self.channel = self.gravity.particles.new_channel_to(self.particles)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0 = 0
        self.L_0 = np.array([0,0,0])
        self.E_0, self.L_0 = self._get_info()

        # state = self.gravity.particles
        state = [0, 0, 0, 0, 0] # Computation time, Error_0, L_0x3
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration
        self.t0_comp = time.time()

        # Initialize variables to save simulation
        if self.settings['Integration']['savestate'] == True:
            steps = (self.settings['Integration']['check_step']) * \
                    self.settings['Integration']['max_steps']
            self.state = np.zeros((steps, self.n_stars, 8)) # action, mass, rx3, vx3, 
            self.cons = np.zeros((steps, 6)) # action, E, Lx3, t
            self.comp_time = np.zeros((self.settings['Integration']['max_steps'], 1)) # computation time
            self._savestate(0, 0, self.particles, self.E_0, self.L_0, 0) # save initial state

        return state, [self.E_0, self.L_0]
    
    def _calculate_reward(self, info, T, W):
        Delta_E, Delta_O = info
        return -(W[0]*Delta_E + W[1]*np.linalg.norm(Delta_O) + W[2]*T)
    
    def step(self, action):

        check_step = self.settings['Integration']['check_step'] # after how many time steps it is checked
        t_step = self.settings['Integration']['t_step']
        self.t_cumul += t_step*check_step

        t0_step = time.time()

        # Apply action
        current_int = self.settings['Integration']['integrators'][action]

        # Time steps
        times = (np.arange(0, (check_step+1), 1)*t_step + self.t_cumul) * self.units_time
        self.gravity.parameters.timestep_parameter = self.settings['Integration']['tstep_param'][action]

        # Integrate
        for i, t in enumerate(times[1:]):
            self.gravity.evolve_model(t)
            self.channel.copy()
            if self.settings['Integration']['savestate'] == True:
                info = self._get_info()
                self._savestate(action, self.iteration*len(times[1:]) + i, self.particles, info[0], info[1], t.value_in(self.units_t)) # save initial state
        self.previous_int = current_int
        
        # Get information for the reward
        T = time.time() - t0_step
        info = self._get_info()
        state = [T, info[0], info[1][0], info[1][1], info[1][2]]
        reward = self._calculate_reward(info, T, self.W) # Use computation time for this step, including changing integrator

        self.iteration += 1
        self.comp_time[self.iteration-1] = T
        if self.settings['Integration']['savestate'] == True:
            np.save(self.settings['Integration']['savefile'] +'_tcomp' + self.namesave, self.comp_time)


        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info)

        if self.settings['Integration']['plot'] == True:
            plot_state(self.particles)

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
        self.gravity.stop()
        plot_trajectory(self.settings, self.namesave)

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
        for i in range(0, self.n_stars):
            body1 = self.particles[i]
            ke += 0.5 * body1.mass.value_in(self.units_m) * np.linalg.norm(body1.velocity.value_in(self.units_l/self.units_t))**2
            for j in range(i+1, self.n_stars):
                body2 = self.particles[j]
                pe -= self.G.value_in(self.units_G) * body1.mass.value_in(self.units_m) * body2.mass.value_in(self.units_m) / \
                    np.linalg.norm(body2.position.value_in(self.units_l) - body1.position.value_in(self.units_l))
        return ke, pe

    def _display_info(self, info):
        print("Iteration: %i/%i, E_E = %0.3E, E_L = %0.3E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0],\
                                 np.linalg.norm(info[1])))
        
    def _savestate(self, action, step, bodies, E, L, t):
        self.state[step, :, 0] = action
        self.state[step, :, 1] = bodies.mass.value_in(self.units_m)
        self.state[step, :, 2:5] = bodies.position.value_in(self.units_l)
        self.state[step, :, 5:] = bodies.velocity.value_in(self.units_l/self.units_t)
        self.cons[step, 0] = action
        self.cons[step, 1] = E
        self.cons[step, 2:5] = L
        self.cons[step, 5] = t

        np.save(self.settings['Integration']['savefile'] +'_state'+ self.namesave, self.state)
        np.save(self.settings['Integration']['savefile'] +'_cons'+ self.namesave, self.cons)

def load_json(filepath):
    """
    load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data

def test_environment(value, name):
    # Test environment
    # settings = load_json("./settings.json")
    a = TimeSteppingEnv(namesave = name)
    value = [0,1,1,3,2,4,0,1,4,2,1,0,3,3]
    # value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    terminated = False
    i = 0
    a.reset()
    while terminated == False:
        x, y, terminated, zz = a.step(value[i%len(value)])
        i += 1

    # a.close()

def load_trajectory(name):
    state = np.load("./TimeStep/Timestepping/simulations/state_state"+name+'.npy')
    cons = np.load("./TimeStep/Timestepping/simulations/state_cons"+name+'.npy')
    tcomp = np.load("./TimeStep/Timestepping/simulations/state_tcomp"+name+'.npy')
    return state, cons, tcomp

# def plot_comparison(state1, state2, state3):
#     plt.plot()
def plot_error(cons):
    for i in range(len(cons)):
        E = (cons[i][:, 1] - cons[i][0, 1]) 
        plt.plot(cons[i][:, -1], E)
    plt.show()

# Simulate
value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
test_environment(value, '_1')
value = [4,4,4,4,4,4,4,4,4,4,4,4,4,4]
test_environment(value, '_2')
value = [0,1,1,3,2,4,0,1,4,2,1,0,3,3]
test_environment(value, '_3')

state1, cons1, tcomp1 = load_trajectory('_1')
state2, cons2, tcomp2 = load_trajectory('_2')
state3, cons3, tcomp3 = load_trajectory('_3')

plot_error([cons1, cons2, cons3])


    



