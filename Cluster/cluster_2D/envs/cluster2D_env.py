import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup
import json

# from helpfunctions import load_json
from amuse.units import units, constants, nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.community.hermite.interface import Hermite
from amuse.community.bhtree.interface import BHTree

from plots import plot_state, plot_trajectory

units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
"""
https://www.gymlibrary.dev/content/environment_creation/
"""

class ClusterEnv(gym.Env):
    def __init__(self, render_mode = None, bodies = 4):
        self.size = 4*bodies
        self.settings = load_json("./settings.json")
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.size), \
                                                high=np.array([np.inf]*self.size), \
                                                dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self.settings['Integration']['integrators']))
        self.W = self.settings['Training']['weights']
    
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
        bodies = new_king_model(self.n_stars, W0, convert_nbody = self.converter)
        bodies.scale_to_standard(self.converter)
        bodies.move_to_center()
        bodies.mass = m_stars | units.MSun

        self.G = constants.G

        # Plot initial state
        if self.settings['Integration']['plot'] == True:
            plot_state(bodies)
        return bodies

    def _get_info(self):
        """
        get energy error, angular momentum error at current state
        """
        E_kin = self.particles.kinetic_energy().value_in(units.m**2 * units.s**(-2)* units.kg)
        E_pot = self.particles.potential_energy().value_in(units.m**2 * units.s**(-2)* units.kg)
        L = self.calculate_angular_m(self.gravity.particles)
        # Delta_E = E_kin + E_pot - self.E_0
        Delta_E = E_kin + E_pot
        # Delta_L = L - self.L_0
        Delta_L = L
        return Delta_E, Delta_L
    
    def _get_integrator(self, index):
        if self.settings['Integration']['integrators'][index] == 'BHTree':             
            integrator = BHTree
        elif self.settings['Integration']['integrators'][index] == 'Hermite':             
            integrator = Hermite
        return integrator
    
    def reset(self, options = None):
        seed = self.settings['Integration']['seed']
        # super().reset(seed=seed) # We need to seed self.np_random

        self.particles = self._initial_conditions(seed)

        # Initialize basic integrator and add particles
        integrator = self._get_integrator(0)
        self.gravity = integrator(self.converter)
        self.previous_int = self.settings['Integration']['integrators'][0]

        self.gravity.particles.add_particles(self.particles)
        self.channel = self.gravity.particles.new_channel_to(self.particles)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0 = 0
        self.L_0 = np.array([0,0,0])
        self.E_0, self.L_0 = self._get_info()

        observation = self.gravity
        info = self._get_info()
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration
        self.t0_comp = time.time()

        # Initialize variables to save simulation
        if self.settings['Integration']['savestate'] == True:
            steps = (self.settings['Integration']['check_step']) * \
                    self.settings['Integration']['max_steps']
            self.state = np.zeros((steps, self.n_stars, 8)) # action, mass, rx3, vx3, 
            self.cons = np.zeros((steps, 5)) # action, E, Lx3, 
            self.comp_time = np.zeros((self.settings['Integration']['max_steps'], 1)) # computation time
            self._savestate(0, 0, self.particles, self.E_0, self.L_0) # save initial state

        return observation, info
    
    def _calculate_reward(self, info, T, W):
        Delta_E, Delta_O = info
        return W[0]*Delta_E + W[1]*Delta_O + W[2]*T
    
    def step(self, action):

        check_step = self.settings['Integration']['check_step'] # after how many time steps it is checked
        t_step = self.settings['Integration']['t_step']

        t0_step = time.time()
        # Apply action
        current_int = self.settings['Integration']['integrators'][action]
        if current_int != self.previous_int: # if it's different, change integrator
            self.gravity.stop()
            self.gravity = self._get_integrator(action)(self.converter)
            self.gravity.particles.add_particles(self.particles)
            self.channel = self.gravity.particles.new_channel_to(self.particles)
            self.t_cumul = 0
        elif current_int == self.previous_int and self.iteration > 0: # Do not accumulate for the first iter
            self.t_cumul += t_step*check_step

        # Time steps
        # times = (np.arange(0, (check_step+1)*t_step, t_step) + self.t_cumul) | units.yr 
        times = (np.arange(0, (check_step+1), 1)*t_step + self.t_cumul) | units.yr 

        # Integrate
        for i, t in enumerate(times[1:]):
            self.gravity.evolve_model(t)
            self.channel.copy()
            if self.settings['Integration']['savestate'] == True:
                info = self._get_info()
                self._savestate(action, self.iteration*len(times[1:]) + i, self.particles, info[0], info[1]) # save initial state
        self.previous_int = current_int
        
        # Get information for the reward
        if self.settings['Integration']['savestate'] == False:
            info = self._get_info()
        reward = self._calculate_reward(info, time.time() - t0_step, self.W) # Use computation time for this step, including changing integrator

        self.iteration += 1
        T = time.time() - self.t0_comp  # computation time 
        self.comp_time[self.iteration-1] = T
        np.save(self.settings['Integration']['savefile'] +'_tcomp', self.comp_time)


        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info)

        if self.settings['Integration']['plot'] == True:
            plot_state(self.particles)

        # finish experiment if max number of iterations is reached
        if (self.iteration == self.settings['Integration']['max_steps']):
            terminated = True
            self.gravity.stop()
            plot_trajectory(self.settings)
        else:
            terminated = False

        return self.particles, reward, terminated, info
    
    def close(self): #TODO: needed?
        # if self.window is not None:
        #     pygame.display.quit()
        #     pygame.quit()
        plot_trajectory(self.settings)

    def calculate_angular_m(self, particles):
        """
        return angular momentum (units m, s, kg)
        """
        L = 0
        for i in range(len(particles)):
            r = particles[i].position.value_in(units.m)
            v = particles[i].velocity.value_in(units.m/units.s)
            m = particles[i].mass.value_in(units.kg)
            L += np.cross(r, m*v)
        return L
    
    def calculate_energy(self): # not needed, verified with amuse
        ke = 0
        pe = 0
        for i in range(0, self.n_stars):
            body1 = self.particles[i]
            ke += 0.5 * body1.mass.value_in(units.kg) * np.linalg.norm(body1.velocity.value_in(units.m/units.s))**2
            for j in range(i+1, self.n_stars):
                body2 = self.particles[j]
                pe -= self.G.value_in(units_G) * body1.mass.value_in(units.kg) * body2.mass.value_in(units.kg) / \
                    np.linalg.norm(body2.position.value_in(units.m) - body1.position.value_in(units.m))
        return ke, pe

    def _display_info(self, info):
        print("Iteration: %i/%i, E_E = %0.3E, E_L = %0.3E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0],\
                                 np.linalg.norm(info[1])))
        
    def _savestate(self, action, step, bodies, E, L):
        self.state[step, :, 0] = action
        self.state[step, :, 1] = bodies.mass.value_in(units.kg)
        self.state[step, :, 2:5] = bodies.position.value_in(units.m)
        self.state[step, :, 5:] = bodies.velocity.value_in(units.m/units.s)
        self.cons[step, 0] = action
        self.cons[step, 1] = E
        self.cons[step, 2:] = L

        np.save(self.settings['Integration']['savefile'] +'_state', self.state)
        np.save(self.settings['Integration']['savefile'] +'_cons', self.cons)

def load_json(filepath):
    """
    load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data

# Test environment
# settings = load_json("./settings.json")
# a = ClusterEnv(settings = settings)
# value = [0,1,1,0,0,1,0,0,0,0,1,1,1,0]
# # value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# terminated = False
# i = 0
# a.reset()
# while terminated == False:
#     x, y, terminated, zz = a.step(value[i%len(value)])
#     i += 1

# a.close()

