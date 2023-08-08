import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup

from helpfunctions import load_json
from amuse.units import units, constants, nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.community.hermite.interface import Hermite
from amuse.community.bhtree.interface import BHTree

"""
https://www.gymlibrary.dev/content/environment_creation/
"""

class Cluster_2D(gym.Env):
    def __init__(self, settings, render_mode = None, bodies = 4):
        self.size = 4*bodies
        self.settings = settings
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
            m_stars = self.settings['Integration']['masses']

        r_cluster = self.settings['Integration']['r_cluster'] | units.au
        self.converter = nbody_system.nbody_to_si(m_stars.sum() | units.MSun, r_cluster)
        W0 = 3
        bodies = new_king_model(self.n_stars, W0, convert_nbody = self.converter)
        bodies.scale_to_standard(self.converter)
        bodies.mass = m_stars | units.kg

        self.G = constants.G

        # Plot initial state
        if self.settings['Integration']['plot'] == True:
            self.plot_state(bodies)
        return bodies
    
    def _get_info(self):
        """
        get energy error, angular momentum error at current state
        """
        E_kin = self.gravity.get_kinetic_energy().value_in(units.m**2 * units.s**(-2)* units.kg)
        E_pot = self.gravity.get_potential_energy(self.G).value_in(units.m**2 * units.s**(-2)* units.kg)
        L = self.calculate_angular_m(self.gravity.particles)
        Delta_E = E_kin + E_pot - self.E_0
        Delta_L = L - self.L_0
        # self.current_state
        # TODO: calculate energy error and angular momentum error (we need initial state)
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
        self.t_cumul = 0 # cumulative time for integration

        # Initialize variables to save simulation
        if self.settings['Integration']['savestate'] == True:
            steps = (self.settings['Integration']['check_step']-1) * \
                    self.settings['Integration']['max_steps']
            print("Steps", steps)
            self.state = np.zeros((steps, self.n_stars, 7)) # mass, rx3, vx3, 
            self.cons = np.zeros((steps, 4)) # E, Lx3
            self._savestate(0, self.particles, self.E_0, self.L_0) # save initial state

        return observation, info
    
    def _calculate_reward(self, info, T, W):
        Delta_E, Delta_O = info
        return W[0]*Delta_E + W[1]*Delta_O + W[2]*T
    
    def step(self, action):
        
        # Time steps
        check_step = self.settings['Integration']['check_step'] # after how many time steps it is checked
        t_step = self.settings['Integration']['t_step']
        times = (np.arange(0, check_step*t_step, t_step) + self.t_cumul) | units.yr 

        # Apply action
        current_int = self.settings['Integration']['integrators'][action]
        if current_int != self.previous_int: # if it's different, change integrator
            self.gravity.stop()
            self.gravity = self._get_integrator(action)(self.converter)
            self.channel = self.gravity.particles.new_channel_to(self.particles)
            self.gravity.particles.add_particles(self.particles)
            self.t_cumul = 0
        else:
            self.t_cumul += check_step*t_step

        # print("=========================================================")
        # print(self.gravity.particles)
        # print(self.particles)
        
        # Integrate 
        t0 = time.time()
        for i, t in enumerate(times[1:]):
            self.gravity.evolve_model(t)
            self.channel.copy()
            if self.settings['Integration']['savestate'] == True:
                info = self._get_info()
                self._savestate(self.iteration*len(times[1:]) + i, self.particles, info[0], info[1]) # save initial state
        T = time.time() - t0 # computation time 
        self.previous_int = current_int
        
        # Get information for the reward
        if self.settings['Integration']['savestate'] == False:
            info = self._get_info()
        reward = self._calculate_reward(info, T, self.W)

        self.iteration += 1

        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info)

        if self.settings['Integration']['plot'] == True:
            self.plot_state(self.particles)

        # finish experiment if max number of iterations is reached
        if (self.iteration == self.settings['Integration']['max_steps']):
            terminated = True
            self.gravity.stop()
            self.plot_trajectory()
        else:
            terminated = False

        return self.particles, reward, terminated, False, info
    
    def close(self): #TODO: needed?
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

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

    def _display_info(self, info):
        print("Iteration: %i/%i, E_E = %0.3E, E_L = %0.3E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0],\
                                 np.linalg.norm(info[1])))
        
    def _savestate(self, step, bodies, E, L):
        self.state[step, :, 0] = bodies.mass.value_in(units.kg)
        self.state[step, :, 1:4] = bodies.position.value_in(units.m)
        self.state[step, :, 4:] = bodies.velocity.value_in(units.m/units.s)
        self.cons[step, 0] = E
        self.cons[step, 1:] = L

        np.save(settings['Integration']['savefile'] +'_state', self.state)
        np.save(settings['Integration']['savefile'] +'_cons', self.cons)


    def plot_state(self, bodies):
        v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
        plt.scatter(bodies.x.value_in(units.au),\
                    bodies.y.value_in(units.au), \
                    c=v.value_in(units.kms), alpha=0.5)
        plt.colorbar()
        plt.show()

    # def plot_error(self):
        # plt.plot(bodies.x.value_in(units.au),\
        #             bodies.y.value_in(units.au), \
        #             c=v.value_in(units.kms), alpha=0.5)
        # plt.colorbar()
        # plt.show()

    def plot_trajectory(self):
        state = np.load(settings['Integration']['savefile'] +'_state.npy')
        cons = np.load(settings['Integration']['savefile'] +'_cons.npy')

        for i in range(self.n_stars):
            plt.scatter(state[0, i, 1], state[0, i, 2], marker = 'o', s = 200)
            plt.plot(state[:, i, 1], state[:, i, 2], marker = 'o')

        plt.show()

        t = np.arange(0, (self.settings['Integration']['check_step']-1) * \
            self.settings['Integration']['t_step'] * self.settings['Integration']['max_steps'], \
            self.settings['Integration']['t_step'])
        plt.plot(t, cons[:, 0])
        plt.plot(t, np.linalg.norm(cons[:, 1:], axis = 1))
        plt.show()

settings = load_json("./settings.json")
a = Cluster_2D(settings)
a.reset()
value = [0,1,1,0,0,1,0,0,0,0,1,1,1,0]
# value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
terminated = False
i = 0
while terminated == False:
    print(value[i])
    x, y, terminated, z, zz = a.step(value[i])
    i += 1

# gym.envs.registration.register(
#     id='gym_examples/Cluster2D-v0', # 
#     entry_point='gym_examples.envs:Cluster2DEnv',
#     max_episode_steps=300,
# )

# env = gym.make('gym_examples/Cluster2D-v0') # example of how to create the env once it's been registered

