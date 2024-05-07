"""
Bridged2Body_env: environment for integration of a 2 body problem using bridge

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024

Based on https://www.gymlibrary.dev/content/environment_creation/
pip install -e ENVS needed to create a package
"""

import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup
import json

import threading

from pyDOE import lhs

from amuse.units import units, constants, nbody_system
from amuse.units.quantities import sign
from amuse.community.hermite.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.symple.interface import symple
from amuse.community.huayno.interface import Huayno
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.lab import Particles, new_powerlaw_mass_distribution
from amuse.ext.bridge import bridge
from amuse.ext.orbital_elements import get_orbital_elements_from_arrays
from amuse.ic import make_planets_oligarch

def plot_state(bodies):
    v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
    plt.scatter(bodies.x.value_in(units.au),\
                bodies.y.value_in(units.au), \
                c=v.value_in(units.kms), alpha=0.5)
    plt.colorbar()
    plt.show()


def load_json(filepath):
    """
    load_json: load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data

    
class ThreeBodyProblem_env(gym.Env):
    def __init__(self, render_mode = None):
        self.settings = load_json("./settings_integration_3BP.json")
        self.n_bodies = self.settings['InitialConditions']['n_bodies']

        self._initialize_RL()


    def _initialize_RL(self):
        # STATE
        if self.settings["RL"]["state"] == 'cart':
            self.observation_space_n = 4*self.n_bodies+1
            self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.observation_space_n), \
                                                high=np.array([np.inf]*self.observation_space_n), \
                                                dtype=np.float64)
            
        # ACTION
        # From a range of paramters
        if self.settings['RL']['action'] == 'range':
            low = self.settings['RL']['range_action'][0]
            high = self.settings['RL']['range_action'][-1]
            n_actions = self.settings['RL']['number_actions']
            self.actions = np.logspace(np.log10(low), np.log10(high), \
                                       num = n_actions, base = 10,
                                       endpoint = True)
        
        self.action_space = gym.spaces.Discrete(len(self.actions)) 

        # Training parameters
        self.W = self.settings['RL']['weights']

    def _initial_conditions(self):

        bodies = Particles(self.settings['InitialConditions']['n_bodies'])
        ranges = self.settings['Integration']['ranges_triple']
        ranges_np = np.array(list(ranges.values()))

        bodies.mass = (1.0, 1.0, 1.0) | units.MSun
        bodies[0].position = (0.0, 0.0, 0.0) | units.au

        np.random.seed(seed = self.settings['InitialConditions']['seed'])
        K = np.random.uniform(low = ranges_np[:, 0], high = ranges_np[:, 1])
        
        bodies[1].position = (K[0], K[1], 0.0) | units.au
        bodies[2].position = (K[2], K[3], 0.0) | units.au
        bodies[0].velocity = (0.0, 10.0, 0.0) | units.kms
        bodies[1].velocity = (-10.0, 0.0, 0.0) | units.kms
        self.converter = nbody_system.nbody_to_si(bodies.mass.sum(), 1 | units.au)
        bodies.move_to_center()

        return bodies

    ## BASIC FUNCTIONS
    def reset(self):
        """
        reset: reset the simulation 
        INPUTS:
            seed: choose the random seed
            steps: simulation steps to be taken
            typereward: type or reward to be applied to the problem
            save_state: save state (True or False)
        OUTPUTS:
            state_RL: state vector to be passed to the RL
            info_prev: information vector of the previous time step (zero vector)
        """
        # Select units
        self.units()

        # Same time step for the integrators and the bridge
        self.particles = self._initial_conditions()

        # TODO: tstep not implemented for now
        self.gravity = self._initialize_integrator(self.settings['Integration']['integrator'])
        self.gravity = self._apply_action(self.actions[0])
        self.gravity.particles.add_particles(self.particles)
            
        self.channel = self.gravity.particles.new_channel_to(self.particles)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0 = self._get_info(self.particles, initial = True)
        self.info_prev = 0.0
        
        # Create state vector
        state_RL = self._get_state(self.particles, 1.0) # |r_i|, |v_i|

        # Initialize time
        self.check_step = self.settings['Integration']['check_step']
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration

        # Plot trajectory
        if self.settings['Integration']['plot'] == True:
            plot_state(self.particles_joined)

        # Initialize variables to save simulation information
        if self.settings['Integration']['savestate'] == True:
            steps = self.settings['Integration']['max_steps'] + 1 # +1 to account for step 0
            self.state = np.zeros((steps, self.n_bodies, 8)) # action, mass, rx3, vx3
            self.cons = np.zeros((steps, 3)) # action, reward, E
            self.comp_time = np.zeros(steps) # computation time
            self._savestate(0, 0, self.particles, 0.0, 0.0, 0.0) # action, step, particles, E, T, R

        self.info_prev = 0.0
        return state_RL, self.info_prev
    
    def step(self, action):
        """
        step: take a step with a given action, evaluate the results
        INPUTS:
            action: integer corresponding to the action taken
        OUTPUTS:
            state: state of the system to be given to the RL algorithm
            reward: reward value obtained after a step
            terminated: True or False, whether to stop the simulation
            info: additional info to pass to the RL algorithm
        """
        self.iteration += 1
        self.t_cumul += self.check_step # add the previous simulation time
        t = (self.t_cumul) | self.units_time

        # Integrate
        self.gravity = self._apply_action(self.actions[action])
        t0_step = time.time()
        self.gravity.evolve_model(t)
        T = time.time() - t0_step

        self.channel.copy()
            
        # Get information for the reward
        info_error = self._get_info(self.particles)
        state = self._get_state(self.particles, info_error)
        reward = self._calculate_reward(info_error, self.info_prev, T, self.actions[action], self.W) # Use computation time for this step, including changing integrator
        self.info_prev = info_error
        
        if self.settings['Integration']['savestate'] == True:
            self._savestate(action, self.iteration, self.particles, \
                            info_error, T, reward) # save initial state
        
        # finish experiment if max number of iterations is reached
        if (abs(info_error) > self.settings['Integration']['max_error_accepted']) or\
              self.iteration == self.settings['Integration']['max_steps']:
            terminated = True
        else:
            terminated = False
            
        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info_error, reward, action)

        # Plot trajectory
        if self.settings['Integration']['plot'] == True and terminated == True:
            plot_state(self.particles)

        info = dict()
        info['TimeLimit.truncated'] = False
        info['Energy_error'] = info_error
        info['tcomp'] = T

        return state, reward, terminated, info
    
    def close(self):
        self.gravity()

    ## ADDITIONAL FUNCTIONS NEEDED
    def units(self):
        # Choose set of units for the problem
        if self.settings['InitialConditions']['units'] == 'si':
            self.G = constants.G
            self.units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
            self.units_energy = units.m**2 * units.s**(-2)* units.kg
            self.units_time = units.Myr

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
    
    def _initialize_integrator(self, integrator_type):
        """
        _initialize_integrator: initialize chosen integrator with the converter and parameters
        INPUTS:
            action: choice of the action for a parameter that depends on the integrator. 
        Options:
            - Hermite: action is the time-step parameter
            - Ph4: action is the time-step parameter
            - Huayno: action is the time-step parameter
            - Symple: action is the time-step size
        OUTPUTS:
            g: integrator
        """
        if integrator_type == 'Hermite': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = Hermite(self.converter)
            else:
                g = Hermite()
            # Collision detection and softening
            # g.stopping_conditions.timeout_detection.enable()
            # g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2
        elif integrator_type == 'Ph4': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = ph4(self.converter, number_of_workers = 1)
            else:
                g = ph4()
            # g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif integrator_type == 'Huayno': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = Huayno(self.converter)
            else:
                g = Huayno()
            # g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif integrator_type == 'Symple': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = symple(self.converter, redirection ='none')
            else:
                g = symple(redirection ='none')
            g.initialize_code()
            
        return g 
    
    def _apply_action(self, action, integrator_type, g):
        if integrator_type == 'Hermite':
            g.parameters.dt_param = action
        elif integrator_type == "Ph4" or \
            integrator_type == "Huayno":
            g.parameters.dt_param =  action
        else:
            g.parameters.timestep = action | self.units_time

        return g

    
    def _get_info(self, particles, initial = False): # change to include multiple energies
        """
        _get_info: get energy error, angular momentum error at current state
        OUTPUTS:
            Step energy error
        """
        E_kin = particles.kinetic_energy().value_in(self.units_energy)
        E_pot = particles.potential_energy(G = self.G).value_in(self.units_energy)
        E_total = E_kin + E_pot
        
        # L = self.calculate_angular_m(particles)
        if initial == True:
            return E_total
        else:
            # Delta_E_total = (E_total - self.E_0_total)
            Delta_E_total = (E_total - self.E_0)/self.E_0
            return Delta_E_total
                    

    def _get_state(self, particles, E):  # TODO: change to include all particles?
        """
        _get_state: create the state vector
        Options:
            - norm: norm of the positions and velocities of each body and the masses
            - cart: 2D cartesian coordinates of the position and angular momentum plus the energy error
            - dis: distance between particles in position and momentum space plus the energy error

        OUTPUTS: 
            state: state array to be given to the reinforcement learning algorithm
        """
        particles_p_nbody = self.converter.to_generic(particles[0:self.n_stars].position).value_in(nbody_system.length)
        particles_v_nbody = self.converter.to_generic(particles[0:self.n_stars].velocity).value_in(nbody_system.length/nbody_system.time)
        particles_m_nbody = self.converter.to_generic(particles[0:self.n_stars].mass).value_in(nbody_system.mass)

        if self.settings['RL']['state'] == 'norm':
            state = np.zeros((self.n_bodies)*3) # m, norm r, norm v

            state[0:self.n_bodies] = particles_m_nbody
            state[self.n_bodies: 2*self.n_bodies]  = np.linalg.norm(particles_p_nbody, axis = 1)
            state[2*self.n_bodies: 3*self.n_bodies] = np.linalg.norm(particles_v_nbody, axis = 1)
       
        elif self.settings['RL']['state'] == 'cart':
            state = np.zeros((self.n_bodies)*4+1) # all r, all v
            for i in range(self.n_bodies):
                state[2*i:2*i+2] = particles_p_nbody[i, 0:2]/10 # convert to 2D. Divide by 10 to same order as v
                state[2*self.n_bodies + 2*i: 2*self.n_bodies + 2*i+2] = particles_v_nbody[i, 0:2]
                state[-1] = -np.log10(abs(E))
        
        elif self.settings['RL']['state'] == 'dist':
            state = np.zeros((self.n_bodies)*2) # dist r, dist v

            counter = 0
            for i in range(self.n_bodies):
                for j in range(i+1, self.n_bodies):
                    state[counter]  = np.linalg.norm(particles_p_nbody[i,:]-particles_p_nbody[j,:], axis = 0) /10
                    state[self.n_bodies+counter ] = np.linalg.norm(particles_v_nbody[i,:]-particles_v_nbody[j,:], axis = 0)
                    counter += 1

            state[-1] = -np.log10(abs(E))

        return state
    
    def _calculate_reward(self, info, info_prev, T, action, W):
        """
        _calculate_reward: calculate the reward associated to a step
        INPUTS:
            info: energy error and change of angular momentum of iteration i
            info_prev: energy error and change of angular momentum of iteration i-1
            T: clock computation time
            action: action taken. Integer value
            W: weights for the terms in the reward function
        OUTPUTS:
            a: reward value
        """
        Delta_E = info
        Delta_E_prev = info_prev

        if Delta_E_prev == 0.0: # for the initial step
            return 0
        else:
            if self.settings['RL']['reward_f'] == 0: 
                a = -(W[0]* np.log10(abs(Delta_E)) + \
                         W[1]*(np.log10(abs(Delta_E))-np.log10(abs(Delta_E_prev)))) *\
                        (W[2]/abs(np.log10(action)) )
                return a
            
            if self.settings['RL']['reward_f'] == 1:
                a = -(W[0]* abs(np.log10(abs(Delta_E)/1e-8))/\
                         abs(np.log10(abs(Delta_E)))**2 +\
                         W[1]*(np.log10(abs(Delta_E))-np.log10(abs(Delta_E_prev))))*\
                         W[2]/abs(np.log10(action))
                return a
            
            elif self.settings['RL']['reward_f'] == 2:
                a = Delta_E
                a = -W[0]*np.log10(abs(a)) + \
                    W[2]/abs(np.log10(action))
                return a
    
    def _display_info(self, info, reward, action):
        """
        _display_info: display information at every step
        INPUTS:
            info: energy error and angular momentum vector
            reward: value of the reward for the given step
            action: action taken at this step
        """
        print("Iteration: %i/%i, E_E = %0.3E,  Action: %i, Reward: %.4E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info,\
                                 action, \
                                 reward))
            
    def _savestate(self, action, step, particles, E, T, R):
        """
        _savestate: save state of the system to file
        INPUTS:
            action: action taken
            step: simulation step
            particles: particles set
            E: energy error
            L: angular momentum
        """
        self.state[step, :, 0] = action
        self.state[step, :, 1] = particles.mass.value_in(self.units_m)
        self.state[step, :, 2:5] = particles.position.value_in(self.units_l)
        self.state[step, :, 5:] = particles.velocity.value_in(self.units_l/self.units_t)

        particles_name_code = []
        for i in range(len(particles)):
            if particles[i].name == 'star':
                particles_name_code.append(0)
            else:
                particles_name_code.append(1)

        self.cons[step, 0] = action
        self.cons[step, 1] = R
        self.cons[step, 2] = E
        self.comp_time[step-1] = T

        np.save(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
             '_state'+ self.settings['Integration']['suffix'], self.state)
        np.save(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
             '_cons'+ self.settings['Integration']['suffix'], self.cons)
        np.save(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
             '_tcomp' + self.settings['Integration']['suffix'], self.comp_time)
        
    def loadstate(self):
        """
        loadstate: load from file
        OUTPUTS:
            state: positions, masses and velocities of the particles
            cons: energy error, angular momentum
            tcomp: computation time
        """
        state = np.load(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
                         '_state'+ self.settings['Integration']['suffix']+'.npy')
        cons = np.load(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
                        '_cons'+ self.settings['Integration']['suffix']+'.npy')
        tcomp = np.load(self.settings['Integration']['savefile'] +  self.settings['Integration']['subfolder'] +\
                         '_tcomp'+ self.settings['Integration']['suffix'] +'.npy')
        return state, cons, tcomp
    
    def plot_orbit(self):
        """
        plot_orbit: plot orbits of the bodies
        """
        state, cons = self.loadstate()

        n_bodies = np.shape(state)[1]

        for i in range(n_bodies):
            plt.plot(state[:, i, 2], state[:, i, 3], marker= 'o', label = self.names[i])
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()


    def calculate_angular_m(self, particles):
        """
        calculate_angular_m: return angular momentum (units m, s, kg)
        INPUTS: 
            particles: particle set with all bodies in the system
        OUTPUTS:
            L: angular momentum vector
        """
        L = 0
        for i in range(len(particles)):
            r = particles[i].position.value_in(self.units_l)
            v = particles[i].velocity.value_in(self.units_l/self.units_t)
            m = particles[i].mass.value_in(self.units_m)
            L += np.cross(r, m*v)
        return L
