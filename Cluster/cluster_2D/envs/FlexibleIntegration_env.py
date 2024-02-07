"""
FlexibleIntegration_env: environment for the use of different integrators for the 
evolution of the gravitational 3-body problem

Author: Veronica Saz Ulibarrena
Last modified: 6-February-2024

Based on https://www.gymlibrary.dev/content/environment_creation/
"""

import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup
import json
from pyDOE import lhs

from amuse.units import units, constants, nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.community.hermite.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.symple.interface import symple
from amuse.community.huayno.interface import Huayno
from amuse.ext.solarsystem import new_solar_system
from amuse.ext.orbital_elements import  generate_binaries
from amuse.lab import new_powerlaw_mass_distribution, Particles

from plots import plot_state, plot_trajectory


def orbital_period(a, Mtot, G = constants.G):
    """"
    orbital_period: calculate the orbital period of a body around a more massive one
    INPUTS"
        a: semi-major axis
        Mtot: total mass of the system 
        G: universal gravitational constant
    OUTPUTS:
        T: orbital period of the minor body with semi-major axis a
    """
    return 2*np.pi*(a**3/(G*Mtot)).sqrt()

class IntegrateEnv_multiple(gym.Env):
    def __init__(self, render_mode = None, integrator = 'Hermite', subfolder = '', suffix = ''):
        """
        IntegrateEnv_multiple: environment to integrate a system of three bodies allowing 
            different integrators. 

        INPUTS:
            integrator: choice of integrator. If None, Hermite taken by default
            subfolder: choice of subfolder to save the evolution of the system. 
                    The folder is chosen from the settings file..
            suffix: additional name to give to saving files
        """

        # Choose observation space size
        self.settings = load_json("./settings_multiple.json")
        self.n_bodies = self.settings['Integration']['n_bodies']
        self.size = 4*self.n_bodies 
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.size), \
                                                high=np.array([np.inf]*self.size), \
                                                dtype=np.float64)

        # Choose integrator to be used
        self.integrator = integrator

        # Create actions
        # from the items of an array
        if self.settings['Integration'][self.integrator]['action'] == 'array': 
            self.actions = self.settings['Integration']['t_step_param']
        # from a range of paramters
        elif self.settings['Integration'][self.integrator]['action'] == 'range':
            low = self.settings['Integration'][self.integrator]['t_step_param'][0]
            high = self.settings['Integration'][self.integrator]['t_step_param'][-1]
            n_actions = self.settings['Integration'][self.integrator]['number_actions']
            self.actions = np.logspace(np.log10(low), np.log10(high), \
                                       num = n_actions, base = 10,
                                       endpoint = True)
        
        self.action_space = gym.spaces.Discrete(len(self.actions)) 
        
        # Other settings
        self.W = self.settings['Training']['weights']
        self.seed_initial = self.settings['Integration']['seed']
        self.subfolder = subfolder # added subfolder where to save files
        self.suffix = suffix # added info for saving files

    def _add_bodies(self, bodies):
        """"
        _add_bodies: create a randomized distribution of bodies and add to particles set. For the case
        of a planetary system
        INPUTS:
            bodies: particle set with planetary system

        OUTPUTS:
            bodies: particle set with planetary system and added particles
        """
        ranges = self.settings['Integration'][self.integrator]['ranges']
        ranges_np = np.array(list(ranges.values()))

        if self.seed_initial != "None":
            np.random.seed(seed = self.seed_initial)
        K = lhs(len(ranges), samples = self.n_bodies) * (ranges_np[:, 1]- ranges_np[:, 0]) + ranges_np[:, 0] 

        for i in range(self.n_bodies):
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


    def _initial_conditions(self):
        """
        _initial_conditions: choose initial conditions to be used in the problem
        The options are:
            - planetary system with the solar system bodies plus some extra eccentric bodies
            - cluster: distribution of stars
            - triple: system of three stars

        OUTPUTS:
            bodies: particle set with the bodies in the system including their masses, positions 
                and velocities
        """
        if self.settings['Integration']['system'] == 'planetary':
            bodies = new_solar_system()
            bodies = self._add_bodies(bodies)
            self.names = bodies.name
            self.n_bodies = len(bodies)

            self.converter = nbody_system.nbody_to_si(bodies.mass.sum(), 1 | units.au)

        elif self.settings['Integration']['system'] == 'cluster':
            alpha_IMF = -2.35
            masses = new_powerlaw_mass_distribution(self.n_bodies, 0.1 |units.MSun, 
                                        100 | units.MSun, alpha_IMF)
            W0 = 3.0
            r_cluster = 1.0 | units.parsec
            self.converter=nbody_system.nbody_to_si(masses.sum(),r_cluster)
            bodies = new_king_model(self.n_bodies, W0, convert_nbody=self.converter)
            bodies.scale_to_standard(self.converter)
            bodies.mass = masses
        
        elif self.settings['Integration']['system'] == 'triple':
            self.n_bodies = 3

            ranges = self.settings['Integration']['ranges_triple']
            ranges_np = np.array(list(ranges.values()))

            bodies = Particles(self.n_bodies)
            
            bodies.mass = (1.0, 1.0, 1.0) | units.MSun # equal mass
            bodies[0].position = (0.0, 0.0, 0.0) | units.au

            if self.seed_initial != "None":
                np.random.seed(seed = self.seed_initial)
            K = np.random.uniform(low = ranges_np[:, 0], high = ranges_np[:, 1])
            # K = lhs(len(ranges), samples =  1) * (ranges_np[:, 1]- ranges_np[:, 0]) + ranges_np[:, 0] 
            
            bodies[1].position = (K[0], K[1], 0.0) | units.au
            bodies[2].position = (K[2], K[3], 0.0) | units.au
            bodies[0].velocity = (0.0, 10.0, 0.0) | units.kms
            bodies[1].velocity = (-10.0, 0.0, 0.0) | units.kms
            self.converter = nbody_system.nbody_to_si(bodies.mass.sum(), 1 | units.au)
        
        # Choose set of units for the problem
        if self.settings['Integration']['units'] == 'si':
            self.G = constants.G
            self.units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
            self.units_energy = units.m**2 * units.s**(-2)* units.kg
            if self.settings['Integration']['system'] == 'planetary' or \
                self.settings['Integration']['system'] == 'triple':
                self.units_time = units.yr
            elif self.settings['Integration']['system'] == 'cluster':
                self.units_time = 1000*units.yr

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
        print(bodies)
        # Plot initial state
        if self.settings['Integration']['plot'] == True:
            plot_state(bodies)
        return bodies

    def _get_info_initial(self):
        """
        _get_info_initial: get energy error, angular momentum error at current state
        OUTPUTS:
            Total energy, Angular momentum vector
        """
        E_kin = self.gravity.particles.kinetic_energy().value_in(self.units_energy)
        E_pot = self.gravity.particles.potential_energy(G = self.G).value_in(self.units_energy)
        L = self.calculate_angular_m(self.gravity.particles)
        return E_kin + E_pot, L
    
    def _get_info(self):
        """
        _get_info: get energy error, angular momentum error at current state
        OUTPUTS:
            Relative energy error, Relative Angular momentum error vector 
        """
        E_kin = self.gravity.particles.kinetic_energy().value_in(self.units_energy)
        E_pot = self.gravity.particles.potential_energy(G = self.G).value_in(self.units_energy)
        L = self.calculate_angular_m(self.gravity.particles)
        Delta_E = (E_kin + E_pot - self.E_0) / self.E_0
        Delta_L = (L - self.L_0) / self.L_0
        return Delta_E, Delta_L
    
    def _get_state(self, particles, E): 
        """
        _get_state: create the state vector
        Options:
            - norm: norm of the positions and velocities of each body and the masses
            - cart: 2D cartesian coordinates of the position and angular momentum plus the energy error
            - dis: distance between particles in position and momentum space plus the energy error

        OUTPUTS: 
            state: state array to be given to the reinforcement learning algorithm
        """
        particles_p_nbody = self.converter.to_generic(particles.position).value_in(nbody_system.length)
        particles_v_nbody = self.converter.to_generic(particles.velocity).value_in(nbody_system.length/nbody_system.time)
        particles_m_nbody = self.converter.to_generic(particles.mass).value_in(nbody_system.mass)

        if self.settings['Integration']['state'] == 'norm':
            state = np.zeros((self.n_bodies)*3) # m, norm r, norm v

            state[0:self.n_bodies] = particles_m_nbody
            state[self.n_bodies: 2*self.n_bodies]  = np.linalg.norm(particles_p_nbody, axis = 1)
            state[2*self.n_bodies: 3*self.n_bodies] = np.linalg.norm(particles_v_nbody, axis = 1)
       
        elif self.settings['Integration']['state'] == 'cart':
            state = np.zeros((self.n_bodies)*4+1) # all r, all v
            for i in range(self.n_bodies):
                state[2*i:2*i+2] = particles_p_nbody[i, 0:2]/10 # convert to 2D
                state[2*self.n_bodies + 2*i: 2*self.n_bodies + 2*i+2] = particles_v_nbody[i, 0:2]
                state[-1] = -np.log10(abs(E))
        
        elif self.settings['Integration']['state'] == 'dist':
            state = np.zeros((self.n_bodies)*2) # dist r, dist v

            counter = 0
            for i in range(self.n_bodies):
                for j in range(i+1, self.n_bodies):
                    state[counter]  = np.linalg.norm(particles_p_nbody[i,:]-particles_p_nbody[j,:], axis = 0) /10
                    state[self.n_bodies+counter ] = np.linalg.norm(particles_v_nbody[i,:]-particles_v_nbody[j,:], axis = 0)
                    counter += 1

            state[-1] = -np.log10(abs(E))

        return state
    

    def _initialize_integrator(self, action):
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
        if self.integrator == 'Hermite': 
            if self.settings['Integration']['units'] == 'si':
                g = Hermite(self.converter)
            else:
                g = Hermite()
            # Collision detection and softening
            g.stopping_conditions.timeout_detection.enable()
            g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2
        elif self.integrator == 'Ph4': 
            if self.settings['Integration']['units'] == 'si':
                g = ph4(self.converter)
            else:
                g = ph4()
            g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif self.integrator == 'Huayno': 
            if self.settings['Integration']['units'] == 'si':
                g = Huayno(self.converter)
            else:
                g = Huayno()
            g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif self.integrator == 'Symple': 
            if self.settings['Integration']['units'] == 'si':
                g = symple(self.converter, redirection ='none')
            else:
                g = symple(redirection ='none')
            g.initialize_code()

        g = self.apply_action(g, action)
            
        return g 
    
    def apply_action(self, g, action):
        """
        apply_action: add action as an integrator parameter
        INPUTS:
            g: integrator
            action: action to be applied
        OUTPUTS:
            g: integrator
        """
        if self.integrator == 'Hermite':
            g.parameters.dt_param = action
        elif self.integrator == 'Ph4':
            g.parameters.timestep_parameter = action
        elif self.integrator == 'Huayno':
            g.parameters.timestep_parameter = action
        elif self.integrator == 'Symple':
            g.parameters.timestep = action | self.units_time
        return g

    def reset(self, options = None, seed = None, steps = None, typereward = None, save_state = None):
        """
        reset: reset the simulation 
        INPUTS:
            options: 
            seed: choose the random seed
            steps: simulation steps to be taken
            typereward: type or reward to be applied to the problem
            save_state: save state (True or False)
        OUTPUTS:
            state_RL: state vector to be passed to the RL
            info_prev: information vector of the previous time step (zero vector)
        """
        if seed != None: 
            self.seed_initial = seed # otherwise use from the settings

        if typereward == None:
            self.typereward = self.settings['Training']['reward_f']
        else:
            self.typereward = typereward # choice of reward function

        self.particles = self._initial_conditions(seed)

        # Initialize basic integrator and add particles
        self.gravity = self._initialize_integrator(self.actions[0]) # start with most restrictive action
        self.gravity.particles.add_particles(self.particles)
        self.channel = self.gravity.particles.new_channel_to(self.particles)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0, self.L_0 = self._get_info_initial()

        # Create state vector
        state_RL = self._get_state(self.particles, self.E_0) # |r_i|, |v_i|
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration
        self.t0_comp = time.time()

        # Initialize variables to save simulation
        if save_state == None:
            self.save_state_to_file = self.settings['Integration']['savestate']
        else:
            self.save_state_to_file = save_state

        # Initialize variables to save simulation information
        if self.save_state_to_file == True:
            if steps == None:
                steps = self.settings['Integration']['max_steps']
            self.state = np.zeros((steps, self.n_bodies, 8)) # action, mass, rx3, vx3, 
            self.cons = np.zeros((steps, 5)) # action, E, Lx3, 
            self.comp_time = np.zeros(self.settings['Integration']['max_steps']) # computation time
            self._savestate(0, 0, self.particles, 0.0, 0.0) # save initial state

        self.info_prev = [0.0, 0.0]
        return state_RL, self.info_prev
    
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
        Delta_E, Delta_O = info
        Delta_E_prev, Delta_O_prev = info_prev

        if Delta_E_prev == 0.0: # for the initial step
            return 0
        else:
            if self.typereward == 1:
                a = -(W[0]* np.log10(abs(Delta_E)) + \
                         W[1]*(np.log10(abs(Delta_E))-np.log10(abs(Delta_E_prev)))) *\
                        (W[2]*1/abs(np.log10(action)))
                return a
            
            elif self.typereward == 2:
                a = -(W[0]* abs(np.log10(abs(Delta_E)/1e-8))/\
                         abs(np.log10(abs(Delta_E)))**2 +\
                         W[1]*(np.log10(abs(Delta_E))-np.log10(abs(Delta_E_prev))))+\
                         W[2]*1/abs(np.log10(action))
                return a

            elif self.typereward == 3:
                a = -(W[0]* abs(np.log10(abs(Delta_E)/1e-8))/\
                         abs(np.log10(abs(Delta_E)))**2 +\
                         W[1]*(np.log10(abs(Delta_E))-np.log10(abs(Delta_E_prev))))*\
                         W[2]*1/abs(np.log10(action))
                return a
                
            elif self.typereward == 4:
                a = -W[0]*np.log10(abs(Delta_E)) + \
                    W[2]/abs(np.log10(action))
                return a
    
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

        check_step = self.settings['Integration'][self.integrator]['check_step'] # final time for step integration
        t0_step = time.time()

        # Apply action
        self.gravity = self.apply_action(self.gravity, self.actions[action])

        self.t_cumul += check_step # add the previous simulation time

        # Integrate
        t = (self.t_cumul) | self.units_time
        self.gravity.evolve_model(t)
        self.channel.copy()

        info_error = self._get_info()
        if self.save_state_to_file == True:
            self._savestate(action, self.iteration, self.gravity.particles, info_error[0], info_error[1]) # save initial state
        
        # Get information for the reward
        T = time.time() - t0_step

        state = self._get_state(self.gravity.particles, info_error[0])
        reward = self._calculate_reward(info_error, self.info_prev, T, self.actions[action], self.W) # Use computation time for this step, including changing integrator
        self.reward = reward
        self.info_prev = info_error

        self.comp_time[self.iteration-1] = T
        if self.save_state_to_file == True:
            np.save(self.settings['Integration']['savefile'] + self.subfolder + '_tcomp' + self.suffix, self.comp_time)

        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info_error, reward, action)

        # Plot trajectory
        if self.settings['Integration']['plot'] == True:
            plot_state(self.gravity.particles)

        # finish experiment if max number of iterations is reached
        if (abs(info_error[0]) > 1e-4):
            terminated = True
        else:
            terminated = False
            
        info = dict()
        info['TimeLimit.truncated'] = False
        info['Energy_error'] = info_error[0]

        return state, reward, terminated, info
    
    def close(self): 
        """
        close: finish simulation epoch
        """
        self.gravity.stop()

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
    
    def calculate_energy(self): # not needed, verified with amuse
        """
        calculate_energy: calculate energy at a certain step
        OUTPUTS:
            ke: kinetic energy
            pe: potential energy
        """
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

    def _display_info(self, info, reward, action):
        """
        _display_info: display information at every step
        INPUTS:
            info: energy error and angular momentum vector
            reward: value of the reward for the given step
            action: action taken at this step
        """
        print("Iteration: %i/%i, E_E = %0.3E, Action: %i, Reward: %.4E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0],\
                                 action, \
                                 reward))
        
    def _savestate(self, action, step, bodies, E, L):
        """
        _savestate: save state of the system to file
        INPUTS:
            action: action taken
            step: simulation step
            bodies: particles set
            E: energy error
            L: angular momentum
        """
        self.state[step, :, 0] = action
        self.state[step, :, 1] = bodies.mass.value_in(self.units_m)
        self.state[step, :, 2:5] = bodies.position.value_in(self.units_l)
        self.state[step, :, 5:] = bodies.velocity.value_in(self.units_l/self.units_t)
        self.cons[step, 0] = action
        self.cons[step, 1] = E
        self.cons[step, 2:] = L

        np.save(self.settings['Integration']['savefile'] + self.subfolder + '_state'+ self.suffix, self.state)
        np.save(self.settings['Integration']['savefile'] + self.subfolder + '_cons'+ self.suffix, self.cons)
    
    def loadstate(self):
        """
        loadstate: load from file
        OUTPUTS:
            state: positions, masses and velocities of the particles
            cons: energy error, angular momentum
            tcomp: computation time
        """
        state = np.load(self.settings['Integration']['savefile'] + self.subfolder + '_state'+ self.suffix+'.npy')
        cons = np.load(self.settings['Integration']['savefile'] + self.subfolder + '_cons'+ self.suffix+'.npy')
        tcomp = np.load(self.settings['Integration']['savefile'] +  self.subfolder + '_tcomp'+ self.suffix+'.npy')
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



def load_json(filepath):
    """
    load_json: load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data
