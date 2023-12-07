import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup
import json

from amuse.units import units, constants, nbody_system
from amuse.ic.kingmodel import new_king_model
from amuse.community.hermite.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.ext.solarsystem import new_solar_system
from amuse.ext.orbital_elements import  generate_binaries
from amuse.lab import new_powerlaw_mass_distribution, Particles
from amuse.ic.kingmodel import new_king_model

from pyDOE import lhs

from plots import plot_state, plot_trajectory


"""
https://www.gymlibrary.dev/content/environment_creation/


Questions:

how to change the timestep parameter. now it doesn't make a difference. although even with the same values
we get different energy errors (round-off error?)
"""
def orbital_period(G, a, Mtot):
    return 2*np.pi*(a**3/(constants.G*Mtot)).sqrt()

class IntegrateEnv_Hermite(gym.Env):
    def __init__(self, render_mode = None, bodies = None, subfolder = '', suffix = ''):
        self.settings = load_json("./settings_hermite.json")
        if bodies == None:
            self.n_bodies = self.settings['Integration']['n_bodies']
        self.size = 4*self.n_bodies #TODO: fix observation space
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.size), \
                                                high=np.array([np.inf]*self.size), \
                                                dtype=np.float64)
        
        self.actions = self.settings['Integration']['t_step_param']

        self.save_state_to_file = self.settings['Integration']['savestate']
        
        self.action_space = gym.spaces.Discrete(len(self.actions)) 
        self.W = self.settings['Training']['weights']
        self.seed_initial = self.settings['Integration']['seed']

        self.suffix = suffix # added info for saving files
        self.subfolder = subfolder

    def _add_bodies(self, bodies):
        ranges = self.settings['Integration']['ranges']
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


    def _initial_conditions(self, seed):
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
            bodies = Particles(self.n_bodies)
            bodies.mass = (1.0, 1.0, 1.0) | units.MSun
            bodies[0].position = (0.0, 0.0, 0.0) | units.au
            bodies[1].position = (10.0, 0.0, 0.0) | units.au
            bodies[2].position = (0.0, 10.0, 0.0) | units.au
            bodies[0].velocity = (0.0, 10.0, 0.0) | units.kms
            bodies[1].velocity = (-10.0, 0.0, 0.0) | units.kms
            self.converter = nbody_system.nbody_to_si(bodies.mass.sum(), 1 | units.au)
        
        if self.settings['Integration']['units'] == 'si':
            # bodies.scale_to_standard(convert_nbody=self.converter)
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
    
    def _get_state(self, particles): 
        """
        Excluding the sun
        """
        # state is the position magnitude of each particle? 
        state = np.zeros((self.n_bodies)*3) # m, all r, all v
        # for i in range(self.n_bodies):
        particles_p_nbody = self.converter.to_generic(particles.position).value_in(nbody_system.length)
        particles_v_nbody = self.converter.to_generic(particles.velocity).value_in(nbody_system.length/nbody_system.time)
        particles_m_nbody = self.converter.to_generic(particles.mass).value_in(nbody_system.mass)

        state[0:self.n_bodies] = particles_m_nbody
        state[self.n_bodies: 2*self.n_bodies]  = np.linalg.norm(particles_p_nbody, axis = 1)
        state[2*self.n_bodies: 3*self.n_bodies] = np.linalg.norm(particles_v_nbody, axis = 1)
        # state[0:self.n_bodies] = np.linalg.norm(particles.position.value_in(self.units_l), axis = 1)
        # state[self.n_bodies:2*self.n_bodies] = np.linalg.norm(particles.velocity.value_in(self.units_l/self.units_t), axis = 1)
        return state
    

    def _initialize_integrator(self, tstep_param):
        if self.settings['Integration']['units'] == 'si':
            g = Hermite(self.converter)
        else:
            g = Hermite()
        # g.parameters.set_defaults()
        g.parameters.dt_param = tstep_param 
        return g 
    
    def reset(self, options = None, seed = None):
        if seed != None: 
            self.seed_initial = seed # otherwise use from the settings
        # super().reset(seed=seed) # We need to seed self.np_random

        self.particles = self._initial_conditions(seed)

        # Initialize basic integrator and add particles
        self.gravity = self._initialize_integrator(self.actions[0])
        
        self.gravity.particles.add_particles(self.particles)
        self.channel = self.gravity.particles.new_channel_to(self.particles)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0, self.L_0 = self._get_info_initial()

        # state = self.gravity.particles
        state_RL = self._get_state(self.particles) # |r_i|, |v_i|
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration
        self.t0_comp = time.time()

        # Initialize variables to save simulation
        if self.save_state_to_file == True:
            steps = self.settings['Integration']['max_steps']
            self.state = np.zeros((steps, self.n_bodies, 8)) # action, mass, rx3, vx3, 
            self.cons = np.zeros((steps, 5)) # action, E, Lx3, 
            self.comp_time = np.zeros(self.settings['Integration']['max_steps']) # computation time
            self._savestate(0, 0, self.particles, 0.0, 0.0) # save initial state

        self.info_prev = [0.0, 0.0]
        return state_RL, self.info_prev
    
    def _calculate_reward(self, info, info_prev, T, action, W):
        Delta_E, Delta_O = info
        Delta_E_prev, Delta_O_prev = info_prev

        # print("RRR", Delta_E, Delta_E_prev, action, W[0]* np.log10(abs(Delta_E)), 
        #       W[1]*(np.log10(Delta_E)-np.log10(Delta_E_prev)), W[2]*np.log10(action))
        if Delta_E_prev == 0.0:
            return 0
        else:
            return (W[0]* np.log10(abs(Delta_E)) + W[1]*(np.log10(abs(Delta_E))-np.log10(abs(Delta_E_prev)))) /\
                W[2]*np.log10(action)
    
    def step(self, action):

        self.iteration += 1

        check_step = self.settings['Integration']['check_step'] # final time for step integration
        t0_step = time.time()
        # print("Action", self.actions[action])

        # Apply action
        self.gravity.parameters.dt_param = self.actions[action] 

        self.t_cumul += check_step

        # Integrate
        # self.gravity.evolve_model(check_step | self.units_time)
        t = (self.t_cumul) | self.units_time
        self.gravity.evolve_model(t)
        self.channel.copy()

        # for i, t in enumerate(times[1:]):
            # self.gravity.evolve_model(t)
            # self.channel.copy()
        if self.save_state_to_file == True:
            info_error = self._get_info()
            self._savestate(action, self.iteration, self.gravity.particles, info_error[0], info_error[1]) # save initial state
        
        # Get information for the reward
        T = time.time() - t0_step
        info = self._get_info()
        state = self._get_state(self.gravity.particles)
        reward = self._calculate_reward(info_error, self.info_prev, T, self.actions[action], self.W) # Use computation time for this step, including changing integrator
        self.reward = reward
        self.info_prev = info_error

        self.comp_time[self.iteration-1] = T
        if self.save_state_to_file == True:
            np.save(self.settings['Integration']['savefile'] + self.subfolder + '_tcomp' + self.suffix, self.comp_time)


        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info)

        if self.settings['Integration']['plot'] == True:
            plot_state(self.gravity.particles)

        # finish experiment if max number of iterations is reached
        if (abs(info_error[0]) > 1e-4):
            terminated = True
            # self.gravity.stop()
            # plot_trajectory(self.settings)
        else:
            terminated = False
            
        info = dict()
        info['TimeLimit.truncated'] = False
        info['Energy_error'] = info_error[0]

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

        np.save(self.settings['Integration']['savefile'] + self.subfolder + '_state'+ self.suffix, self.state)
        np.save(self.settings['Integration']['savefile'] + self.subfolder + '_cons'+ self.suffix, self.cons)
    
    def loadstate(self):
        state = np.load(self.settings['Integration']['savefile'] + self.subfolder + '_state'+ self.suffix+'.npy')
        cons = np.load(self.settings['Integration']['savefile'] + self.subfolder + '_cons'+ self.suffix+'.npy')
        tcomp = np.load(self.settings['Integration']['savefile'] +  self.subfolder + '_tcomp'+ self.suffix+'.npy')
        return state, cons, tcomp

    def plot_orbit(self):
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
    load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data


# Test environment

# def run_many_hermite_cases(values):
#     cases = len(values)
#     a = IntegrateEnv()
#     for j in range(cases):
#         print("Case %i"%j)
#         print(values[j][0])
#         print(a.actions[values[j][0]])
#         a.suffix = ('_case%i'%j)
#         value = values[j]
#         terminated = False
#         i = 0
#         a.reset()
#         while terminated == False:
#             x, y, terminated, zz = a.step(value[i%len(value)])
#             i += 1
#         a.close()

# def evaluate_many_symple_cases(values):
#     cases = len(values)
#     a = IntegrateEnv()

#     # Load run information
#     state = list()
#     cons = list()
#     for i in range(cases):
#         a.suffix = ('_case%i'%i)
#         state.append(a.loadstate()[0])
#         cons.append(a.loadstate()[1])


#     bodies = len(state[0][0,:,0])
#     steps = len(cons[0][:,0])
    
#     # Calculate the energy errors
#     # baseline = cons[4] # Take highest order as a baseline
#     E_E = np.zeros((steps, cases))
#     E_M = np.zeros((steps, cases))
#     for i in range(cases):
#         # E_E[:, i] = abs((cons[i][:, 1] - cons[i][0, 1])/ cons[i][0, 1]) # absolute relative energy error
#         # E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:])/ cons[i][0, 2:], axis = 1) # relative angular momentum error
#         E_E[:, i] = abs(cons[i][:, 1]) # absolute relative energy error
#         E_M[:, i] = np.linalg.norm((cons[i][:, 2:] - cons[i][0, 2:]), axis = 1) # relative angular momentum error


#     # plot
#     colors = ['red', 'green', 'blue', 'orange', 'grey', 'yellow', 'black']
#     labels = ['Order 1', 'Order 2', 'Order 3', 'Order 5', 'Order 10', 'Mixed order']

#     fig, ax = plt.subplots(2, 1, layout = 'constrained')
#     x_axis = np.arange(0, steps, 1)
#     for i in range(cases):
#         ax[0].plot(x_axis, E_E[:, i], color = colors[i], label = 't_step = %E'%(a.actions[values[i][0]]))
#         ax[1].plot(x_axis, E_M[:, i], color = colors[i], label = 't_step = %E'%(a.actions[values[i][0]]))

#     ax[0].legend()
#     ax[0].set_yscale('log')
#     ax[0].set_ylabel('Energy error')
#     ax[1].set_ylabel('Angular momentum error')
#     ax[1].set_xlabel('Step')

#     plt.show()
    


# # settings = load_json("./settings.json")
# # Run all possibilities
# a = IntegrateEnv()
# steps = 100 # large value just in case
# values = list()
# for i in range(len(a.actions)):
#     values.append([i]*steps)

# # values = values[0:4]
# values.append(np.random.randint(0, len(a.actions), size = steps))
# run_many_symple_cases(values)
# evaluate_many_symple_cases(values)


# # a = IntegrateEnv()
# # terminated = False
# # i = 0
# # a.reset()
# # value = values[0]
# # print(value)

# # while terminated == False:
# #     print("dsklfjsdk", value[i%len(value)])
# #     x, y, terminated, zz = a.step(value[i%len(value)])
# #     i += 1
# # a.close()

# # a.plot_orbit()

