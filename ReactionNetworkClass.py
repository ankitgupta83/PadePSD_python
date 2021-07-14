import numpy as np
import math
import random


class ReactionNetworkDefinition(object):
    def __init__(self, num_species, num_reactions, reactant_matrix, product_matrix,
                 parameter_dict, reaction_dict, species_labels, output_species_labels, time_unit):
        # public attributes:
        self.num_species = num_species
        self.num_reactions = num_reactions
        # reactant matrices rows represent number of molecules of each species consumed in that reaction.
        self.reactant_matrix = reactant_matrix
        # product matrices rows represent number of molecules of species produced in that reaction.
        self.product_matrix = product_matrix
        self.stoichiometry_matrix = product_matrix - reactant_matrix
        # contains information about the parameters
        self.parameter_dict = parameter_dict
        self.reaction_dict = reaction_dict
        self.species_labels = species_labels
        self.output_species_labels = output_species_labels
        self.output_species_indices = [self.species_labels.index(i)
                                       for i in self.output_species_labels]

        self.output_function_size = None
        self.time_unit = time_unit

    def mass_action_propensity(self, state, reaction_no, rate_constant_key):
        prop = self.parameter_dict[rate_constant_key]
        for j in range(self.num_species):
            for k in range(self.reactant_matrix[reaction_no][j]):  # check order of indices
                prop *= float(state[j] - k)
            prop = prop / math.factorial(self.reactant_matrix[reaction_no][j])
        return prop

    def hill_propensity_activation(self, state, species_no, parameter_key1, parameter_key2, parameter_key3,
                                   parameter_key4):
        # implements propensity b + a*x^h/(k + x^h) with x=X[species_no]
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        xp = float(state[species_no])
        return b + a * (xp ** h) / (k + (xp ** h))

    def hill_propensity_repression(self, state, species_no, parameter_key1, parameter_key2, parameter_key3,
                                   parameter_key4):
        # implements propensity  a/(k + x^h) with x=X[species_no] + b
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        xp = float(state[species_no])
        return b + a / (k + (xp ** h))

    def linear(self, state, species_no, parameter_key1, parameter_key2):
        # implements propensity max(a + b*x, 0) with x=X[species_no]
        a = self.parameter_dict[parameter_key1]
        b = self.parameter_dict[parameter_key2]
        xp = float(state[species_no])
        return max(a + b * xp, 0)

    def propensity_vector(self, state):
        raise NotImplementedError

    def set_propensity_vector(self):
        def func(state_current):
            prop = np.zeros(self.num_reactions)
            for k in range(self.num_reactions):
                reaction_type = self.reaction_dict[k][0]
                if reaction_type == 'mass action':
                    prop[k] = self.mass_action_propensity(state_current, k, self.reaction_dict[k][1])
                elif reaction_type == 'Hill_activation':
                    prop[k] = self.hill_propensity_activation(state_current, self.reaction_dict[k][1],
                                                              self.reaction_dict[k][2], self.reaction_dict[k][3],
                                                              self.reaction_dict[k][4], self.reaction_dict[k][5])
                elif reaction_type == 'Hill_repression':
                    prop[k] = self.hill_propensity_repression(state_current, self.reaction_dict[k][1],
                                                              self.reaction_dict[k][2], self.reaction_dict[k][3],
                                                              self.reaction_dict[k][4], self.reaction_dict[k][5])
                elif reaction_type == 'linear':
                    prop[k] = self.linear(state_current, self.reaction_dict[k][1], self.reaction_dict[k][2],
                                          self.reaction_dict[k][3])
                else:
                    raise NotImplementedError
            return prop

        self.propensity_vector = func

    def gillespie_ssa_next_reaction(self, state):
        prop = self.propensity_vector(state)
        sum_prop = np.sum(prop)
        if sum_prop == 0:
            delta_t = math.inf
            next_reaction = -1
        else:
            prop = np.cumsum(np.divide(prop, sum_prop))
            delta_t = -math.log(np.random.uniform(0, 1)) / sum_prop
            next_reaction = sum(prop < np.random.uniform(0, 1))
        return delta_t, next_reaction

    def update_state(self, next_reaction, state):
        if next_reaction != -1:
            state = state + self.stoichiometry_matrix[next_reaction, :]
        return state

    def run_gillespie_ssa(self, initial_state, stop_time):
        """
        Runs Gillespie's SSA without storing any values until stop_time; start time is 0 and
        initial_state is specified
        """
        t = 0
        state_curr = initial_state
        while 1:
            delta_t, next_reaction = self.gillespie_ssa_next_reaction(state_curr)
            t = t + delta_t
            if t > stop_time:
                return state_curr
            else:
                state_curr = self.update_state(next_reaction, state_curr)

    def generate_sampled_ssa_trajectory(self, cut_off_time, stop_time, num_time_samples, seed=None):

        """
        Create a uniformly sampled SSA Trajectory.
        """
        if seed is None:
            random.seed(seed)
        sampling_times = np.linspace(cut_off_time, stop_time, num_time_samples)
        state_curr = self.initial_state
        state_curr = self.run_gillespie_ssa(state_curr, cut_off_time)
        states_array = np.array([state_curr])
        for j in range(sampling_times.size - 1):
            state_curr = self.run_gillespie_ssa(state_curr, sampling_times[j + 1] - sampling_times[j])
            states_array = np.append(states_array, [state_curr], axis=0)
        return sampling_times, states_array

    def generate_sampled_ssa_trajectories(self, cut_off_time, stop_time, num_time_samples, num_trajectories=1,
                                          seed=None):

        """
        Create several uniformly sampled SSA Trajectories.
        """
        states_trajectories = np.zeros([num_trajectories, num_time_samples, self.num_species])
        for i in range(num_trajectories):
            times, states_trajectories[i, :, :] = \
                self.generate_sampled_ssa_trajectory(cut_off_time, stop_time, num_time_samples, seed)
        return times, states_trajectories

    def PadePSD(self, omega):
        return None

    def PSD_analytical(self, omega):
        return None

    def PSDArray(self, omega_lim):
        omega = np.linspace(0, omega_lim, 1000)
        psd = omega * 0
        for i in np.arange(np.shape(omega)[0]):
            psd[i] = self.PSD_analytical(omega[i])
        return omega, psd
