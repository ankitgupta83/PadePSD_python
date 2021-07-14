import random
import numpy as np
from scipy.special import comb
import sympy as sym
import math


def ConstructPadeApproximation(pade_derivatives_infty_mean, pade_derivatives_s0_mean, s0,
                               order):
    p1 = order[0]
    p2 = order[1]
    p = (p1 + p2) // 2
    if p1 > 0:
        pade_derivatives_s0 = pade_derivatives_s0_mean[:p1]
    else:
        pade_derivatives_s0 = []
    if p2 > 0:
        pade_derivatives_infty = -pade_derivatives_infty_mean[:p2]
    else:
        pade_derivatives_infty = []

    for i in np.arange(p2):
        for j in np.arange(i):
            pade_derivatives_infty[i] += -comb(i, j) * ((-s0) ** (i - j)) * pade_derivatives_infty_mean[j]
    if p1 == 0:
        pade_derivatives_infty = np.pad(pade_derivatives_infty, (1, 0), 'constant', constant_values=0)
        # the constant value does not affect the result
    power_series_coeffs = np.append(np.flip(pade_derivatives_infty), pade_derivatives_s0)
    a = lambda i: power_series_coeffs[i + p2]
    x = sym.Symbol('x')
    Base = [[a(p1 - i) for i in range(1 + j, p + j + 2)] for j in range(0, p)]
    # Q matrix for the denominator
    Q = Base.copy()
    Q.insert(0, [(x - s0) ** j for j in range(0, p + 1)])
    Q = sym.Matrix(Q)
    power_series_coeffs[p2] = power_series_coeffs[p2] / 2
    P = Base.copy()
    if p1 >= p2:
        S = []
        for j1 in range(0, p):
            S.append(0)
            for j0 in range(0, p - j1):
                S[j1] += a(j0) * ((x - s0) ** (j0 + j1))
        S.append(- ((x - s0) ** p) * a(0))
        P.insert(0, S)
    else:
        T = []
        for j1 in range(0, p + 1):
            T.append(0)
            for j0 in range(0, j1 + 1):
                T[j1] += -a(-j0) * ((x - s0) ** (j1 - j0))
        P.insert(0, T)
    P = sym.Matrix(P)
    nom, denom = sym.fraction(sym.simplify(P.det()))

    if sym.polys.polytools.degree(denom, x) > 0:
        numerator_poly = sym.polys.polytools.quo(nom, denom)
    else:
        numerator_poly = nom / denom

    nom, denom = sym.fraction(sym.simplify(Q.det()))

    if sym.polys.polytools.degree(denom, x) > 0:
        denominator_poly = sym.polys.polytools.quo(nom, denom)
    else:
        denominator_poly = nom / denom

    simplified_expression = sym.cancel(sym.simplify(numerator_poly / denominator_poly + a(0)))
    G_p = sym.collect(simplified_expression, x)
    nom, denom = sym.fraction(G_p)
    nom = sym.expand(nom)
    denom = sym.expand(denom)
    deg_den = sym.degree(denom, gen=x)
    coeff_gp_nom = [nom.coeff(x, deg_den - 1 - idx) / denom.coeff(x, deg_den) for idx in range(deg_den)]
    coeff_gp_denom = [denom.coeff(x, deg_den - idx) / denom.coeff(x, deg_den) for idx in range(deg_den + 1)]
    # this is the function that characterises the PSD
    G_p = sym.collect(sym.simplify(sym.Poly(coeff_gp_nom, x) / sym.Poly(coeff_gp_denom, x)), x)
    omega = sym.Symbol('omega', real=True)
    psd_function = sym.collect(sym.simplify(2 * sym.re(G_p.subs(x, omega * sym.I))), omega)
    return psd_function, G_p


def ConstructPadeApproximationPSD(psd_function, omega_limit=np.pi):
    omega_samples = np.linspace(0, omega_limit, 1000)
    psd = omega_samples * 0
    omega = sym.Symbol('omega', real=True)
    for i in np.arange(np.shape(omega_samples)[0]):
        psd[i] = psd_function.subs(omega, omega_samples[i])
    return omega_samples, psd


class PadePSD(object):
    def __init__(self, network, stop_time, cut_off_time, s0=1, test_s_values=None):
        self.network = network
        self.stop_time = stop_time
        self.cut_off_time = cut_off_time
        self.unique_states = None
        self.approximation_order = None
        self.s0 = s0
        self.output_species = self.network.species_labels.index(network.output_species_labels[0])
        self.test_s_values = test_s_values

    def gillespie_ssa_next_reaction_aux(self, state_aux):
        state = state_aux[:self.network.num_species]
        prop = self.network.propensity_vector(state)
        sum_prop = np.sum(prop) + self.s0 + np.sum(self.test_s_values)
        if sum_prop == 0:
            delta_t = math.inf
            next_reaction = -1
        else:

            prop = np.concatenate((prop, np.insert(self.test_s_values, 0, self.s0)), axis=None)
            prop = np.cumsum(np.divide(prop, sum_prop))
            delta_t = -math.log(np.random.uniform(0, 1)) / sum_prop
            next_reaction = sum(prop < np.random.uniform(0, 1))
        return delta_t, next_reaction

    def update_state_aux(self, next_reaction, state_aux):

        state = state_aux[:self.network.num_species]
        hist = state_aux[self.network.num_species:self.network.num_species + self.approximation_order[0]]
        aux_hist = state_aux[self.network.num_species + self.approximation_order[0]:]
        if next_reaction != -1:
            if next_reaction < self.network.num_reactions:
                state = state + self.network.stoichiometry_matrix[next_reaction, :]
            elif next_reaction == self.network.num_reactions:
                hist = np.insert(hist, 0, state[self.output_species])
                hist = np.delete(hist, -1)
            else:
                aux_hist[next_reaction - self.network.num_reactions - 1] = state[self.output_species]
        return np.concatenate((state, hist, aux_hist), axis=None)

    def run_gillespie_ssa_aux(self, initial_state, stop_time):
        """
        Runs Gillespie's SSA without storing any values until stop_time; start time is 0 and
        initial_state is specified
        """
        t = 0
        state_curr = initial_state
        while 1:
            delta_t, next_reaction = self.gillespie_ssa_next_reaction_aux(state_curr)
            t = t + delta_t
            if t > stop_time:
                return state_curr
            else:
                state_curr = self.update_state_aux(next_reaction, state_curr)

    def generate_sampled_ssa_trajectory_aux(self, cut_off_time, stop_time, num_time_samples, seed=None):

        """
        Create a uniformly sampled SSA Trajectory.
        """
        if seed is None:
            random.seed(seed)
        sampling_times = np.linspace(cut_off_time, stop_time, num_time_samples)
        state_curr = np.append(self.network.initial_state, np.zeros(self.approximation_order[0]
                                                                    + np.shape(self.test_s_values)[0]))
        state_curr = self.run_gillespie_ssa_aux(state_curr, cut_off_time)
        states_array = np.array([state_curr])
        for j in range(sampling_times.size - 1):
            state_curr = self.run_gillespie_ssa_aux(state_curr, sampling_times[j + 1] - sampling_times[j])
            states_array = np.append(states_array, [state_curr], axis=0)
        return sampling_times, states_array

    def generate_sampled_ssa_trajectories_aux(self, cut_off_time, stop_time, num_time_samples, num_trajectories=1,
                                              seed=None):

        """
        Create several uniformly sampled SSA Trajectories.
        """
        states_trajectories = np.zeros(
            [num_trajectories, num_time_samples, (self.network.num_species + self.approximation_order[0]
                                                  + np.shape(self.test_s_values)[0])])
        for i in range(num_trajectories):
            times, states_trajectories[i, :, :] = \
                self.generate_sampled_ssa_trajectory_aux(cut_off_time, stop_time, num_time_samples, seed)
        return times, states_trajectories

    def RecursiveAdd(self, state, n):
        if n == 0:
            self.unique_states.add(np.array2string(state))
        else:
            self.RecursiveAdd(state, n - 1)
            for k in np.arange(self.network.num_reactions):
                self.RecursiveAdd(self.network.update_state(k, state), n - 1)

    def RecursiveGenerator(self, state, n):
        output_species_index = self.network.species_labels.index(self.network.output_species_labels[0])
        key = np.array2string(state)
        if self.unique_states[key][n] is not None:
            return self.unique_states[key][n]
        elif n == 0:
            self.unique_states[key][n] = state[output_species_index]
            return self.unique_states[key][n]
        else:
            prop = self.network.propensity_vector(state)
            value = - np.sum(prop) * self.RecursiveGenerator(state, n - 1)
            for k in np.arange(self.network.num_reactions):
                if prop[k] > 0:
                    value += prop[k] * self.RecursiveGenerator(self.network.update_state(k, state), n - 1)
            self.unique_states[key][n] = value
            return value

    def ComputeProductValues(self, state, n1, n2):
        prop = self.network.propensity_vector(state)
        f1 = self.RecursiveGenerator(state, n1)
        f2 = self.RecursiveGenerator(state, n2)
        value = 0
        for k in np.arange(self.network.num_reactions):
            if prop[k] > 0:
                value += prop[k] * (self.RecursiveGenerator(self.network.update_state(k, state), n1) - f1) \
                         * (self.RecursiveGenerator(self.network.update_state(k, state), n2) - f2)
        return value

    def EstimatePadeDerivatives(self, order, num_time_samples, num_trajectories=1):
        self.approximation_order = order
        output_species_index = self.network.species_labels.index(self.network.output_species_labels[0])
        print('Generating trajectories of the augmented CTMC')
        sampling_times, states_array = self.generate_sampled_ssa_trajectories_aux(self.cut_off_time, self.stop_time,
                                                                                  num_time_samples,
                                                                                  num_trajectories,
                                                                                  seed=None)
        print('Finished generating trajectories')
        self.unique_states = set()
        for trj in np.arange(num_trajectories):
            for i in np.arange(num_time_samples):
                state_aux = states_array[trj, i, :]
                state = state_aux[:self.network.num_species]
                if self.approximation_order[1] > 0:
                    self.RecursiveAdd(state, self.approximation_order[1] - 1)
        self.unique_states = {key: np.full(self.approximation_order[1], None) for key in self.unique_states}
        print('Finished creating state dictionary')
        ListSize = len(self.unique_states.keys())
        print("No. of unique states", ListSize)
        if self.approximation_order[1] > 0:
            PadeDerivativesInfinity = np.zeros([num_trajectories, self.approximation_order[1]])
        else:
            PadeDerivativesInfinity = np.zeros([num_trajectories, 1])
        PadeDerivatives_s0_Value = np.zeros([num_trajectories, self.approximation_order[0]])
        G_at_test_s_values = np.zeros([num_trajectories, np.shape(self.test_s_values)[0]])
        mean_output = np.zeros([num_trajectories])
        for trj in np.arange(num_trajectories):
            # print("Starting trajectory: ", trj + 1)
            time_counter = 0
            for i in np.arange(num_time_samples):
                # if sampling_times[i] > self.cut_off_time + time_counter * 10:
                #     print("Reached time", self.cut_off_time + time_counter * 10)
                #     time_counter += 1
                state_aux = states_array[trj, i, :]
                state = state_aux[:self.network.num_species]
                hist = state_aux[self.network.num_species:self.network.num_species + self.approximation_order[0]]
                aux_hist = state_aux[self.network.num_species + self.approximation_order[0]:]
                PadeDerivativesInfinity[trj, 0] += (state[output_species_index] ** 2)
                for l in np.arange(np.shape(self.test_s_values)[0]):
                    G_at_test_s_values[trj, l] += (state[output_species_index] * aux_hist[l])
                mean_output[trj] += state[output_species_index]
                for n in np.arange(1, self.approximation_order[1]):
                    inc = 0
                    m = n // 2
                    if n % 2 == 0:
                        for k in np.arange(m):
                            if k > 0:
                                inc += comb(n, k) * self.RecursiveGenerator(state, k) * \
                                       self.RecursiveGenerator(state, n - k)
                            inc += comb(n - 1, k) * self.ComputeProductValues(state, k, n - 1 - k)
                        inc += 0.5 * comb(n, m) * (self.RecursiveGenerator(state, m) ** 2)
                    else:
                        for k in np.arange(1, m + 1):
                            inc += comb(n, k) * self.RecursiveGenerator(state, k) * self.RecursiveGenerator(state,
                                                                                                            n - k)
                            inc += comb(n - 1, k - 1) * self.ComputeProductValues(state, k - 1, n - k)
                        inc += 0.5 * comb(n - 1, m) * self.ComputeProductValues(state, m, m)
                    PadeDerivativesInfinity[trj, n] += -   inc
                for n in np.arange(self.approximation_order[0]):
                    PadeDerivatives_s0_Value[trj, n] += (state[output_species_index] * hist[n])
        mean_output = mean_output / num_time_samples
        PadeDerivativesInfinity = PadeDerivativesInfinity / num_time_samples
        PadeDerivatives_s0_Value = PadeDerivatives_s0_Value / num_time_samples
        G_at_test_s_values = G_at_test_s_values / num_time_samples
        for l in np.arange(np.shape(self.test_s_values)[0]):
            G_at_test_s_values[:, l] -= mean_output ** 2
            G_at_test_s_values[:, l] = G_at_test_s_values[:, l] / self.test_s_values[l]
        if self.approximation_order[1] > 0:
            PadeDerivativesInfinity[:, 0] -= mean_output ** 2
        for n in np.arange(0, self.approximation_order[0]):
            PadeDerivatives_s0_Value[:, n] -= mean_output ** 2
            PadeDerivatives_s0_Value[:, n] = ((-1) ** n) * PadeDerivatives_s0_Value[:, n] / (
                    self.s0 ** (n + 1))
        pade_dict = {'Infinity': [np.mean(PadeDerivativesInfinity, axis=0), np.std(PadeDerivativesInfinity, axis=0)],
                     's0': [np.mean(PadeDerivatives_s0_Value, axis=0),
                            np.std(PadeDerivatives_s0_Value, axis=0)],
                     'Test_s_vals': [np.mean(G_at_test_s_values, axis=0), np.std(G_at_test_s_values, axis=0)]
                     }
        return pade_dict
