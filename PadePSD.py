import random
import numpy as np
from scipy.special import comb
from scipy import linalg
import sympy as sym
import math

def ConstructMultipointPadeApproximant(pade_dict, svals, order, p, known_den_poly=[]):
    # p is the degree of the rational Pade approximant...the degree of denominator is p (leading coefficient is 1)
    # and the degree of numerator is (p-1). known_den_poly are the coefficients (ascending power order) of the
    # denominator polynomial with known coefficients. The leading order coefficient is assumed to be one and it is
    # omitted

    order_len = np.shape(order)[0]
    total_order = np.sum(order)
    known_den_poly = np.append(known_den_poly, 1.0)
    q = len(known_den_poly) - 1  # the degree of the known denominator polynomial
    b = np.zeros(total_order)
    A = np.zeros([total_order, 2 * p])
    order_sum = 0
    for order_counter in range(order_len):
        if order[order_counter] == 0:
            continue
        else:
            s_l = svals[order_counter]
            pade_derivatives = pade_dict[s_l][0][:order[order_counter]]
            if s_l == np.infty:
                for j in range(order[order_counter]):
                    b[order_sum + j] = pade_derivatives[j]
                    A[order_sum + j, p - 1 - j] = 1
                    for i in range(max(p - j, 0), p):
                        A[j + order_sum, i + p] = -pade_derivatives[j + i - p]
            else:
                for j in range(order[order_counter]):
                    for k in range(j + 1):
                        b[order_sum + j] += comb(p, k) * (s_l ** (p - k)) * pade_derivatives[j - k]
                    for i in range(j, p):
                        A[order_sum + j, i] = comb(i, j) * (s_l ** (i - j))
                    for i in range(p, 2 * p):
                        for k in range(min(i - p, j) + 1):
                            A[order_sum + j, i] -= comb(i - p, k) * (s_l ** (i - p - k)) * pade_derivatives[j - k]
            order_sum += order[order_counter]

    # construct the convolution matrix
    C = np.zeros([2 * p, 2 * p - q])
    new_d = np.zeros(2 * p)
    new_d[2 * p - q: 2 * p] = known_den_poly[:-1]
    C[:p, :p] = np.eye(p)
    for j in range(p):
        for i in range(max(j - q, 0), min(p - q - 1, j) + 1):
            C[j + p, i + p] = known_den_poly[j - i]
    b = b - np.matmul(A, new_d)
    coeffs = linalg.lstsq(np.matmul(A, C), b)[0]
    num_coeff = coeffs[:p]
    den_coeffs = np.convolve(np.append(coeffs[p:], 1), known_den_poly)
    x = sym.Symbol('x')
    den_ = sum(co * x ** i for i, co in enumerate(den_coeffs))
    num_ = sum(co * x ** i for i, co in enumerate(num_coeff))
    # this function characterises the PSD
    G_p = num_ / den_
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


def ConstructPSD_from_G_p(G_p):
    x = sym.Symbol('x')
    omega = sym.Symbol('omega', real=True)
    psd_function = sym.collect(sym.simplify(2 * sym.re(G_p.subs(x, omega * sym.I))), omega)
    return psd_function


class PadePSD(object):
    def __init__(self, network, stop_time, cut_off_time, s_vals, test_s_values=None):
        self.network = network
        self.stop_time = stop_time
        self.cut_off_time = cut_off_time
        self.unique_states = None
        self.approximation_order = None
        self.s_vals = s_vals
        self.output_species = self.network.species_labels.index(network.output_species_labels[0])
        self.test_s_values = test_s_values

    def gillespie_ssa_next_reaction_aux(self, state_aux):
        state = state_aux[:self.network.num_species]
        prop = self.network.propensity_vector(state)
        if self.s_vals[-1] != np.infty:
            s_values = self.s_vals
        else:
            s_values = self.s_vals[:-1]
        sum_prop = np.sum(prop) + np.sum(s_values) + np.sum(self.test_s_values)
        if sum_prop == 0:
            delta_t = math.inf
            next_reaction = -1
        else:
            prop = np.concatenate((prop, np.insert(self.test_s_values, 0, s_values)), axis=None)
            prop = np.cumsum(np.divide(prop, sum_prop))
            delta_t = -math.log(np.random.uniform(0, 1)) / sum_prop
            next_reaction = sum(prop < np.random.uniform(0, 1))
        return delta_t, next_reaction

    def update_state_aux(self, next_reaction, state_aux):
        if self.s_vals[-1] != np.infty:
            approx_order = self.approximation_order
        else:
            approx_order = self.approximation_order[:-1]
        state = state_aux[:self.network.num_species]
        hist = state_aux[self.network.num_species:self.network.num_species + np.sum(approx_order)]
        aux_hist = state_aux[self.network.num_species + np.sum(approx_order):]
        if next_reaction != -1:
            if next_reaction < self.network.num_reactions:
                state = self.network.update_state(next_reaction, state)
            elif next_reaction < self.network.num_reactions + np.shape(approx_order)[0]:
                temp_array = np.cumsum(np.insert(approx_order, 0, 0))
                starting_index = temp_array[next_reaction - self.network.num_reactions]
                ending_index = temp_array[next_reaction - self.network.num_reactions + 1]
                hist = np.insert(hist, starting_index, state[self.output_species])
                hist = np.delete(hist, ending_index)
            else:
                aux_hist[next_reaction - self.network.num_reactions - np.shape(approx_order)[0]] = state[
                    self.output_species]
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
        if self.s_vals[-1] != np.infty:
            approx_order = self.approximation_order
        else:
            approx_order = self.approximation_order[:-1]
        if seed is None:
            random.seed(seed)
        sampling_times = np.linspace(cut_off_time, stop_time, num_time_samples)
        state_curr = np.append(self.network.initial_state,
                               np.zeros(np.sum(approx_order) + np.shape(self.test_s_values)[0]))
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
        if self.s_vals[-1] != np.infty:
            approx_order = self.approximation_order
        else:
            approx_order = self.approximation_order[:-1]
        states_trajectories = np.zeros(
            [num_trajectories, num_time_samples,
             (self.network.num_species + np.sum(approx_order) + np.shape(self.test_s_values)[0])])
        for i in range(num_trajectories):
            times, states_trajectories[i, :, :] = \
                self.generate_sampled_ssa_trajectory_aux(cut_off_time, stop_time, num_time_samples, seed)
        return times, states_trajectories

    def RecursiveAdd(self, state, n):
        if n == 0:
            self.unique_states.add(np.array2string(state))
        else:
            self.RecursiveAdd(state, n - 1)
            prop = self.network.propensity_vector(state)
            for k in np.arange(self.network.num_reactions):
                if prop[k] > 0:
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
        if self.s_vals[-1] != np.infty:
            approx_order = self.approximation_order
            approx_order_infty = 0
        else:
            approx_order = self.approximation_order[:-1]
            approx_order_infty = self.approximation_order[-1]
        temp_array = np.cumsum(np.insert(approx_order, 0, 0))
        output_species_index = self.network.species_labels.index(self.network.output_species_labels[0])
        print('Generating trajectories of the augmented CTMC')
        sampling_times, states_array = self.generate_sampled_ssa_trajectories_aux(self.cut_off_time, self.stop_time,
                                                                                  num_time_samples,
                                                                                  num_trajectories,
                                                                                  seed=None)
        print('Finished generating trajectories')
        if self.s_vals[-1] == np.infty and self.approximation_order[-1] > 0:
            self.unique_states = set()
            for trj in np.arange(num_trajectories):
                for i in np.arange(num_time_samples):
                    state_aux = states_array[trj, i, :]
                    state = state_aux[:self.network.num_species]
                    self.RecursiveAdd(state, self.approximation_order[-1] - 1)
            self.unique_states = {key: np.full(self.approximation_order[-1], None) for key in self.unique_states}
            print('Finished creating state dictionary')
            ListSize = len(self.unique_states.keys())
            print("No. of unique states", ListSize)
        PadeDerivatives_values = {}
        for i in range(np.shape(self.s_vals)[0]):
            PadeDerivatives_values[self.s_vals[i]] = np.zeros([num_trajectories, self.approximation_order[i]])
        G_at_test_s_values = np.zeros([num_trajectories, np.shape(self.test_s_values)[0]])
        mean_output = np.zeros([num_trajectories])
        total_power = np.zeros([num_trajectories])
        for trj in np.arange(num_trajectories):
            # print("Starting trajectory: ", trj + 1)
            time_counter = 0
            for i in np.arange(num_time_samples):
                # if sampling_times[i] > self.cut_off_time + time_counter * 10:
                #     print("Reached time", self.cut_off_time + time_counter * 10)
                #     time_counter += 1
                state_aux = states_array[trj, i, :]
                state = state_aux[:self.network.num_species]
                hist = state_aux[self.network.num_species:self.network.num_species + np.sum(approx_order)]
                aux_hist = state_aux[self.network.num_species + np.sum(approx_order):]
                total_power[trj] += (state[output_species_index] ** 2)
                for l in np.arange(np.shape(self.test_s_values)[0]):
                    G_at_test_s_values[trj, l] += (state[output_species_index] - aux_hist[l]) ** 2
                mean_output[trj] += state[output_species_index]
                for n in np.arange(1, approx_order_infty):
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
                    PadeDerivatives_values[self.s_vals[-1]][trj, n] += -   inc
                for counter in np.arange(np.shape(approx_order)[0]):
                    for n in np.arange(approx_order[counter]):
                        PadeDerivatives_values[self.s_vals[counter]][trj, n] += (
                                                                                        state[output_species_index] -
                                                                                        hist[temp_array[
                                                                                                 counter] + n]) ** 2
        mean_output = mean_output / num_time_samples
        total_power = total_power / num_time_samples - mean_output ** 2
        for l in np.arange(np.shape(self.test_s_values)[0]):
            G_at_test_s_values[:, l] = - 0.5 * G_at_test_s_values[:, l] / num_time_samples + total_power
            G_at_test_s_values[:, l] = G_at_test_s_values[:, l] / self.test_s_values[l]
        if approx_order_infty > 0:
            PadeDerivatives_values[self.s_vals[-1]][:, 0] = total_power
            for n in np.arange(1, approx_order_infty):
                PadeDerivatives_values[self.s_vals[-1]][:, n] = PadeDerivatives_values[self.s_vals[-1]][:, n] \
                                                                / num_time_samples

        for counter in np.arange(np.shape(approx_order)[0]):
            for n in np.arange(approx_order[counter]):
                PadeDerivatives_values[self.s_vals[counter]][:, n] = - 0.5 * PadeDerivatives_values[
                                                                                 self.s_vals[counter]][:,
                                                                             n] / num_time_samples + total_power
                PadeDerivatives_values[self.s_vals[counter]][:, n] = ((-1) ** n) * \
                                                                     PadeDerivatives_values[self.s_vals[counter]][:, n] \
                                                                     / (self.s_vals[counter] ** (n + 1))
        pade_dict = {}
        for i in range(np.shape(self.s_vals)[0]):
            pade_dict[self.s_vals[i]] = [np.mean(PadeDerivatives_values[self.s_vals[i]], axis=0),
                                         np.std(PadeDerivatives_values[self.s_vals[i]], axis=0)]
        pade_dict['Test_s_vals'] = [np.mean(G_at_test_s_values, axis=0), np.std(G_at_test_s_values, axis=0)]
        pade_dict['total_power'] = np.average(total_power)
        return pade_dict


class EstimateGain(object):
    def __init__(self, network, s_vals):
        self.network = network
        self.s_vals = s_vals

    def EstimateMeanLaplaceTransform(self, G_p, input_reaction=0, num_trajectories=1):
        output_species_index = self.network.species_labels.index(self.network.output_species_labels[0])
        nom, denom = sym.fraction(G_p)
        x = sym.Symbol('x')
        if self.s_vals[-1] != np.infty:
            s_values = np.sort(self.s_vals)
        else:
            s_values = np.sort(self.s_vals[:-1])
        # save the original values for restoration
        orig_param_dict = self.network.parameter_dict
        # identify the zeroth-order inflow reactions
        for j in np.arange(self.network.num_reactions):
            if np.sum(self.network.reactant_matrix[j, :]) == 0:
                print(f"Reaction {j} is zeroth order")
                if self.network.reaction_dict[j][0] == 'mass action':
                    param = self.network.reaction_dict[j][1]
                elif self.network.reaction_dict[j][0] == 'Hill_repression' or self.network.reaction_dict[j][0] == \
                        'Hill_activation':
                    param = self.network.reaction_dict[j][5]
                elif self.network.reaction_dict[j][0] == 'linear':
                    param = self.network.reaction_dict[j][2]
                self.network.parameter_dict[param] = 0
        self.network.set_propensity_vector()
        # do the computations
        output_state_matrix = np.zeros([num_trajectories, len(s_values)])
        for j in np.arange(num_trajectories):
            u = -np.log(np.random.uniform())
            tau_values = u / s_values
            t = 0
            state = self.network.product_matrix[input_reaction, :]
            for i in np.arange(len(s_values)):
                state = self.network.run_gillespie_ssa(state, tau_values[len(s_values) - i - 1] - t)
                t = tau_values[len(s_values) - i - 1]
                output_state_matrix[j, len(s_values) - i - 1] = state[output_species_index]

        for i in np.arange(len(s_values)):
            s_val = s_values[len(s_values) - i - 1]
            output_state_matrix[:, len(s_values) - i - 1] = denom.subs(x, s_val) * output_state_matrix[:,
                                                                                   len(s_values) - i - 1] / s_val

        means = np.mean(output_state_matrix, axis=0)
        stdevs = np.std(output_state_matrix, axis=0)
        print(f"Mean:{means}")
        print(f"Stdevs. {stdevs}")
        # restore the network
        self.network.parameter_dict = orig_param_dict
        self.network.set_propensity_vector()
        return means, stdevs
