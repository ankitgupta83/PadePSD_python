import numpy as np
import ReactionNetworkClass as rxn
import scipy.optimize
from scipy import linalg
import sympy as sym


class birth_death_network(rxn.ReactionNetworkDefinition):
    """birth death network"""

    def __init__(self):
        num_species = 1
        num_reactions = 2
        species_labels = ["protein"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        reactant_matrix[1, 0] = 1  # X --> 0
        product_matrix[0, 0] = 1  # 0 --> X
        parameter_dict = {'birth rate': 10.0, 'degradation rate': 1.0}
        time_unit = "sec."
        reaction_dict = {0: ['mass action', 'birth rate'],
                         1: ['mass action', 'degradation rate']
                         }

        super(birth_death_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                  product_matrix, parameter_dict, reaction_dict,
                                                  species_labels, output_species_labels, time_unit)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()

    def PSD_analytical(self, omega):
        k = self.parameter_dict['birth rate']
        gamma = self.parameter_dict['degradation rate']
        psd = 2 * k / (omega ** 2 + gamma ** 2)
        return psd

class self_regulatory_network(rxn.ReactionNetworkDefinition):
    """birth death network"""

    def __init__(self):
        num_species = 1
        num_reactions = 2
        species_labels = ["protein"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        reactant_matrix[1, 0] = 1  # X --> 0
        product_matrix[0, 0] = 1  # 0 --> X
        parameter_dict = {'birth rate': 10.0, 'degradation rate': 0.05, 'Hill coefficient': 0.5, 'Hill constant': 10.0,
                          'basal rate': 0}
        time_unit = "sec."
        reaction_dict = {  # 0: ['mass action', 'birth rate'],
            0: ['Hill_repression', 0, 'birth rate', 'Hill constant', 'Hill coefficient', 'basal rate'],
            1: ['mass action', 'degradation rate']
        }

        super(self_regulatory_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                      product_matrix, parameter_dict, reaction_dict,
                                                      species_labels, output_species_labels, time_unit)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()


class cons_gene_expression_network(rxn.ReactionNetworkDefinition):
    """constitutive gene-expression network"""

    def __init__(self):
        num_species = 2
        num_reactions = 4
        species_labels = ["mRNA", "protein"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. 0 --> M
        product_matrix[0, 0] = 1
        # 2. M --> M + P
        reactant_matrix[1, 0] = 1
        product_matrix[1, 0] = 1
        product_matrix[1, 1] = 1
        # 3. M --> 0
        reactant_matrix[2, 0] = 1
        # 4. P -->0
        reactant_matrix[3, 1] = 1

        # define parameters
        parameter_dict = {'transcription rate': 10, 'translation rate': 2, 'mRNA degradation rate': 1,
                          'protein degradation rate': 0.5}
        time_unit = "min."
        reaction_dict = {0: ['mass action', 'transcription rate'],
                         1: ['mass action', 'translation rate'],
                         2: ['mass action', 'mRNA degradation rate'],
                         3: ['mass action', 'protein degradation rate']
                         }
        super(cons_gene_expression_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                           product_matrix, parameter_dict, reaction_dict,
                                                           species_labels, output_species_labels, time_unit)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()

    def PSD_analytical(self, omega):
        k_r = self.parameter_dict['transcription rate']
        k_p = self.parameter_dict['translation rate']
        gamma_r = self.parameter_dict['mRNA degradation rate']
        gamma_p = self.parameter_dict['protein degradation rate']
        psd = 2 * k_r * k_p / (gamma_r * (gamma_p ** 2 + omega ** 2)) + 2 * k_r * (k_p ** 2) / (
                (gamma_r ** 2 + omega ** 2)
                * (gamma_p ** 2 + omega ** 2))
        return psd


class cons_gene_expression_cell_cycle_network(rxn.ReactionNetworkDefinition):
    """constitutive gene-expression network with cell cycle"""

    def __init__(self):
        num_species = 4
        num_reactions = 5
        species_labels = ["mRNA", "protein", "cell cycle counter", "transition_indicator"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. 0 --> M
        product_matrix[0, 0] = 1
        # 2. M --> M + P
        reactant_matrix[1, 0] = 1
        product_matrix[1, 0] = 1
        product_matrix[1, 1] = 1
        # 3. M --> 0
        reactant_matrix[2, 0] = 1
        # 4. P -->0
        reactant_matrix[3, 1] = 1
        # 5  0 --> cell cycle counter
        product_matrix[4, 2] = 1

        # define parameters
        parameter_dict = {'transcription rate': 10, 'translation rate': 2, 'mRNA degradation rate': 1,
                          'protein degradation rate': 0.5, 'cell cycle progression rate': 2.5,
                          'cell_cycle_length': 10}
        time_unit = "min."
        reaction_dict = {0: ['mass action', 'transcription rate'],
                         1: ['mass action', 'translation rate'],
                         2: ['mass action', 'mRNA degradation rate'],
                         3: ['mass action', 'protein degradation rate'],
                         4: ['mass action', 'cell cycle progression rate']
                         }
        super(cons_gene_expression_cell_cycle_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                                      product_matrix, parameter_dict, reaction_dict,
                                                                      species_labels, output_species_labels, time_unit)
        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()

    # def update_state(self, next_reaction, state):
    #     if next_reaction != -1:
    #         state = state + self.stoichiometry_matrix[next_reaction, :]
    #     if next_reaction == 4:
    #         state[2] = state[2] % self.parameter_dict['cell_cycle_length']
    #         if state[2] == (self.parameter_dict['cell_cycle_length'] - 1):
    #             state[3] = 1
    #         if state[2] == 0:
    #             state[3] = 0
    #             state[0] = state[0] // 2  # partitioning in half
    #             state[1] = state[1] // 2
    #     return state

    # symmetric binomial partition
    def update_state(self, next_reaction, state):
        prob = 0.5
        if next_reaction != -1:
            state = state + self.stoichiometry_matrix[next_reaction, :]
        if next_reaction == 4:
            state[2] = state[2] % self.parameter_dict['cell_cycle_length']
            if state[2] == (self.parameter_dict['cell_cycle_length'] - 1):
                state[3] = 1
            if state[2] == 0:
                state[3] = 0
                state[0] = np.random.binomial(state[0], prob)
                state[1] = np.random.binomial(state[1], prob)
        return state


class feedback_gene_expression_network(rxn.ReactionNetworkDefinition):
    """nonlinear feedback gene-expression network"""

    def __init__(self):
        num_species = 2
        num_reactions = 4
        species_labels = ["mRNA", "protein"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. 0 --> M
        product_matrix[0, 0] = 1
        # 2. M --> M + P
        reactant_matrix[1, 0] = 1
        product_matrix[1, 0] = 1
        product_matrix[1, 1] = 1
        # 3. M --> 0
        reactant_matrix[2, 0] = 1
        # 4. P -->0
        reactant_matrix[3, 1] = 1

        # define parameters
        parameter_dict = {'base transcription rate': 10.0, 'Hill constant': 10.0, 'Hill coefficient': 1.0,
                          'basal rate': 135.81644,  # theta * 67.90822
                          'translation rate': 1.0,
                          'mRNA degradation rate': 1.0, 'protein degradation rate': 1.0}
        time_unit = "sec."
        reaction_dict = {0: ['Hill_repression', 1, 'base transcription rate', 'Hill constant', 'Hill coefficient',
                             'basal rate'],
                         1: ['mass action', 'translation rate'],
                         2: ['mass action', 'mRNA degradation rate'],
                         3: ['mass action', 'protein degradation rate']
                         }
        super(feedback_gene_expression_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                               product_matrix, parameter_dict, reaction_dict,
                                                               species_labels, output_species_labels, time_unit)

        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()


class negative_feedback_network(rxn.ReactionNetworkDefinition):
    """negative feedback network with linearisation"""

    def __init__(self):
        num_species = 2
        num_reactions = 4
        species_labels = ["controller", "output"]
        output_species_labels = ["output"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. I_0 --> I_0 + C
        product_matrix[0, 0] = 1
        # 2. C --> C + O
        reactant_matrix[1, 0] = 1
        product_matrix[1, 0] = 1
        product_matrix[1, 1] = 1
        # 3. C --> 0
        reactant_matrix[2, 0] = 1
        # 4. O -->0
        reactant_matrix[3, 1] = 1

        # define parameters
        parameter_dict = {'constant term': 50.0, 'slope': -0.5,
                          'production rate': 2.0,
                          'controller degradation rate': 1.0, 'output degradation rate': 0.5}
        time_unit = "sec."
        reaction_dict = {0: ['linear', 1, 'constant term', 'slope'],
                         1: ['mass action', 'production rate'],
                         2: ['mass action', 'controller degradation rate'],
                         3: ['mass action', 'output degradation rate']
                         }
        super(negative_feedback_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                        product_matrix, parameter_dict, reaction_dict,
                                                        species_labels, output_species_labels, time_unit)

        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()

    def PSD_analytical(self, omega):
        beta_0 = self.parameter_dict['constant term']
        beta_fb = -self.parameter_dict['slope']
        k_o = self.parameter_dict['production rate']
        gamma_c = self.parameter_dict['controller degradation rate']
        gamma_o = self.parameter_dict['output degradation rate']
        num = 2 * gamma_o * k_o * beta_0 * (gamma_c ** 2 + k_o * gamma_c + omega ** 2)
        den = ((gamma_c * gamma_o + beta_fb * k_o) ** 2 + (omega ** 2) * (
                gamma_c ** 2 + gamma_o ** 2 - 2 * k_o * beta_fb) +
               omega ** 4) * (gamma_c * gamma_o + beta_fb * k_o)
        psd = num / den
        return psd


class negative_feedforward_network(rxn.ReactionNetworkDefinition):
    """negative feedforward network with linearisation"""

    def __init__(self):
        num_species = 2
        num_reactions = 4
        species_labels = ["controller", "output"]
        output_species_labels = ["output"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. I_0 --> I_0 + C
        product_matrix[0, 0] = 1
        # 2. I_0 --> I_0 + O
        product_matrix[1, 1] = 1
        # 3. C --> 0
        reactant_matrix[2, 0] = 1
        # 4. O -->0
        reactant_matrix[3, 1] = 1

        # define parameters
        parameter_dict = {'constant term': 50.0, 'slope': -3.0,
                          'controller production rate': 1.0,
                          'controller degradation rate': 1.0, 'output degradation rate': 0.5}
        time_unit = "sec."
        reaction_dict = {0: ['mass action', 'controller production rate'],
                         1: ['linear', 0, 'constant term', 'slope'],
                         2: ['mass action', 'controller degradation rate'],
                         3: ['mass action', 'output degradation rate']
                         }
        super(negative_feedforward_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                           product_matrix, parameter_dict, reaction_dict,
                                                           species_labels, output_species_labels, time_unit)

        self.initial_state = np.zeros(self.num_species)
        self.set_propensity_vector()

    def PSD_analytical(self, omega):
        beta_0 = self.parameter_dict['constant term']
        beta_ff = -self.parameter_dict['slope']
        k_c = self.parameter_dict['controller production rate']
        gamma_c = self.parameter_dict['controller degradation rate']
        gamma_o = self.parameter_dict['output degradation rate']
        psd = (2 / (gamma_o ** 2 + omega ** 2)) * (
                beta_0 - beta_ff * k_c / gamma_c + (beta_ff ** 2) * k_c / (gamma_c ** 2
                                                                           + omega ** 2))
        return psd


class rna_splicing_network(rxn.ReactionNetworkDefinition):
    """linear rna splicing network"""

    def __init__(self):
        num_species = 4
        num_reactions = 6
        species_labels = ["gene off", "gene on", "unspliced mrna", "spliced mrna"]
        output_species_labels = ["spliced mrna"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. gene switching on
        reactant_matrix[0, 0] = 1
        product_matrix[0, 1] = 1
        # 2. gene switching off
        reactant_matrix[1, 1] = 1
        product_matrix[1, 0] = 1
        # 3. transcription reaction
        product_matrix[2, 2] = 1
        # 4. splicing reaction
        reactant_matrix[3, 2] = 1
        product_matrix[3, 3] = 1
        # 4. degradation reaction for spliced mrna
        reactant_matrix[4, 3] = 1
        # 5. degradation reaction for the unspliced mrna
        reactant_matrix[5, 2] = 1

        # define parameters
        parameter_dict = {'k_on': 1.0, 'k_off': 3.0, 'alpha_on': 3.0, 'alpha_off': 0.2,
                          'beta': 2.0, 'gamma_r': 0.5, 'gamma_u': 1.0}

        slope = parameter_dict['alpha_on'] - parameter_dict['alpha_off']
        parameter_dict["slope"] = slope
        time_unit = "sec."
        reaction_dict = {0: ['mass action', 'k_on'],  # gene switching on
                         1: ['mass action', 'k_off'],  # gene switching off
                         2: ['linear', 1, 'alpha_off', 'slope'],  # transcription reaction
                         3: ['mass action', 'beta'],  # splicing reaction
                         4: ['mass action', 'gamma_r'],  # degradation reaction for spliced mRNA
                         5: ['mass action', 'gamma_u']  # degradation reaction for unspliced mRNA
                         }
        super(rna_splicing_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                   product_matrix, parameter_dict, reaction_dict,
                                                   species_labels, output_species_labels, time_unit)

        self.initial_state = np.zeros(self.num_species)
        self.initial_state[0] = 1
        self.set_propensity_vector()

    def PSD_analytical(self, omega):
        gamma_r = self.parameter_dict['gamma_r']
        gamma_u = self.parameter_dict['gamma_u']
        beta = self.parameter_dict['beta']
        k_on = self.parameter_dict['k_on']
        k_off = self.parameter_dict['k_off']
        alpha_on = self.parameter_dict['alpha_on']
        alpha_off = self.parameter_dict['alpha_off']
        # omega = sym.Symbol('omega')
        psd_gene_switch = 2 * k_on * k_off / ((k_on + k_off) * ((k_on + k_off) ** 2 + omega ** 2))
        mean_transcription_rate = (alpha_off * k_off + alpha_on * k_on) / (k_on + k_off)
        factor = ((alpha_on - alpha_off) ** 2) * (beta ** 2) / (
                    ((beta + gamma_u) ** 2 + omega ** 2) * (gamma_r ** 2 + omega ** 2))
        psd = 2 * beta * mean_transcription_rate / (
                    (gamma_r ** 2 + omega ** 2) * (beta + gamma_u)) + psd_gene_switch * factor
        return psd


class antithetic_gene_expression_network(rxn.ReactionNetworkDefinition):
    """antithetic gene-expression network"""

    def __init__(self):
        name = "antithetic gene expression"
        num_species = 4
        num_reactions = 7
        species_labels = ["mRNA", "protein", "Z1", "Z2"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. Z_1 --> Z_1 + M
        product_matrix[0, 2] = 1
        product_matrix[0, 0] = 1
        reactant_matrix[0, 2] = 1
        # 2. M --> M + P
        reactant_matrix[1, 0] = 1
        product_matrix[1, 0] = 1
        product_matrix[1, 1] = 1
        # 3. M --> 0
        reactant_matrix[2, 0] = 1
        # 4. P -->0
        reactant_matrix[3, 1] = 1
        # 5. P -->P + Z_2
        reactant_matrix[4, 1] = 1
        product_matrix[4, 1] = 1
        product_matrix[4, 3] = 1
        # 6. Z_1 + Z_2 -->0
        reactant_matrix[5, 2] = 1
        reactant_matrix[5, 3] = 1
        # 7. 0 --> Z_1
        product_matrix[6, 2] = 1

        # define parameters
        parameter_dict = {'activation rate': 5.0,
                          'translation rate': 2.0,
                          'mRNA degradation rate': 2.0,
                          'protein degradation rate': 1.0,
                          'theta': 1,
                          'eta': 100,
                          'mu': 10,
                          'k_fb': 5,
                          'feedback_type': 'Hill'  # could be Hill or linear
                          }
        time_unit = "sec."
        reaction_dict = {0: ['mass action', 'activation rate'],
                         1: ['mass action', 'translation rate'],
                         2: ['mass action', 'mRNA degradation rate'],
                         3: ['mass action', 'protein degradation rate'],
                         4: ['mass action', 'theta'],
                         5: ['mass action', 'eta'],
                         6: ['mass action', 'mu']
                         }
        super(antithetic_gene_expression_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                                 product_matrix, parameter_dict, reaction_dict,
                                                                 species_labels, output_species_labels, time_unit)
        self.initial_state = np.zeros(self.num_species)

    def propensity_vector(self, state):
        prop = np.zeros(self.num_reactions)
        for k in range(1, self.num_reactions):
            prop[k] = self.mass_action_propensity(state, k, self.reaction_dict[k][1])
        prop[0] = self.mass_action_propensity(state, 0, self.reaction_dict[0][1])
        reference = self.parameter_dict['mu'] / self.parameter_dict['theta']
        output = state[1]
        k_fb = self.parameter_dict['k_fb']
        if k_fb > 0 and self.parameter_dict['feedback_type'] == 'Hill':
            prop[0] += 4 * k_fb * (reference ** 2) / (reference + output)
        elif k_fb > 0 and self.parameter_dict['feedback_type'] == 'linear':
            prop[0] += k_fb * max(3 * reference - output, 0)
        return prop


def function_err(x, N_pro, p_tot, K, hill_coeff):
    return p_tot - x - 2 * N_pro * (x ** hill_coeff) / (x ** hill_coeff + K ** hill_coeff)


def ComputeFreeProteins(N_pro, p_tot, K, hill_coeff):
    return scipy.optimize.bisect(function_err, 0, p_tot, args=(N_pro, p_tot, K, hill_coeff,), xtol=2e-3, maxiter=100,
                                 full_output=False, disp=True)


class repressilator_network(rxn.ReactionNetworkDefinition):
    """repressilator network"""

    def __init__(self):
        name = "repressilator network"
        num_species = 5
        num_reactions = 10
        species_labels = ["p_tot_1", "p_tot_2", "p_tot_3", "N_o", "N_t"]
        output_species_labels = ["p_tot_2"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)

        # 1. 0 --> P_tot1
        product_matrix[0, 0] = 1
        # 2. 0 --> P_tot2
        product_matrix[1, 1] = 1
        # 3. 0 --> P_tot3
        product_matrix[2, 2] = 1
        # 4. P_tot1 -->0
        reactant_matrix[3, 0] = 1
        # 5. P_tot2 -->0
        reactant_matrix[4, 1] = 1
        # 6. P_tot3 -->0
        reactant_matrix[5, 2] = 1
        # 7. 0 --> N_0
        product_matrix[6, 3] = 1
        # 8. 0 --> N_t
        product_matrix[7, 4] = 1
        # 9. N_0 --> 0
        reactant_matrix[8, 3] = 1
        # 10. N_t --> 0
        reactant_matrix[9, 4] = 1

        # define parameters
        parameter_dict = {'lambda': 30,
                          'K1': 5,
                          'K2': 10,
                          'K3': 10,
                          'Mean_No': 10,
                          'Mean_Nt': 0,  # 40, 0
                          'Hill factor': 2,  # 1.5/2
                          'dilution rate': 1
                          }
        time_unit = "min."
        reaction_dict = {3: ['mass action', 'dilution rate'],
                         4: ['mass action', 'dilution rate'],
                         5: ['mass action', 'dilution rate'],
                         6: ['mass action', 'Mean_No'],
                         7: ['mass action', 'Mean_Nt'],
                         8: ['mass action', 'dilution rate'],
                         9: ['mass action', 'dilution rate'],
                         }
        super(repressilator_network, self).__init__(num_species, num_reactions, reactant_matrix,
                                                    product_matrix, parameter_dict, reaction_dict,
                                                    species_labels, output_species_labels, time_unit)
        self.initial_state = np.zeros(self.num_species)

    def propensity_vector(self, state):
        prop = np.zeros(self.num_reactions)
        for k in range(3, self.num_reactions):
            prop[k] = self.mass_action_propensity(state, k, self.reaction_dict[k][1])
        p_tot_1 = state[0]
        p_tot_2 = state[1]
        p_tot_3 = state[2]
        p_free_1 = ComputeFreeProteins(state[3] + state[4], p_tot_1, self.parameter_dict['K1'],
                                       self.parameter_dict['Hill factor'])
        p_free_2 = ComputeFreeProteins(state[3], p_tot_2, self.parameter_dict['K2'],
                                       self.parameter_dict['Hill factor'])
        p_free_3 = ComputeFreeProteins(state[3], p_tot_3, self.parameter_dict['K3'],
                                       self.parameter_dict['Hill factor'])
        H = self.parameter_dict['Hill factor']
        K1 = self.parameter_dict['K1']
        K2 = self.parameter_dict['K2']
        K3 = self.parameter_dict['K3']
        prop[0] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K1 ** H) / (K1 ** H + p_free_3 ** H))
        prop[1] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K2 ** H) / (K2 ** H + p_free_1 ** H))
        prop[2] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K3 ** H) / (K3 ** H + p_free_2 ** H))
        return prop


class repressilator_network_with_gene_expression(rxn.ReactionNetworkDefinition):
    """repressilator network with gene expression"""

    def __init__(self):
        name = "repressilator network"
        num_species = 7
        num_reactions = 14
        species_labels = ["p_tot_1", "p_tot_2", "p_tot_3", "N_o", "N_t", "mRNA", "protein"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)

        # 1. 0 --> P_tot1
        product_matrix[0, 0] = 1
        # 2. 0 --> P_tot2
        product_matrix[1, 1] = 1
        # 3. 0 --> P_tot3
        product_matrix[2, 2] = 1
        # 4. P_tot1 -->0
        reactant_matrix[3, 0] = 1
        # 5. P_tot2 -->0
        reactant_matrix[4, 1] = 1
        # 6. P_tot3 -->0
        reactant_matrix[5, 2] = 1
        # 7. 0 --> N_0
        product_matrix[6, 3] = 1
        # 8. 0 --> N_t
        product_matrix[7, 4] = 1
        # 9. N_0 --> 0
        reactant_matrix[8, 3] = 1
        # 10. N_t --> 0
        reactant_matrix[9, 4] = 1
        # 11. 0 --> mRNA
        product_matrix[10, 5] = 1
        # 12. mRNA --> mRNA + protein
        reactant_matrix[11, 5] = 1
        product_matrix[11, 5] = 1
        product_matrix[11, 6] = 1
        # 13. mRNA --> 0
        reactant_matrix[12, 5] = 1
        # 13. protein --> 0
        reactant_matrix[13, 6] = 1
        # define parameters
        parameter_dict = {'lambda': 30,
                          'K1': 5,
                          'K2': 10,
                          'K3': 10,
                          'Mean_No': 10,
                          'Mean_Nt': 0,
                          'Hill factor': 1.5,
                          'dilution rate': 1,
                          'theta': 0.4,
                          'k_r': 50,
                          'k_p': 2,
                          'k_fb': 0.5,
                          'gamma_r': 1,
                          'gamma_p': 0.5
                          }
        time_unit = "min."
        reaction_dict = {3: ['mass action', 'dilution rate'],
                         4: ['mass action', 'dilution rate'],
                         5: ['mass action', 'dilution rate'],
                         6: ['mass action', 'Mean_No'],
                         7: ['mass action', 'Mean_Nt'],
                         8: ['mass action', 'dilution rate'],
                         9: ['mass action', 'dilution rate'],
                         11: ['mass action', 'k_p'],
                         12: ['mass action', 'gamma_r'],
                         13: ['mass action', 'gamma_p'],
                         }

        super(repressilator_network_with_gene_expression, self).__init__(num_species, num_reactions, reactant_matrix,
                                                                         product_matrix, parameter_dict, reaction_dict,
                                                                         species_labels, output_species_labels,
                                                                         time_unit)
        self.initial_state = np.zeros(self.num_species)

    def propensity_vector(self, state):
        prop = np.zeros(self.num_reactions)
        for k in range(3, 10):
            prop[k] = self.mass_action_propensity(state, k, self.reaction_dict[k][1])
        for k in range(11, 14):
            prop[k] = self.mass_action_propensity(state, k, self.reaction_dict[k][1])
        p_tot_1 = state[0]
        p_tot_2 = state[1]
        p_tot_3 = state[2]
        prop[10] = max(self.parameter_dict['theta'] * p_tot_2 + self.parameter_dict['k_r'] -
                       self.parameter_dict['k_fb'] * state[6], 0)

        p_free_1 = ComputeFreeProteins(state[3] + state[4], p_tot_1, self.parameter_dict['K1'],
                                       self.parameter_dict['Hill factor'])
        p_free_2 = ComputeFreeProteins(state[3], p_tot_2, self.parameter_dict['K2'],
                                       self.parameter_dict['Hill factor'])
        p_free_3 = ComputeFreeProteins(state[3], p_tot_3, self.parameter_dict['K3'],
                                       self.parameter_dict['Hill factor'])
        H = self.parameter_dict['Hill factor']
        K1 = self.parameter_dict['K1']
        K2 = self.parameter_dict['K2']
        K3 = self.parameter_dict['K3']
        prop[0] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K1 ** H) / (K1 ** H + p_free_3 ** H))
        prop[1] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K2 ** H) / (K2 ** H + p_free_1 ** H))
        prop[2] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K3 ** H) / (K3 ** H + p_free_2 ** H))
        return prop


class repressilator_network_with_nonlinear_gene_expression(rxn.ReactionNetworkDefinition):
    """repressilator network with nonlinear gene expression"""

    def __init__(self):
        name = "repressilator nonlinear ge network"
        num_species = 7
        num_reactions = 14
        species_labels = ["p_tot_1", "p_tot_2", "p_tot_3", "N_o", "N_t", "mRNA", "protein"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)

        # 1. 0 --> P_tot1
        product_matrix[0, 0] = 1
        # 2. 0 --> P_tot2
        product_matrix[1, 1] = 1
        # 3. 0 --> P_tot3
        product_matrix[2, 2] = 1
        # 4. P_tot1 -->0
        reactant_matrix[3, 0] = 1
        # 5. P_tot2 -->0
        reactant_matrix[4, 1] = 1
        # 6. P_tot3 -->0
        reactant_matrix[5, 2] = 1
        # 7. 0 --> N_0
        product_matrix[6, 3] = 1
        # 8. 0 --> N_t
        product_matrix[7, 4] = 1
        # 9. N_0 --> 0
        reactant_matrix[8, 3] = 1
        # 10. N_t --> 0
        reactant_matrix[9, 4] = 1
        # 11. 0 --> mRNA
        product_matrix[10, 5] = 1
        # 12. mRNA --> mRNA + protein
        reactant_matrix[11, 5] = 1
        product_matrix[11, 5] = 1
        product_matrix[11, 6] = 1
        # 13. mRNA --> 0
        reactant_matrix[12, 5] = 1
        # 13. protein --> 0
        reactant_matrix[13, 6] = 1
        # define parameters
        parameter_dict = {'lambda': 30,
                          'K1': 5,
                          'K2': 10,
                          'K3': 10,
                          'Mean_No': 10,
                          'Mean_Nt': 0,
                          'Hill factor': 1.5,
                          'dilution rate': 1,
                          'theta': 0.4,
                          'base transcription rate': 10.0,
                          'Hill constant': 10.0,
                          'Hill coefficient': 1.0,
                          'k_p': 2,
                          'gamma_r': 1,
                          'gamma_p': 0.5
                          }
        time_unit = "min."
        reaction_dict = {3: ['mass action', 'dilution rate'],
                         4: ['mass action', 'dilution rate'],
                         5: ['mass action', 'dilution rate'],
                         6: ['mass action', 'Mean_No'],
                         7: ['mass action', 'Mean_Nt'],
                         8: ['mass action', 'dilution rate'],
                         9: ['mass action', 'dilution rate'],
                         11: ['mass action', 'k_p'],
                         12: ['mass action', 'gamma_r'],
                         13: ['mass action', 'gamma_p'],
                         }

        super(repressilator_network_with_nonlinear_gene_expression, self).__init__(num_species, num_reactions,
                                                                                   reactant_matrix,
                                                                                   product_matrix, parameter_dict,
                                                                                   reaction_dict,
                                                                                   species_labels,
                                                                                   output_species_labels,
                                                                                   time_unit)
        self.initial_state = np.zeros(self.num_species)

    def propensity_vector(self, state):
        prop = np.zeros(self.num_reactions)
        for k in range(3, 10):
            prop[k] = self.mass_action_propensity(state, k, self.reaction_dict[k][1])
        for k in range(11, 14):
            prop[k] = self.mass_action_propensity(state, k, self.reaction_dict[k][1])
        p_tot_1 = state[0]
        p_tot_2 = state[1]
        p_tot_3 = state[2]
        prop[10] = self.parameter_dict['theta'] * p_tot_2 + self.parameter_dict['base transcription rate'] / (
                self.parameter_dict['Hill constant'] + (state[6] ** self.parameter_dict['Hill coefficient']))
        p_free_1 = ComputeFreeProteins(state[3] + state[4], p_tot_1, self.parameter_dict['K1'],
                                       self.parameter_dict['Hill factor'])
        p_free_2 = ComputeFreeProteins(state[3], p_tot_2, self.parameter_dict['K2'],
                                       self.parameter_dict['Hill factor'])
        p_free_3 = ComputeFreeProteins(state[3], p_tot_3, self.parameter_dict['K3'],
                                       self.parameter_dict['Hill factor'])
        H = self.parameter_dict['Hill factor']
        K1 = self.parameter_dict['K1']
        K2 = self.parameter_dict['K2']
        K3 = self.parameter_dict['K3']
        prop[0] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K1 ** H) / (K1 ** H + p_free_3 ** H))
        prop[1] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K2 ** H) / (K2 ** H + p_free_1 ** H))
        prop[2] = self.parameter_dict['lambda'] * self.parameter_dict['Mean_No'] * (
                (K3 ** H) / (K3 ** H + p_free_2 ** H))
        return prop