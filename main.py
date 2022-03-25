import sys
import ReactionNetworkExamples as rxn_examples
import numpy as np
import PSD_Estimator as psd
import json
import pickle
import time
import os
import PadePSD
import plotting as plt_file
import matplotlib.pyplot as plt

def main(argv):
    print(argv)
    config_filename = "./Configs/" + argv[1]
    print("The configuration file is: " + config_filename)
    config_file = open(config_filename, )
    config_data = json.load(config_file)
    config_file.close()
    # get the network
    network = getattr(rxn_examples, config_data['network_config']['network_name'])()
    # set up output folder
    results_folder_path = "./Results/" + config_data['network_config']['output_folder_name'] + "/"
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    stop_time = config_data['network_config']['stop_time']
    cut_off_time = config_data['network_config']['cut_off_time']
    output_species = network.species_labels.index(network.output_species_labels[0])

    # generate sampled trajectories if needed
    if config_data['data sampling_config']['new_data_required'] == "True":
        start_time = time.time()
        print("Starting simulated data generation")
        num_time_samples = config_data['data sampling_config']['num_time_samples']
        num_trajectories = config_data['data sampling_config']['num_trajectories']
        sampling_times, states_array = network.generate_sampled_ssa_trajectories(cut_off_time, stop_time,
                                                                                 num_time_samples, num_trajectories)
        print("Finished simulated data generation in %3u seconds" % (time.time() - start_time))
        output_states_array = states_array[:, :, output_species]
        header_list = 'Time'
        for i in range(1, num_trajectories + 1):
            header_list += ',Trajectory {}'.format(i)
        np.savetxt('{}Trajectory_Data.csv'.format(results_folder_path),
                   np.transpose(np.vstack((sampling_times, states_array[:, :, output_species]))),
                   fmt=['%.5f'] + ['%d' for _ in range(num_trajectories)],
                   header=header_list,
                   delimiter=",",
                   comments='')
    else:
        trajectory_data = np.transpose(np.loadtxt('{}Trajectory_Data.csv'.format(results_folder_path),
                                                  delimiter=",", skiprows=1))
        sampling_times = trajectory_data[0, :]
        output_states_array = trajectory_data[1:np.shape(trajectory_data)[0], :]
        del trajectory_data
    # compute PSD from simulated data
    omega_dft, psd_dft, Mean_output_dft, Standard_dev_output_dft = psd.PSD_Estimator_Numerical(sampling_times, output_states_array)
    psd_dft_mean = np.mean(psd_dft, axis=0)
    psd_dft_std = np.std(psd_dft, axis=0)

    s_vals = config_data['Pade_Approximation_config']['s_vals']
    if s_vals[-1] == "infinity":
        s_vals[-1] = np.infty
    order = config_data['Pade_Approximation_config']['order']
    test_s_values = np.array(config_data['Pade_Approximation_config']['test_s_values'])
    p = config_data['Pade_Approximation_config']['rational_approx_degree']
    known_den_poly = config_data['Pade_Approximation_config']['known_den_poly_coeffs']
    # known_num_poly = config_data['Pade_Approximation_config']['known_num_poly_coeffs']
    # estimate Pade derivatives if needed
    if config_data['Pade_Approximation_config']['pade_derivative_estimation_required'] == "True":

        num_trajectories = config_data['Pade_Approximation_config']['num_trajectories']
        num_time_samples = config_data['Pade_Approximation_config']['num_time_samples']
        # set up the Pade PSD object
        start_time = time.time()
        pade_psd = PadePSD.PadePSD(network, stop_time=stop_time, cut_off_time=cut_off_time, s_vals=s_vals,
                                   test_s_values=test_s_values)
        print("Starting estimation of the required Pade derivatives and the direct G(s) estimates")
        pade_dict = pade_psd.EstimatePadeDerivatives(order=order, num_time_samples=num_time_samples, num_trajectories=num_trajectories)
        print("Estimation of the required quantities completed in %3u seconds\n" % (time.time() - start_time))
        # save the data
        Pade_derivative_data_file = open(results_folder_path + "Pade_derivative_data" + ".pkl", "wb")
        pickle.dump(pade_dict, Pade_derivative_data_file)
        Pade_derivative_data_file.close()
    else:
        Pade_derivative_data_file = open(results_folder_path + "Pade_derivative_data" + ".pkl", "rb")
        pade_dict = pickle.load(Pade_derivative_data_file)
    print("\nEstimated Pade derivatives are as follows:\n")
    for i in range(np.shape(s_vals)[0]):
        if s_vals[i] != np.infty:
            print("for s = %3f" % s_vals[i])
        else:
            print("for s = infinity")
        print('order, mean, std.')
        np.savetxt(sys.stdout, np.transpose(
            np.vstack((np.arange(order[i]), pade_dict[s_vals[i]][0][:order[i]], pade_dict[s_vals[i]][1][:order[i]]))),
                   fmt="%d %.3f %.3f")
        print()

    print("Estimated validation test values are as follows:\n")
    print('s mean, std.')
    np.savetxt(sys.stdout,
               np.transpose(np.vstack((test_s_values, pade_dict['Test_s_vals'][0], pade_dict['Test_s_vals'][1]))),
               fmt="%.3f %.3f %.3f")
    total_power = pade_dict['total_power']
    print()


    pade_derivatives = pade_dict
    pade_psd_function, G_p = PadePSD.ConstructMultipointPadeApproximant(pade_derivatives, s_vals, order, p,
                                                                        known_den_poly)
    print("G_p(x):\n", G_p)
    print("PSD Expression:\n", pade_psd_function)
    # save the G_p and PSD
    Gp_data_file = open(results_folder_path + "G_p" + ".pkl", "wb")
    pickle.dump(G_p, Gp_data_file)
    Pade_derivative_data_file.close()

    omega_limit = config_data['Plotting_config']['ratio_omega_max_vs_pi'] * np.pi
    omega_pade, psd_pade = PadePSD.ConstructPadeApproximationPSD(pade_psd_function, omega_limit)

    # compute analytical PSD if available
    if config_data['network_config']['analytical_psd_available'] == "True":
        omega_anal, psd_anal = network.PSDArray(omega_limit)
    else:
        omega_anal = None
        psd_anal = None

    # make a dictionary of all PSD estimations
    psd_dict = {'DFT': [omega_dft, psd_dft_mean, psd_dft_std],
                'Pade': [omega_pade, psd_pade],
                'Analytical': [omega_anal, psd_anal]
                }

    if config_data['Plotting_config']['normalization'] == "True":
        TotalPowerPade = total_power * np.pi
        TotalPowerExperimental = sum(psd_dft_mean) * (omega_dft[1] - omega_dft[0])
        psd_dict['DFT'][1] = psd_dict['DFT'][1] / TotalPowerExperimental
        psd_dict['DFT'][2] = psd_dict['DFT'][2] / TotalPowerExperimental
        psd_dict['Pade'][1] = psd_dict['Pade'][1] / TotalPowerPade
        if config_data['network_config']['analytical_psd_available'] == "True":
            psd_dict['Analytical'][1] = psd_dict['Analytical'][1] / TotalPowerPade

    save_pdf = True
    number_of_plot_trajectories = config_data['Plotting_config']['num_trajectories']
    trajectory_length = config_data['Plotting_config']['trajectory_length']
    plt_file.plotFunctionTrajectory(sampling_times, output_states_array[:number_of_plot_trajectories, :],
                                    number_of_plot_trajectories, trajectory_length, results_folder_path,
                                    save_pdf)
    plt_file.generate_psd_comparison_plot(psd_dict, network.time_unit, results_folder_path, omega_limit, save_pdf)
    plt_file.generate_test_s_values_comparison(test_s_values, pade_dict['Test_s_vals'][0], pade_dict['Test_s_vals'][1],
                                               G_p,
                                               results_folder_path,
                                               save_pdf)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
