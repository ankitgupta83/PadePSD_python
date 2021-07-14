import ReactionNetworkExamples as rxn_examples
import numpy as np
import PSD_Estimator as psd
import json
import time
import os
import sys
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
    omega_dft, psd_dft = psd.PSD_Estimator_Numerical(sampling_times, output_states_array)
    psd_dft_mean = np.mean(psd_dft, axis=0)
    psd_dft_std = np.std(psd_dft, axis=0)

    s0 = config_data['Pade_Approximation_config']['s_0']
    order = config_data['Pade_Approximation_config']['order']
    test_s_values = np.array(config_data['Pade_Approximation_config']['test_s_values'])
    # estimate Pade derivatives if needed
    if config_data['Pade_Approximation_config']['pade_derivative_estimation_required'] == "True":

        num_trajectories = config_data['Pade_Approximation_config']['num_trajectories']
        num_time_samples = config_data['Pade_Approximation_config']['num_time_samples']
        # set up Pade PSD object
        start_time = time.time()
        pade_psd = PadePSD.PadePSD(network, stop_time=stop_time, cut_off_time=cut_off_time, s0=s0,
                                   test_s_values=test_s_values)
        print("Starting estimation of the required Pade derivatives and the direct G(s) estimates")
        pade_dict = pade_psd.EstimatePadeDerivatives(order=order, num_time_samples=num_time_samples
                                                     , num_trajectories=num_trajectories)
        print("Estimation of the required quantities completed in %3u seconds\n" % (time.time() - start_time))
        pade_derivatives_infty_mean = pade_dict['Infinity'][0]
        pade_derivatives_infty_std = pade_dict['Infinity'][1]
        pade_derivatives_s0_mean = pade_dict['s0'][0]
        pade_derivatives_finite_s_std = pade_dict['s0'][1]
        test_s_values_mean = pade_dict['Test_s_vals'][0]
        test_s_values_std = pade_dict['Test_s_vals'][1]
        header_list = 'order, s0 mean, s0 std.'
        np.savetxt('{}Pade_Derivative_s0_Data.csv'.format(results_folder_path),
                   np.transpose(
                       np.vstack((np.arange(order[0]), pade_derivatives_s0_mean, pade_derivatives_finite_s_std))),
                   fmt=['%d', '%.5f', '%.5f'],
                   header=header_list,
                   delimiter=",",
                   comments='')
        header_list = 'order, infinity mean, infinity std.'
        np.savetxt('{}Pade_Derivative_infty_Data.csv'.format(results_folder_path),
                   np.transpose(
                       np.vstack(
                           (np.arange(max(order[1], 1)), pade_derivatives_infty_mean, pade_derivatives_infty_std))),
                   fmt=['%d', '%.5f', '%.5f'],
                   header=header_list,
                   delimiter=",",
                   comments='')
        np.savetxt('{}Test_values_data.csv'.format(results_folder_path),
                   np.transpose(
                       np.vstack(
                           (test_s_values, test_s_values_mean, test_s_values_std))),
                   fmt=['%.3f', '%.5f', '%.5f'],
                   header='s_value, mean, std.',
                   delimiter=",",
                   comments='')
    else:
        pade_Derivative_s0_data = np.transpose(
            np.loadtxt('{}Pade_Derivative_s0_Data.csv'.format(results_folder_path),
                       delimiter=",", skiprows=1))
        pade_Derivative_infty_data = np.transpose(
            np.loadtxt('{}Pade_Derivative_infty_Data.csv'.format(results_folder_path),
                       delimiter=",", skiprows=1))
        if pade_Derivative_s0_data.ndim == 1:
            pade_Derivative_s0_data = np.expand_dims(pade_Derivative_s0_data, axis=1)
        if pade_Derivative_infty_data.ndim == 1:
            pade_Derivative_infty_data = np.expand_dims(pade_Derivative_infty_data, axis=1)
        if order[1] > 0:
            pade_derivatives_infty_mean = pade_Derivative_infty_data[1, :]
        else:
            pade_derivatives_infty_mean = None
        if order[0] > 0:
            pade_derivatives_s0_mean = pade_Derivative_s0_data[1, :]
        else:
            pade_derivatives_s0_mean = None
        del pade_Derivative_s0_data, pade_Derivative_infty_data

        test_s_values_data = np.transpose(
            np.loadtxt('{}Test_values_data.csv'.format(results_folder_path),
                       delimiter=",", skiprows=1))
        test_s_values_mean = test_s_values_data[1, :]
        test_s_values_std = test_s_values_data[2, :]
        del test_s_values_data

    pade_psd_function, G_p = PadePSD.ConstructPadeApproximation(pade_derivatives_infty_mean,
                                                                pade_derivatives_s0_mean, s0, order)

    print("G_p(x):\n", G_p)
    print("PSD Expression:\n", pade_psd_function)
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
        TotalPowerPade = pade_derivatives_infty_mean[0] * np.pi
        TotalPowerExperimental = sum(psd_dft_mean)*(omega_dft[1] - omega_dft[0])
        psd_dict['DFT'][1] = psd_dict['DFT'][1] / TotalPowerExperimental
        psd_dict['DFT'][2] = psd_dict['DFT'][2] / TotalPowerExperimental
        psd_dict['Pade'][1] = psd_dict['Pade'][1] / TotalPowerPade
        if config_data['network_config']['analytical_psd_available'] == "True":
            psd_dict['Analytical'][1] = psd_dict['Analytical'][1] / TotalPowerPade

    save_pdf = False
    number_of_plot_trajectories = config_data['Plotting_config']['num_trajectories']
    trajectory_length = config_data['Plotting_config']['trajectory_length']
    plt_file.plotFunctionTrajectory(sampling_times, output_states_array[:number_of_plot_trajectories, :],
                                    number_of_plot_trajectories, trajectory_length, results_folder_path,
                                    save_pdf)
    plt_file.generate_psd_comparison_plot(psd_dict, network.time_unit, results_folder_path, omega_limit, save_pdf)
    plt_file.generate_test_s_values_comparison(test_s_values, test_s_values_mean, test_s_values_std, G_p,
                                               results_folder_path,
                                               save_pdf)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
