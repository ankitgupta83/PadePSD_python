from scipy.fft import fft, fftfreq
import numpy as np


def PSD_Estimator_Numerical(sampling_times, state_values):
    delta_t = (sampling_times[1] - sampling_times[0])
    num_trajectories = np.shape(state_values)[0]
    num_sample_points = np.shape(state_values)[1]
    psd_vals = []
    mean_zero_state_values = np.zeros(np.shape(state_values))
    Mean_output = np.mean(np.mean(state_values, axis=1))
    Standard_dev_output = np.mean(np.std(state_values, axis=1))
    CV = Standard_dev_output/Mean_output
    print(f"Output statistics: Mean  = {Mean_output}, Std. = {Standard_dev_output} and CV  = {CV}")
    print("Num trajectories =  %5u" % num_trajectories)
    for i in range(num_trajectories):
        # subtract signal mean
        mean_zero_state_values[i, :] = state_values[i, :] - np.mean(state_values[i, :])
        # compute FFT and transform to PSD
        psd = (delta_t / num_sample_points) * np.abs(fft(mean_zero_state_values[i, :]))[:num_sample_points // 2] ** 2
        psd_vals.append(psd)
    omega = 2 * np.pi * fftfreq(num_sample_points, delta_t)[:num_sample_points // 2]
    return omega, np.array(psd_vals), Mean_output, Standard_dev_output
