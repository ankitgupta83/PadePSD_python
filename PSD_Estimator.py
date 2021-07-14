from scipy.fft import fft, fftfreq
import numpy as np


def PSD_Estimator_Numerical(sampling_times, state_values):
    delta_t = (sampling_times[1] - sampling_times[0])
    num_trajectories = np.shape(state_values)[0]
    num_sample_points = np.shape(state_values)[1]
    psd_vals = []
    mean_zero_state_values = np.zeros(np.shape(state_values))
    for i in range(num_trajectories):
        # subtract signal mean
        mean_zero_state_values[i, :] = state_values[i, :] - np.mean(state_values[i, :])
        # compute FFT and transform to PSD
        psd = (delta_t / num_sample_points) * np.abs(fft(mean_zero_state_values[i, :]))[:num_sample_points // 2] ** 2
        psd_vals.append(psd)
    omega = 2 * np.pi * fftfreq(num_sample_points, delta_t)[:num_sample_points // 2]
    return omega, np.array(psd_vals)
