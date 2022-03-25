import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from sympy.abc import x

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    #"font.weight": "bold",  # bold fonts
    "text.latex.preamble": [r'\usepackage{sfmath} \boldmath'],
    "xtick.labelsize": 15,  # large tick labels
    'font.size': 15,
    'figure.figsize': [6.4, 4.8]}  # default: 6.4 and 4.8 620/72, 300/72
)
legend_font = font_manager.FontProperties(family='Arial',
                                          weight='bold',
                                          style='normal', size=16)


def plotFunctionTrajectory(xvalues, yvalues, num_trajectories, trajectory_length, results_folder_path, save_pdf=False):
    sns.set_style("ticks")

    sns.set_context("paper", font_scale=1.4)
    time_step = xvalues[1] - xvalues[0]
    num_time_points = int(trajectory_length / time_step) + 1
    for i in range(num_trajectories):
        ax = sns.lineplot(x=np.linspace(0, trajectory_length, num_time_points - 1), y=yvalues[i, -num_time_points:-1],
                          linewidth=1.5, label="Cell " + str(i + 1))
    # ax.set(ylabel='Output $X_n(t)$')
    # ax.set(xlabel="time t") b
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', linewidth=1.0)
    ax.grid(b=True, which='minor', linewidth=0.5)
    leg = plt.legend(prop=legend_font)
    leg.get_frame().set_edgecolor('black')
    if save_pdf:
        plt.savefig(results_folder_path + "Trajectory_plot.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


def generate_psd_comparison_plot(psd_dict, time_unit, results_folder_path, omega_limit=np.pi, save_pdf=False):
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)
    plt.figure("PSD Comparison Plot")
    ax = sns.lineplot(x=psd_dict['DFT'][0], y=psd_dict['DFT'][1], color='black', linewidth=1, label="DFT")
    plt.fill_between(psd_dict['DFT'][0], psd_dict['DFT'][1] - psd_dict['DFT'][2],
                     psd_dict['DFT'][1] + psd_dict['DFT'][2], alpha=0.5, color='grey')
    xlabel = "frequency $\omega$ [rad/" + time_unit + ']'
    ax = sns.lineplot(x=psd_dict['Pade'][0], y=psd_dict['Pade'][1], color='red', linewidth=2, label="Pad\\'{e} PSD")
    if psd_dict['Analytical'][0] is not None:
        ax = sns.lineplot(x=psd_dict['Analytical'][0], y=psd_dict['Analytical'][1], color='green',
                          linestyle=(0, (5, 5)),
                          linewidth=2, label="Analytical")
        # ax.lines[1].set_linestyle("--")
    ax.set_xlim([0, omega_limit])
    ax.set_ylim([0, np.max(psd_dict['DFT'][1] + 1.2 * psd_dict['DFT'][2])])
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', linewidth=1.0)
    ax.grid(b=True, which='minor', linewidth=0.5)
    # ax.set(ylabel='PSD')
    # ax.set(xlabel=xlabel)
    leg = plt.legend(prop=legend_font)
    leg.get_frame().set_edgecolor('black')
    if save_pdf:
        plt.savefig(results_folder_path + "PSD_comparison_plot.pdf", bbox_inches='tight', transparent="False", pad_inches=0)


def generate_test_s_values_comparison(test_s_values, test_s_values_mean, test_s_values_std, G_p, results_folder_path,
                                      save_pdf=False):
    sns.set_style("whitegrid")
    plt.figure("G(s) Validation plot")
    ax = plt.plot(test_s_values, test_s_values_mean, 'rD', label="Test values G(s)")
    plt.errorbar(test_s_values, test_s_values_mean, yerr=test_s_values_std, ls='none', color='r')
    # plt.fill_between(test_s_values, test_s_values_mean - test_s_values_std,
    #                  test_s_values_mean + test_s_values_std, alpha=0.5, color='grey')
    s_vals = np.linspace(test_s_values[0] - 1, test_s_values[-1] + 1)
    G_s_vals = s_vals * 0
    for i in np.arange(np.shape(s_vals)[0]):
        G_s_vals[i] = G_p.evalf(subs={x: s_vals[i]})

    ax = sns.lineplot(x=s_vals, y=G_s_vals, color='blue',
                      linewidth=1.5, label="Estimated G(s)")
    ax.legend()
    ax.set_xlim([max(test_s_values[0] - 1, 0), test_s_values[-1] + 1])
    ax.set_ylim([max(test_s_values[0] - 1, 0), 1.5 * max(test_s_values_mean)])
    ax.set(ylabel="G(s)")
    ax.set(xlabel="s")
    if save_pdf:
        plt.savefig(results_folder_path + "G_validation_plot.pdf", bbox_inches='tight', transparent="False", pad_inches=0)