import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average_filter(x_data, y_data, window_size):
    y_filtered = np.convolve(y_data, np.ones(window_size)/window_size, mode='valid')
    x_filtered = x_data[(window_size-1)//2 : -(window_size-1)//2]
    return x_filtered, y_filtered

def compute_mean(x_data, y_data, n_locations):
    indices = np.linspace(0, len(x_data)-1, n_locations, dtype=int)
    x_mean = x_data[indices]
    y_mean = np.mean(np.digitize(y_data, np.linspace(min(y_data), max(y_data), n_locations)))
    return x_mean, y_mean

my_path = os.getcwd()
my_data = 'ArrayData'
Settings = ['Zero']
# Scenarios = ['tinySquare1noRot', 'tinySquare1noRotLow', 'tinySquare1Rot']
# Scenarios = ['tinySquare2noRot', 'tinySquare2noRotLow', 'tinySquare2Rot']
Scenarios2 = ['Dataset 1', 'Dataset 2']

# Colours
Colours = plt.cm.viridis(np.linspace(0, 0, 2))
PlotLinalgError = 1


nBins = 25

fig, axs = plt.subplots(1, 2, figsize=(10, 4))


for settingNumber, setting in enumerate(Settings):
    for scenarioNumber, scenario in enumerate(Scenarios2):
        dx_plot_zero_stored = np.array([])
        dx_error_plot_zero_stored = np.array([])
        eta_error_plot_zero_stored = np.array([])

        dx_plot_true_stored = np.array([])
        dx_error_plot_true_stored = np.array([])
        eta_error_plot_true_stored = np.array([])
        for scenarios2 in range(3):
            if scenarios2 == 0:
                Scenarios = ['tinySquare1noRotLow', 'tinySquare1noRot']
            elif scenarios2 == 1:
                Scenarios = ['tinySquare2noRotLow', 'tinySquare2noRot']
            else:
                Scenarios = ['tinySquare3noRotLow', 'tinySquare3noRot']
            # for scenario in Scenarios:
            scenario = Scenarios[scenarioNumber]
            my_file = scenario + setting + '.npz'
            Data = np.load(os.path.join(my_path, my_data, my_file))
            dx_stored = Data['dx_stored']
            dx_est_stored = Data['dx_est_stored']
            eta_error_stored = Data['eta_error_stored']
            StoreLinAlgError = Data['StoreLinAlgError']

            TimeSteps = dx_stored.shape[2]

            cmap = plt.cm.viridis(np.linspace(0, 1, 1))
            color = cmap

            for timestep in range(TimeSteps):
                plot_linalgerror = StoreLinAlgError[:, timestep]
                if PlotLinalgError == 0:
                    plot_linalgerror = 0 * plot_linalgerror
                dx_error = (dx_stored[:, :, timestep] - dx_est_stored[:, :, timestep])
                dx_plot = np.sqrt(dx_stored[0, :, timestep]**2 + dx_stored[1, :, timestep]**2 + dx_stored[2, :, timestep]**2)
                dx_error_plot = np.sqrt((dx_stored[0, :, timestep] - dx_est_stored[0, :, timestep])**2 + 
                                            (dx_stored[1, :, timestep] - dx_est_stored[1, :, timestep])**2 + 
                                            (dx_stored[2, :, timestep] - dx_est_stored[2, :, timestep])**2)
                # dx_error = dx_error[plot_linalgerror == 0]
                dx_plot = dx_plot[plot_linalgerror == 0]
                dx_error_plot = dx_error_plot[plot_linalgerror == 0]
                # axs[row, col].plot(np.linspace(minplot, maxplot, 100), np.linspace(minplot, maxplot, 100), color='red')

                if setting == 'Zero':
                    dx_plot_zero_stored = np.append(dx_plot_zero_stored, dx_plot)
                    dx_error_plot_zero_stored = np.append(dx_error_plot_zero_stored, dx_error_plot)
                elif setting == 'True':
                    dx_plot_true_stored = np.append(dx_plot_true_stored, dx_plot)
                    dx_error_plot_true_stored = np.append(dx_error_plot_true_stored, dx_error_plot)
                # axs[0, col].scatter(dx_plot, dx_error_plot, c=[colour], alpha=0.5)
                # axs[row, col].set_ylim([0, 25])
                # axs[row, col].set_xlim([0, 100])

                eta_error = 180 / np.pi * eta_error_stored[:, :, timestep]
                # eta_plot = 1000 * np.sqrt(dx_stored[0, :, timestep]**2 + dx_stored[1, :, timestep]**2 + dx_stored[2, :, timestep]**2)
                eta_error_plot = 180 / np.pi * np.sqrt( eta_error_stored[0, :, timestep]**2 + 
                                                        eta_error_stored[1, :, timestep]**2 + 
                                                        eta_error_stored[2, :, timestep]**2)
                # eta_error = eta_error[plot_linalgerror == 0]
                # eta_plot = eta_plot[plot_linalgerror == 0]  
                eta_error_plot = eta_error_plot[plot_linalgerror == 0]  
                # axs[1, col].set_ylim([-1, 180])
                # eta_plot_stored = np.append(eta_plot_stored, eta_plot)
                if setting == 'Zero':
                    eta_error_plot_zero_stored = np.append(eta_error_plot_zero_stored, eta_error_plot)
                elif setting == 'True':
                    eta_error_plot_true_stored = np.append(eta_error_plot_true_stored, eta_error_plot)
                # axs[1, col].scatter(eta_plot, eta_error_plot, c=[colour], alpha=0.5)
                # axs[1, col].plot(np.linspace(minplot2, maxplot2, 100), np.linspace(minplot2, maxplot2, 100), color='green')

        # for setting in Settings:
        if setting == 'Zero':
            valid_indices = ~np.isnan(dx_error_plot_zero_stored)# & ~np.isnan(dx_plot_zero_stored) & ~np.isnan(eta_error_plot_zero_stored)
            dx_plot_zero_stored = dx_plot_zero_stored[valid_indices]
            dx_error_plot_zero_stored = dx_error_plot_zero_stored[valid_indices]
            eta_error_plot_zero_stored = eta_error_plot_zero_stored[valid_indices]
            threshold = np.percentile(dx_error_plot_zero_stored, 100)
            indices = np.where(dx_error_plot_zero_stored <= threshold)[0]
            dx_plot_zero_stored = dx_plot_zero_stored[indices]
            dx_error_plot_zero_stored = dx_error_plot_zero_stored[indices]
            eta_error_plot_zero_stored = eta_error_plot_zero_stored[indices]
            sorted_indices = np.argsort(dx_plot_zero_stored)
            dx_plot_zero_stored = dx_plot_zero_stored[sorted_indices]
            dx_error_plot_zero_stored = dx_error_plot_zero_stored[sorted_indices]
            eta_error_plot_zero_stored = eta_error_plot_zero_stored[sorted_indices]

        elif setting == 'True':
            valid_indices = ~np.isnan(dx_error_plot_true_stored) #& ~np.isnan(dx_plot_true_stored) & ~np.isnan(eta_error_plot_true_stored)
            dx_plot_true_stored = dx_plot_true_stored[valid_indices]
            dx_error_plot_true_stored = dx_error_plot_true_stored[valid_indices]
            eta_error_plot_true_stored = eta_error_plot_true_stored[valid_indices]   
            threshold = np.percentile(dx_error_plot_true_stored, 100)
            indices = np.where(dx_error_plot_true_stored <= threshold)[0]
            dx_plot_true_stored = dx_plot_true_stored[indices]
            dx_error_plot_true_stored = dx_error_plot_true_stored[indices]
            eta_error_plot_true_stored = eta_error_plot_true_stored[indices]
            sorted_indices = np.argsort(dx_plot_true_stored)
            dx_plot_true_stored = dx_plot_true_stored[sorted_indices]
            dx_error_plot_true_stored = dx_error_plot_true_stored[sorted_indices]
            eta_error_plot_true_stored = eta_error_plot_true_stored[sorted_indices]

        if setting == 'True':
            # Bin data and compute stats for dx
            bin_centers_dx, mean_dx = moving_average_filter(dx_plot_true_stored, dx_error_plot_true_stored, nBins)
            # Bin data and compute stats for eta
            bin_centers_eta, mean_eta = moving_average_filter(dx_plot_true_stored, eta_error_plot_true_stored, nBins)
            
            # axs[0].scatter(dx_plot_true_stored/0.1, dx_error_plot_true_stored, c=Colours[scenarioNumber])
            # axs[1].scatter(dx_plot_true_stored/0.1, eta_error_plot_true_stored, c=Colours[scenarioNumber])
            # Plot dx error
            # axs[0].plot(bin_centers_dx/0.1, mean_dx, c=Colours[settingNumber + 2*scenarioNumber])
            # # Plot eta error
            # valid_indices_eta = ~np.isnan(mean_eta)
            # mean_eta = mean_eta[valid_indices_eta]
            # bin_centers_eta = bin_centers_eta[valid_indices_eta]
            # axs[1].plot(bin_centers_eta/0.1, mean_eta, c=Colours[settingNumber + 2*scenarioNumber])
            # Plot dx error
            axs[0].scatter(1000*dx_plot_true_stored, 1000*dx_error_plot_true_stored, c=Colours[scenarioNumber], s=1)
            # Plot eta error
            axs[1].scatter(1000*dx_plot_true_stored, eta_error_plot_true_stored, c=Colours[scenarioNumber], s=1)

        if setting == 'Zero':
            # Bin data and compute stats for dx
            bin_centers_dx, mean_dx = moving_average_filter(dx_plot_zero_stored, dx_error_plot_zero_stored, nBins)
            # Bin data and compute stats for eta
            bin_centers_eta, mean_eta = moving_average_filter(dx_plot_zero_stored, eta_error_plot_zero_stored, nBins)
            
            # Plot dx error
            # axs[0].plot(bin_centers_dx/0.1, mean_dx, c=Colours[scenarioNumber])
            # axs[0].plot(bin_centers_dx/0.1, dx_error_plot_zero_stored, c=Colours[settingNumber + 2*scenarioNumber])
            # Plot eta error
            # valid_indices_eta = ~np.isnan(mean_eta)
            # mean_eta = mean_eta[valid_indices_eta]
            # bin_centers_eta = bin_centers_eta[valid_indices_eta]
            # axs[1].plot(bin_centers_eta/0.1, mean_eta, c=Colours[scenarioNumber])
            # axs[1].plot(bin_centers_dx/0.1, mean_dx, c=Colours[settingNumber + 2*scenarioNumber])

            # Plot dx error
            axs[0].scatter(1000*dx_plot_zero_stored, 1000*dx_error_plot_zero_stored, c=Colours[scenarioNumber], s=1)
            # Plot eta error
            axs[1].scatter(1000*dx_plot_zero_stored, eta_error_plot_zero_stored, c=Colours[scenarioNumber], s=1)        
# Load in data
my_path = os.getcwd()
my_data = 'Data'
data_x = np.load(os.path.join(my_path, my_data, 'MC_step_distance_x.npz'))
#data_y = np.load(os.path.join(my_path, my_data, 'MC_step_distance_y.npz'))
data_z = np.load(os.path.join(my_path, my_data, 'MC_step_distance_z.npz'))

cov_stored = data_x['cov_stored']

Xrange = 1000*data_x['Xrange']#/np.sqrt(data_x['theta'][0])


# Create figure

# mean_values = 1000*np.mean(np.sqrt(cov_stored[0, 0, :, :]+ cov_stored[1, 1, :, :] + cov_stored[2, 2, :, :])/3, axis=0)
# axs[0].plot(Xrange, 4*mean_values, c= 'black', linewidth=3, linestyle = '--')

# mean_values = 180/np.pi*np.mean(np.sqrt(cov_stored[3, 3, :, :] + cov_stored[4, 4, :, :] +cov_stored[5, 5, :, :])/3, axis=0)
# axs[1].plot(Xrange, 4*mean_values, c='black', linewidth=3, linestyle = '--')


for col in range(len(Scenarios2)):
    axs[col].set_xscale('log')
    axs[col].set_yscale('log')
    axs[col].grid(True)
    axs[col].grid(True, which='minor', linestyle=':', linewidth='0.5')
    axs[col].set_xlabel(r"$||\Delta \mathbf{p}||_2$ [mm]", fontsize = 16, fontweight='bold')

axs[0].set_ylabel(r"$||\mathbf{\epsilon} ||_2$ [mm]", fontsize = 16, fontweight='bold')
axs[1].set_ylabel(r"$||\mathbf{\eta} ||_2$ [°]", fontsize = 16, fontweight='bold')

# axs[1, col].set_title(f'{Scenarios[row]}')
for ax in axs:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
axs[0].set_title('Positional error', fontsize = 16, fontweight='bold')
axs[1].set_title('Rotational error', fontsize = 16, fontweight='bold')    
axs[0].set_ylim([0.3, 1e4])
axs[1].set_ylim([0.05, 1e2])
axs[0].set_xlim([0.3, 1e3])
axs[1].set_xlim([0.3, 1e3])

# handles = [plt.Line2D([], [], color=colour, marker='o', linestyle='None', markersize=8) for colour in Colours]
# handles.append(plt.Line2D([], [], color='black', linewidth=3, linestyle='--'))
# labels = [f'{scenario} ' for scenario in ScenariosNames]
# # labels.append(f'Standard deviation')
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.525, 1.0750), ncol=4, fontsize = 16)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle


plt.show()
fig.set_dpi(10)
my_path = os.getcwd()
my_figures = 'Figures'
my_file = 'ExperimentDistancePlot.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')