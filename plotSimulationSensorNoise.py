import numpy as np
import matplotlib.pyplot as plt
import os

# Load in data
my_path = os.getcwd()
my_data = 'Data'
data_x = np.load(os.path.join(my_path, my_data, 'MC_step_noise_x.npz'))
# data_y = np.load(os.path.join(my_path, my_data, 'MC_step_noise_y.npz'))
data_z = np.load(os.path.join(my_path, my_data, 'MC_step_noise_z.npz'))

cov_stored_x = data_x['cov_stored']
cov_stored_single_fused_x = data_x['cov_stored_single_fused']
cov_stored_single_avg_x = data_x['cov_stored_single_mean']
# cov_stored_y = data_y['cov_stored']
# cov_stored_single_fused_y = data_y['cov_stored_single_fused']
# cov_stored_single_avg_y = data_y['cov_stored_single_mean']
cov_stored_z = data_z['cov_stored']
cov_stored_single_fused_z = data_z['cov_stored_single_fused']
cov_stored_single_avg_z = data_z['cov_stored_single_mean']
Xrange = data_x['Xrange']
# theta = data_x['theta']

# colours = ['royalblue', 'forestgreen', 'darkred', 'cornflowerblue', 'lawngreen', 'hotpink']
linestyles = ['-', '-.', ':']
Colours = plt.cm.viridis(np.linspace(0, 1, 3))

# Create figure
fig = plt.figure(figsize=(10, 4))

for i in range(2):
    for j in range(1):
        ax = fig.add_subplot(1, 2, i+1)
        plt.grid()
        labels = [r'Case 1', r'Case 2', r'Case 3']

        if i == 0:
            labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$']
            if j == 0: plt.ylabel(r'Average pos. var. $\mathbf{[mm^2]}$', fontsize = 18, fontweight='bold')
        else:
            labels = [r'$\eta_1$', r'$\eta_2$', r'$\eta_3$']
            if j == 0: plt.ylabel(r'Average rot. var. $\mathbf{[\circ^2]}$', fontsize = 18, fontweight='bold')
        if i == 0: plt.xlabel(r' ', fontsize = 18, fontweight='bold')

        if j == 0:
            cov_stored = cov_stored_x
            cov_stored_single_fused = cov_stored_single_fused_x
            cov_stored_single_avg = cov_stored_single_avg_x
        # elif j == 1:
        #     cov_stored = cov_stored_y 
        #     cov_stored_single_fused = cov_stored_single_fused_y
        #     cov_stored_single_avg = cov_stored_single_avg_y
        else: 
            cov_stored = cov_stored_z
            cov_stored_single_fused = cov_stored_single_fused_z
            cov_stored_single_avg = cov_stored_single_avg_z

        mean_values = 0
        mean_values_single_fused = 0
        mean_values_single_avg = 0
        for k in range(3):

            mean_values = np.mean(cov_stored[3*i+k, 3*i+k, :, :], axis=0)
            mean_values_single_fused = np.mean(cov_stored_single_fused[3*i+k, 3*i+k, :, :], axis=0)
            mean_values_single_avg = np.mean(cov_stored_single_avg[3*i+k, 3*i+k, :, :], axis=0)

            if i == 0:
                plt.plot(Xrange, mean_values*(1000)**2, linestyle=linestyles[0], label=labels[k], color = Colours[k], linewidth=4)
                plt.plot(Xrange, mean_values_single_fused*(1000)**2, linestyle=linestyles[1], color = Colours[k], linewidth=4)
                plt.plot(Xrange, mean_values_single_avg*(1000)**2, linestyle=linestyles[2], color = Colours[k], linewidth=4)
            else:
                plt.plot(Xrange, mean_values*(180/np.pi)**2, linestyle=linestyles[0], label=labels[k], color = Colours[k], linewidth=4)

        # Share legend for every row, only in the first column
        if j == 0:
            plt.legend(ncol=3, loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 1.25))
        
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(which='major', linewidth=1, color='lightgrey')
        # Set x-axis limit
        plt.xlim(Xrange.min(), Xrange.max())
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # # Share titles for every column
        # if i == 0:
        #     if j == 0:
        #         ax.set_title(r'step in $x_1$ direction (in-plane)')
        #     elif j == 1:
        #         ax.set_title(r'step in $x_3$ direction (out-of-plane)')
        #     #elif j == 2:
        #         #ax.set_title(r'step in $x_3$ direction')

# Share xlabel for every column
fig.text(0.5, 0.025, r'SNR $\sigma_{f}/\sigma_{y}$ [$-$]', ha='center', fontsize = 16, fontweight='bold')

plt.tight_layout()
plt.show()

# Save the figure
my_figures = 'Figures'
my_file = 'SensorNoise.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')

