import numpy as np
import matplotlib.pyplot as plt
import os

my_path = os.getcwd()
my_data = 'Data'
my_file = 'MC_configuration_3d.npz'
data = np.load(os.path.join(my_path, my_data, my_file))
cov_stored = data['cov_stored']

Nsize = cov_stored.shape[2]
Ndirection = cov_stored.shape[5] 
sensorSeparation = np.logspace(-4, 2, Nsize, base=10)/4
legendLabels = [r'Cube', r'Square', r'Line']
legendLabels2 = [r'$\epsilon_1, \eta_1$', r'$\epsilon_2, \eta_2$', r'$\epsilon_3, \eta_3$']
Colours = plt.cm.viridis(np.linspace(1, 0, 3))
lineStyles = ['-', '--', ':']
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
lineMarkers = ['s', '^', 'o']

''' Plot for the first set of covariances '''
for i, legendLabel in enumerate(legendLabels):
    for j in range(Ndirection):
        for k in range(3):
            for l in range(2):
                if l == 0:
                    axs[l, j].plot(sensorSeparation, (1000)**2*np.mean(cov_stored[l*3+k, l*3+k, :, i, :, j], 0), c=Colours[i], linestyle=lineStyles[k], label=legendLabel, linewidth=4)
                    axs[l, j].set_ylim(1e-2, 1e5)
                else:
                    axs[l, j].plot(sensorSeparation, (180/np.pi)**2*np.mean(cov_stored[l*3+k, l*3+k, :, i, :, j], 0), c=Colours[i], linestyle=lineStyles[k], label=legendLabel, linewidth=4)
                    axs[l, j].set_ylim(1e-5, 1e4)

                axs[l, j].set_yscale('log')
                axs[l, j].set_xscale('log')
                axs[l, j].grid(True, which='major', linestyle='-', linewidth='1', color='lightgrey')
                axs[l, j].set_xlim(0.05, 1.5)
                axs[l, j].tick_params(axis='x', labelsize=14)
                axs[l, j].tick_params(axis='y', labelsize=14)
axs[0, 0].set_ylabel(r'Average pos. var. $\mathbf{[mm^2]}$', fontsize = 18, fontweight='bold')
axs[1, 0].set_ylabel(r'Average rot. var. $\mathbf{[\circ^2]}$', fontsize = 18, fontweight='bold')
axs[0, 0].set_title(r'$\Delta p_1$ step (in-plane)', fontsize = 18, fontweight='bold')
axs[0, 1].set_title(r'$\Delta p_3$ step (out-of-plane)', fontsize = 18, fontweight='bold')
axs[1, 0].set_xlabel(r' ', fontsize = 18, fontweight='bold')
axs[1, 1].set_xlabel(r' ', fontsize = 18, fontweight='bold')

''' Create a custom legend '''
lines = [plt.Line2D([0], [0], color=Colours[i], linestyle=lineStyles[k]) for i in range(3) for k in range(3)]
legend_labels = [f"{legendLabels[i]} {legendLabels2[j]}" for i in range(3) for j in range(3)]
fig.legend(lines, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=16)

fig.text(0.5, 0.025, r'Sensor separation $\alpha/l$ [$-$]', ha='center', fontsize = 16, fontweight='bold')

plt.tight_layout()
plt.show()

''' Save the figure '''
my_figures = 'Figures'
my_file = 'SensorConfiguration.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')
