import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Load in data
my_path = os.getcwd()
my_data = 'Data'

fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for o in range(2):
    if o == 0:
        data = np.load(os.path.join(my_path, my_data, 'MC_errors_x.npz'))
        # data_y = np.load(os.path.join(my_path, my_data, 'MC_errors_y.npz'))
    else:
        data = np.load(os.path.join(my_path, my_data, 'MC_errors_z.npz'))

    XestStored = data['XestStored']
    Nmc = data['Nmc']
    etaStored = data['etaStored']
    steps = data['steps']
    stepSize = data['stepSize']
    # Xcenter = data['Xcenter']
    Xcenter = np.zeros((3, steps+1))
    if o == 0:
        Xcenter[0, :] = np.linspace(0, steps*stepSize, steps+1)
    else: 
        Xcenter[2, :] = np.linspace(0, steps*stepSize, steps+1)

    sensorErrorLevels = [0, 1, 2, 3]
    sensorErrorLabels = ['None', 'Low', 'Med.', 'High']
    sensorErrorSettings = ['Measurement bias', 'Misalign. Rot.', 'Misalign. Pos.']

    Colours = plt.cm.viridis(np.linspace(0, 1, len(sensorErrorLevels)))

    for j, sensorErrorSetting in enumerate(sensorErrorSettings):
        for i, sensorErrorLabel in reversed(list(enumerate(sensorErrorLabels))):
            XestStoredPlot = 0
            etaStoredPlot = 0
            for l in range(Nmc):
                XestStoredPlot += 1000/Nmc*np.sqrt((XestStored[0, :, i, j, l] - Xcenter[0, :])**2 + (XestStored[1, :, i, j, l] - Xcenter[1, :])**2 + (XestStored[2, :, i, j, l] - Xcenter[2, :])**2)
                etaStoredPlot += 180/np.pi/Nmc*np.sqrt(etaStored[0, :, i, j, l]**2 + etaStored[1, :, i, j, l]**2 + etaStored[2, :, i, j, l]**2)
            if i == 2:
                print(np.array([[XestStoredPlot[-1], etaStoredPlot[-1]]]))
            axes[0, j + o*3].fill_between(1000*np.linspace(0, (steps+1)*stepSize, steps+1), XestStoredPlot, 0, label=sensorErrorLabel, color=Colours[i])
            axes[1, j + o*3].fill_between(1000*np.linspace(0, (steps+1)*stepSize, steps+1), etaStoredPlot, 0, label=sensorErrorLabel, color=Colours[i])
        if o == 0:
            axes[0, j + o*3].set_xlabel('Displacement in $x_1$ [mm]', fontsize = 12)
            axes[1, j + o*3].set_xlabel('Displacement in $x_1$ [mm]', fontsize = 12)
        else:
            axes[0, j + o*3].set_xlabel('Displacement in $x_3$ [mm]', fontsize = 12)
            axes[1, j + o*3].set_xlabel('Displacement in $x_3$ [mm]', fontsize = 12)
        
        axes[0, j + o*3].yaxis.tick_right()
        axes[0, j + o*3].tick_params(axis='y', which='both', direction='in', labelright=True, labelleft=False)
        axes[1, j + o*3].yaxis.tick_right()
        axes[1, j + o*3].tick_params(axis='y', which='both', direction='in', labelright=True, labelleft=False)
        axes[0, j + o*3].set_ylabel(r'Avg. $\|| \epsilon \||_2$ drift [mm]', fontsize = 12)
        axes[1, j + o*3].set_ylabel(r'Avg. $\|| \eta \||_2$ drift [°]', fontsize = 12)
        axes[0, j + o*3].set_title(sensorErrorSetting, fontsize = 16)
# Creating artificial handles and labels for the legend
handles = [Patch(color=Colours[i]) for i in range(len(sensorErrorLabels))]
labels = sensorErrorLabels

# Place the legend on top of the figure
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=16)


plt.tight_layout()
plt.show()

my_path = os.getcwd()
my_figures = 'Figures'
my_file = 'SensorErrorsSum.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')





# fig, axes = plt.subplots(6, 6, figsize=(10, 10))
# for j, sensorErrorSetting in enumerate(sensorErrorSettings):
#     for i, sensorErrorLabel in reversed(list(enumerate(sensorErrorLabels))):
#         for m in range(3):
#             XestStoredPlot = 0
#             etaStoredPlot = 0
#             for l in range(Nmc):
#                 XestStoredPlot += np.abs(XestStored[m, :, i, j, l] - Xcenter[m, :])/Nmc
#                 etaStoredPlot += 180/np.pi*np.abs(etaStored[m, :, i, j, l])/Nmc

#             axes[0+m, j].fill_between(np.linspace(0, (steps+1)*stepSize, steps+1), XestStoredPlot, 0, label=sensorErrorLabel, color=Colours[i])
#             axes[3+m, j].fill_between(np.linspace(0, (steps+1)*stepSize, steps+1), etaStoredPlot, 0, label=sensorErrorLabel, color=Colours[i])
#             axes[0+m, j].set_xlabel('Displacement in $x_1$ [m]')
#             axes[3+m, j].set_xlabel('Displacement in $x_1$ [m]')
#             axes[3+m, j].set_ylabel('$x_1$ [m]')
#         axes[0, j].set_ylabel('Mean error $\epsilon_1$ [m]')
#         axes[1, j].set_ylabel('Mean error $\epsilon_2$ [m]')
#         axes[2, j].set_ylabel('Mean error $\epsilon_3$ [m]')
#         axes[3, j].set_ylabel('Mean error $\eta_1$ [°]')
#         axes[4, j].set_ylabel('Mean error $\eta_2$ [°]')
#         axes[5, j].set_ylabel('Mean error $\eta_3$ [°]')
#     axes[0, j].set_title(sensorErrorSetting)
# # Creating artificial handles and labels for the legend
# handles = [Patch(color=Colours[i]) for i in range(len(sensorErrorLabels))]
# labels = sensorErrorLabels

# # Place the legend on top of the figure
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(sensorErrorLabels))

# plt.tight_layout()
# plt.show()

