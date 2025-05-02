import numpy as np
from matplotlib import pyplot as plt

import linAlg as linAlg
import helper as helper
import GP as GP 

import os

def reshapematrix(A):
    B = np.zeros((A.shape[0], 1))
    for i in range(A.shape[2]):
        B = np.hstack((B, A[:, :, i]))
    return B[:, 1:]

my_path = os.getcwd()
my_data = 'ArrayData'

Settings = ['Zero']
Scenarios = ['tinySquare1noRotLow',  'tinySquare2noRotLow', 'tinySquare3noRotLow', 'tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']
ScenarioNames = ['Dataset 1',  'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5', 'Dataset 6']
# Scenarios = ['tinySquare1noRot', 'tinySquare1noRotLow', 'tinySquare1Rot']
# Scenarios = ['bigsquare1', 'bigsquare1normalheight', 'bigsquare1tilted']
magNames = ['full', 'Even', 'Odd']
PlotLinalgError = 1

# Second 3D scatter plot
fig, axs = plt.subplots(5, len(Scenarios), figsize=(len(Scenarios)*4, 20))  # Create a figure with len(Scenarios) rows and 4 columns
cmap = plt.cm.viridis(np.linspace(0, 1, 4))
# Plot data for the first set of subplots (3 rows, 2 columns)
for setting in Settings:
    for col, scenario in enumerate(Scenarios):
        for indx, magName in enumerate(magNames):

            my_file = scenario + setting + magName + '.npz'
            Data = np.load(os.path.join(my_path, my_data, 'WalkPath' + my_file))
            theta = np.array([0.15**2, 5**2, 0.0012, 15**2])

            dx_stored = Data['dx_stored']
            cov_stored = Data['cov_stored']
            dx_est_stored = Data['dx_est_stored']
            R_true_stored = Data['R_true_stored']
            R_est_stored = Data['R_est_stored']
            eta_error_stored = Data['eta_error_stored']
            StoreLinAlgError = Data['StoreLinAlgError']
            dx_stored_n = dx_stored*0
            dx_est_stored_n = dx_est_stored*0
            eta_est_stored_n = dx_est_stored*0
            eta_est2_stored_n = dx_est_stored*0
            eta_true_stored_n = dx_est_stored*0
            dx_est_stored_n2 = dx_est_stored*0
            TimeSteps = dx_stored.shape[2]
            Steps = dx_stored.shape[1]
            Rtrue = np.eye(3)
            Rest = np.eye(3)

            dx_est_stored[np.isnan(dx_est_stored)] = 0
            #dx_stored[np.isnan(dx_stored)] = 0
            for i1 in range(R_est_stored.shape[2]):
                for j2 in range(R_est_stored.shape[3]):
                    if np.isnan(R_est_stored[:, :, i1, j2]).any():
                        R_est_stored[:, :, i1, j2] = np.eye(3) #R_true_stored[:, :, i1, j2]

            for j in range(TimeSteps):
                for i in range(Steps):
                    Rest = Rest @ R_est_stored[:, :, i, j]
                    Rtrue = Rtrue @ R_true_stored[:, :, i, j]
                    
                    dx_est_stored_n[:, i:i+1, j] = Rest @ dx_est_stored[:, i:i+1, j]
                    dx_stored_n[:, i:i+1, j] = Rtrue @ dx_stored[:, i:i+1, j]
                    eta_est_stored_n[:, i:i+1, j] = linAlg.R2eta(Rtrue.T @ Rest)
                    eta_est2_stored_n[:, i:i+1, j] = linAlg.R2eta(Rest)
                    eta_true_stored_n[:, i:i+1, j] = linAlg.R2eta(Rtrue)

            dx_est_stored_n[np.isnan(dx_est_stored_n)] = 0
            dx_est_sum = 1000*reshapematrix(dx_est_stored)
            dx_sum = 1000*reshapematrix(dx_stored)
            dx_sum2 = dx_sum*0
            dx_est_sum_n = 1000*reshapematrix(dx_est_stored_n)
            dx_sum_n = 1000*reshapematrix(dx_stored_n)
            eta_n = 180/np.pi*reshapematrix(eta_est_stored_n)
            eta2_n = 180/np.pi*reshapematrix(eta_est2_stored_n)
            eta_true_n = 180/np.pi*reshapematrix(eta_true_stored_n)
        
            for i in range(3):
                dx_est_sum[i, :] = np.cumsum(dx_est_sum[i, :])
                dx_sum2[i, :] = np.cumsum(np.abs(dx_sum[i, :]))

                dx_sum[i, :] = np.cumsum(dx_sum[i, :])
                dx_est_sum_n[i,:] = np.cumsum(dx_est_sum_n[i, :])
                dx_sum_n[i, :] = np.cumsum(dx_sum_n[i, :])
            dx_est_sum_n -= np.mean(dx_sum_n, 1).reshape(3, 1)
            dx_sum_n -= np.mean(dx_sum_n, 1).reshape(3, 1)
            # Assuming your data has a third dimension, for example, dx_sum_n[2, :] and dx_est_sum_n[2, :]
            if indx == 0:
                axs[0, col].plot(dx_sum_n[0, :], dx_sum_n[1, :], color = cmap[0], label='Ground truth position or orientation')
            axs[0, col].plot(dx_est_sum_n[0, :], dx_est_sum_n[1, :], color = cmap[1+indx], label='Estimated position or orientation')
            if col == 0:
                axs[0, col].set_ylabel(r'$p_2$ [mm]', fontsize = 16, fontweight='bold')
                axs[1, col].set_ylabel(r'$p_3$ [mm]', fontsize = 16, fontweight='bold')
                axs[2, col].set_ylabel(r'Roll [$\degree$]', fontsize = 16, fontweight='bold')
                axs[3, col].set_ylabel(r'Pitch [$\degree$]', fontsize = 16, fontweight='bold')
                axs[4, col].set_ylabel(r'Yaw [$\degree$]', fontsize = 16, fontweight='bold')
            axs[0, col].set_xlabel(r'$p_1$ [mm]', fontsize = 16, fontweight='bold')
            for allCols in range(1, 5):
                axs[allCols, col].set_xlabel('Time [s]', fontsize = 16, fontweight='bold')

            axs[0, col].set_title(ScenarioNames[col], fontsize = 16, fontweight='bold')
            # Assuming your data has a third dimension, for example, dx_sum_n[2, :] and dx_est_sum_n[2, :]
            if indx == 0:
                axs[1, col].plot(np.linspace(0, dx_sum_n.shape[1], dx_sum_n.shape[1])/10, dx_sum_n[2, :], color = cmap[0])
            axs[1, col].plot(np.linspace(0, dx_est_sum_n.shape[1], dx_est_sum_n.shape[1])/10, dx_est_sum_n[2, :], color = cmap[1+indx])
            # axs[2, col].plot(np.linspace(0, eta_n.shape[1], eta_n.shape[1]), np.sqrt(eta_n[0, :]**2 + eta_n[1, :]**2 + eta_n[2, :]**2), label='Estimate')
            axs[2, col].plot(np.linspace(0, eta2_n.shape[1], eta2_n.shape[1])/10, eta2_n[0, :], color = cmap[1+indx])#, label='$\eta_1 Est$')
            axs[3, col].plot(np.linspace(0, eta2_n.shape[1], eta2_n.shape[1])/10, eta2_n[1, :], color = cmap[1+indx])#, label='$\eta_2 Est$')
            axs[4, col].plot(np.linspace(0, eta2_n.shape[1], eta2_n.shape[1])/10, eta2_n[2, :], color = cmap[1+indx])#, label='$\eta_3 Est$')

            axs[2, col].plot(np.linspace(0, eta_true_n.shape[1], eta_true_n.shape[1])/10, eta_true_n[0, :], color = cmap[0])
            axs[3, col].plot(np.linspace(0, eta_true_n.shape[1], eta_true_n.shape[1])/10, eta_true_n[1, :], color = cmap[0])
            axs[4, col].plot(np.linspace(0, eta_true_n.shape[1], eta_true_n.shape[1])/10, eta_true_n[2, :], color = cmap[0])
            if col == 0:
                print('pos error', dx_sum_n[:, -1] - dx_est_sum_n[:, -1])
                print('rot error', eta_true_n[:, -1] - eta2_n[:, -1])
                print('pos final', dx_sum2[:, -1])


    # Create a single legend on top of the figure
    handles1, labels1 = axs[0, 0].get_legend_handles_labels()
    # handles2, labels2 = axs[2, 0].get_legend_handles_labels()
    handles = handles1# + handles2
    labels = labels1# + labels2
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, .925), fontsize = 16)
    plt.show()
    my_path = os.getcwd()
    my_figures = 'Figures'
    my_file = 'WalkPath2D' + 'HalfArray' + '.pdf'
    fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')
