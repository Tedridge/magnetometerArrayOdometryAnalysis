import numpy as np
import os

import linAlg as linAlg
import helper as helper
import GP as GP 

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import gridspec 

def reshapematrix(A):
    B = np.zeros((A.shape[0], 1))
    for i in range(A.shape[2]):
        B = np.hstack((B, A[:, :, i]))
    return B[:, 1:]


def makeIntoCountour(posPred, fPred, Nplot = 50):
    startPred = np.min(np.min(posPred))
    endPred = np.max(np.max(posPred))
    P0 = np.linspace(startPred, endPred, Nplot)
    P1 = np.linspace(startPred, endPred, Nplot)
    P0plot, P1plot = np.meshgrid(P0, P1)
    fPredNorm = np.sqrt(fPred[0, :] ** 2 + fPred[1, :] ** 2 + fPred[2, :] ** 2)
    fPlotNorm = griddata((posPred[0, :], posPred[1, :]), fPredNorm, (P0plot, P1plot), method="cubic")
    return P0plot, P1plot, fPlotNorm

theta = np.array([0.15**2, 5**2, 0.05, 15**2])
modelParameters = {"theta": theta}


x = np.linspace(-0.3, 0.3, num=50)  # Adjust 'num' for desired resolution
y = np.linspace(-0.3, 0.3, num=50)
z = np.array([0])  # Single z value

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

Xpred = (np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])).T



my_path = os.getcwd()
my_data = 'ArrayData'

Settings = ['Zero']
Scenarios = ['tinySquare1noRotLow',  'tinySquare2noRotLow', 'tinySquare3noRotLow', 'tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']
ScenarioNames = ['Dataset 1',  'Dataset 2', 'Dataset 3', 'Dataset 4', 'Dataset 5', 'Dataset 6']
# Scenarios = ['tinySquare1noRot', 'tinySquare1noRotLow', 'tinySquare1Rot']
# Scenarios = ['bigsquare1', 'bigsquare1normalheight', 'bigsquare1tilted']
magNames = ['full']
PlotLinalgError = 1

Nplot = 10
P0plot = np.zeros((Nplot, Nplot, 6))
P1plot = np.zeros((Nplot, Nplot, 6))
fPlotNorm = np.zeros((Nplot, Nplot, 6))

# Second 3D scatter plot
# fig, axs = plt.subplots(
#     5, len(Scenarios),
#     figsize=(len(Scenarios)*4, 14),
#     gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1]}
# )

fig = plt.figure(figsize=(len(Scenarios)*4, 16))
gs = gridspec.GridSpec(5, len(Scenarios), height_ratios=[1.8, 1.2, 1.2, 1.2, 1.2])
axs = np.empty((5, len(Scenarios)), dtype=object)

for row in range(5):
    for col in range(len(Scenarios)):
        axs[row, col] = fig.add_subplot(gs[row, col])
cmap = plt.cm.magma(np.linspace(0, 1, 3))


for setting in Settings:
    for col, scenario in enumerate(Scenarios):
        my_file = scenario + setting + 'full' + '.npz'
        Data = np.load(os.path.join(my_path, my_data, 'WalkPath' + my_file))

        Xdata = Data['Xdata']
        ydata_n = Data['ydata_n']
        fPred, covPred = GP.predictCurlFreeCon(Xdata[:, ::15], Xpred, ydata_n[:, ::15], modelParameters)
        P0plot[:, :, col], P1plot[:, :, col], fPlotNorm[:, :, col] = makeIntoCountour(Xpred*1000, fPred, Nplot)
Vmin1 = np.min(fPlotNorm[:, :, :3])
Vmax1 = np.max(fPlotNorm[:, :, :3])

Vmin2 = np.min(fPlotNorm[:, :, 3:])
Vmax2 = np.max(fPlotNorm[:, :, 3:])

Vmin = np.array([Vmin1, Vmin1, Vmin1, Vmin2, Vmin2, Vmin2])
Vmax = np.array([Vmax1, Vmax1, Vmax1, Vmax2, Vmax2, Vmax2])
for setting in Settings:
    for col, scenario in enumerate(Scenarios):
        sc = axs[0, col].contourf(P0plot[:, :, col], P1plot[:, :, col], fPlotNorm[:, :, col], levels=25, cmap='viridis', vmin=Vmin[col], vmax=Vmax[col])
        if col == 0:  
            cbar_ax = fig.add_axes([0.0455, 0.945, 0.435, 0.01]) 
            cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
            cbar.ax.set_xlabel(r"Magnetic field norm [$\mu$T]", fontsize=12, fontweight='bold')
        if col == 3:  
            cbar_ax = fig.add_axes([0.545, 0.945, 0.435, 0.01])  
            cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
            cbar.ax.set_xlabel(r"Magnetic field norm [$\mu$T]", fontsize=12, fontweight='bold')
        for c in sc.collections:
            c.set_rasterized(True)
                
        axs[0, col].set_xlim(-300, 300)
        axs[0, col].set_ylim(-300, 300)

        my_file = scenario + setting + 'full' + '.npz'
        Data = np.load(os.path.join(my_path, my_data, 'WalkPath' + my_file))

        dx_stored = Data['dx_stored']
        cov_stored = Data['cov_stored']
        dx_est_stored = Data['dx_est_stored']
        R_true_stored = Data['R_true_stored']
        R_est_stored = Data['R_est_stored']
        eta_error_stored = Data['eta_error_stored']
        StoreLinAlgError = Data['StoreLinAlgError']

        # axs[0, col].imshow(fPlotNorm, extent=[P0plot.min(), P0plot.max(), P1plot.min(), P1plot.max()], origin='lower', cmap='viridis')
        # plt.show()
        dx_stored_n = dx_stored*0
        dx_est_stored_n = dx_est_stored*0
        eta_est_stored_n = dx_est_stored*0
        eta_est2_stored_n = dx_est_stored*0
        eta_true_stored_n = dx_est_stored*0
        dx_est_stored_n2 = dx_est_stored*0
        cov_stored_n = cov_stored*0
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
                cov_stored_n[:, :, i, j] = np.kron(np.eye(2), Rest) @ cov_stored[:, :, i, j] @ np.kron(np.eye(2), Rest.T)
                dx_est_stored_n[:, i:i+1, j] = Rest @ dx_est_stored[:, i:i+1, j]
                dx_stored_n[:, i:i+1, j] = Rtrue @ dx_stored[:, i:i+1, j]
                eta_est_stored_n[:, i:i+1, j] = linAlg.R2eta(Rtrue.T @ Rest)
                eta_est2_stored_n[:, i:i+1, j] = linAlg.R2eta(Rest)
                eta_true_stored_n[:, i:i+1, j] = linAlg.R2eta(Rtrue)

        dx_est_stored_n[np.isnan(dx_est_stored_n)] = 0
        cov_stored_n[np.isnan(cov_stored_n)] = 0
        cov_stored_n_sum = np.cumsum(cov_stored_n.reshape(6, 6, -1), axis=2)

        sigma_stored_n_sum = np.sqrt(cov_stored_n_sum)
        sigma_stored_n_sum[:3, :3, :] *= 1000
        sigma_stored_n_sum[3:, 3:, :] *= 180/np.pi
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

        axs[0, col].plot(dx_sum_n[0, :], dx_sum_n[1, :], color = cmap[0])
        axs[0, col].plot(dx_est_sum_n[0, :], dx_est_sum_n[1, :], color = cmap[1])
        # axs[0, col].scatter(dx_sum_n[0, -1], dx_sum_n[1, -1], c = "red")
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

        axs[1, col].plot(np.linspace(0, dx_sum_n.shape[1], dx_sum_n.shape[1])/10, dx_sum_n[2, :], color = cmap[0], label='Ground truth position or orientation')
        axs[1, col].plot(np.linspace(0, dx_est_sum_n.shape[1], dx_est_sum_n.shape[1])/10, dx_est_sum_n[2, :], color = cmap[1], label='Estimated position or orientation')
        axs[1, col].fill_between(np.linspace(0, sigma_stored_n_sum.shape[2], sigma_stored_n_sum.shape[2])/10, 
                     dx_est_sum_n[2, :] - sigma_stored_n_sum[2, 2, :], 
                     dx_est_sum_n[2, :] + sigma_stored_n_sum[2, 2, :], 
                     color=cmap[1], alpha=0.3, label='1 standard deviation')
        # axs[2, col].plot(np.linspace(0, eta_n.shape[1], eta_n.shape[1]), np.sqrt(eta_n[0, :]**2 + eta_n[1, :]**2 + eta_n[2, :]**2), label='Estimate')
        axs[2, col].plot(np.linspace(0, eta2_n.shape[1], eta2_n.shape[1])/10, eta2_n[0, :], color = cmap[1])#, label='$\eta_1 Est$')
        axs[3, col].plot(np.linspace(0, eta2_n.shape[1], eta2_n.shape[1])/10, eta2_n[1, :], color = cmap[1])#, label='$\eta_2 Est$')
        axs[4, col].plot(np.linspace(0, eta2_n.shape[1], eta2_n.shape[1])/10, eta2_n[2, :], color = cmap[1])#, label='$\eta_3 Est$')
        axs[2, col].fill_between(np.linspace(0, sigma_stored_n_sum.shape[2], sigma_stored_n_sum.shape[2])/10, 
                 eta2_n[0, :] - sigma_stored_n_sum[3, 3, :], 
                 eta2_n[0, :] + sigma_stored_n_sum[3, 3, :], 
                 color=cmap[1], alpha=0.3)
        axs[3, col].fill_between(np.linspace(0, sigma_stored_n_sum.shape[2], sigma_stored_n_sum.shape[2])/10, 
                 eta2_n[1, :] - sigma_stored_n_sum[4, 4, :], 
                 eta2_n[1, :] + sigma_stored_n_sum[4, 4, :], 
                 color=cmap[1], alpha=0.3)
        axs[4, col].fill_between(np.linspace(0, sigma_stored_n_sum.shape[2], sigma_stored_n_sum.shape[2])/10, 
                 eta2_n[2, :] - sigma_stored_n_sum[5, 5, :], 
                 eta2_n[2, :] + sigma_stored_n_sum[5, 5, :], 
                 color=cmap[1], alpha=0.3)
        axs[2, col].plot(np.linspace(0, eta_true_n.shape[1], eta_true_n.shape[1])/10, eta_true_n[0, :], color = cmap[0])
        axs[3, col].plot(np.linspace(0, eta_true_n.shape[1], eta_true_n.shape[1])/10, eta_true_n[1, :], color = cmap[0])
        axs[4, col].plot(np.linspace(0, eta_true_n.shape[1], eta_true_n.shape[1])/10, eta_true_n[2, :], color = cmap[0])
        print(dx_sum_n[:, -1] - dx_est_sum_n[:, -1])
        print(dx_sum2[:, -1])
for rowndx in range(1, 5):
    for colndx in range(6):
        axs[rowndx, colndx].grid(True)

fig.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at top for colorbars

handles1, labels1 = axs[1, 1].get_legend_handles_labels()
# handles2, labels2 = axs[2, 0].get_legend_handles_labels()
handles = handles1# + handles2
labels = labels1# + labels2
fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.995), fontsize = 16)

# cbar = fig.colorbar(sc, ax=axs[0, :], orientation='horizontal', fraction=0.05, pad=0.1)
# cbar.set_label('Colourbar Label', fontsize=16, fontweight='bold')
plt.show()

my_path = os.getcwd()
my_figures = 'Figures'
my_file = 'WalkPath2D' + '.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')
