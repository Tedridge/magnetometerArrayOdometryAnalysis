import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import griddata
import linAlg


def reshapematrix(A):
    ''' Reshape 3D matrix A into 2D matrix by stacking along axis=2 '''
    B = np.zeros((A.shape[0], 1))
    for i in range(A.shape[2]): 
        B = np.hstack((B, A[:, :, i]))
    return B[:, 1:]


''' Settings '''
splitParts = 4
scenario = 'bigsquare1'
setting = 'Zero'
my_path = os.getcwd()
my_data = 'ArrayData'
cmap = plt.cm.magma(np.linspace(0, 1, 3))

''' Load in the data '''
my_file = scenario + setting + 'full' + '.npz'
Data = np.load(os.path.join(my_path, my_data, 'WalkPath' + my_file))

Xdata = Data['Xdata']
Xcenter = Data['Xcenter']
Xcenter[1, :] = -Xcenter[1, :]
Xcenter -= np.mean(Xcenter, axis=1, keepdims=True)

Xdata[[0, 1], :] = Xdata[[1, 0], :]
Xdata[0, :] = -Xdata[0, :]
Xdata -= np.mean(Xdata, axis=1, keepdims=True)

ydata_n = Data['ydata_n']
posStoredBig = Data['posStoredBig']
posStoredBig[[0, 1], :] = posStoredBig[[1, 0], :]
posStoredBig[0, :] = -posStoredBig[0, :]
posStoredBig[1, :] = -posStoredBig[1, :]
posStoredBig[0, :] += 100 / 1000
posStoredBig[1, :] += 150 / 1000
if scenario == 'bigsquare1normalheight':
    posStoredBig[1, :] += 50 / 1000

yStoredBig = Data['yStoredBig']

dx_stored = Data['dx_stored']
cov_stored = Data['cov_stored']
dx_est_stored = Data['dx_est_stored']
R_true_stored = Data['R_true_stored']
R_est_stored = Data['R_est_stored']
eta_error_stored = Data['eta_error_stored']
StoreLinAlgError = Data['StoreLinAlgError']

dx_stored_n = dx_stored * 0
dx_est_stored_n = dx_est_stored * 0
eta_est_stored_n = dx_est_stored * 0
eta_est2_stored_n = dx_est_stored * 0
eta_true_stored_n = dx_est_stored * 0
cov_stored_n = cov_stored * 0

TimeSteps = dx_stored.shape[2]
Steps = dx_stored.shape[1]
splitNumber = int(R_est_stored.shape[2] * R_est_stored.shape[3] / splitParts)

dx_est_stored[np.isnan(dx_est_stored)] = 0

''' process data '''
initRot = 0.02
if scenario == 'bigsquare1normalheight':
    initRot = np.pi / 2 - 0.125

Rtrue = linAlg.Rz(initRot)
Rest = linAlg.Rz(initRot)
Rest3 = linAlg.Rz(initRot)

counter = 0
for j in range(TimeSteps):
    for i in range(Steps):
        if counter != 0:
            Rest3 = Rest3 @ R_est_stored[:, :, i, j]
            Rest = Rest @ R_est_stored[:, :, i, j]
            Rtrue = Rtrue @ R_true_stored[:, :, i, j]
        if counter % int(TimeSteps * Steps / splitParts) == 0:
            Rest3 = Rtrue
        cov_stored_n[:, :, i, j] = np.kron(np.eye(2), np.eye(3)) @ cov_stored[:, :, i, j] @ np.kron(np.eye(2), np.eye(3))
        dx_est_stored_n[:, i:i+1, j] = Rtrue @ dx_est_stored[:, i:i+1, j]
        dx_stored_n[:, i:i+1, j] = Rtrue @ dx_stored[:, i:i+1, j]
        eta_est_stored_n[:, i:i+1, j] = linAlg.R2eta(Rtrue.T @ Rest3)
        eta_est2_stored_n[:, i:i+1, j] = linAlg.R2eta(Rest3)
        eta_true_stored_n[:, i:i+1, j] = linAlg.R2eta(Rtrue)
        counter += 1

print(f"Number of NaN values in cov_stored_n: {np.isnan(cov_stored_n).sum()}")
cov_stored_n[np.isnan(cov_stored_n)] = 0
cov_stored_n = cov_stored_n.reshape(6, 6, -1)

cov_stored_n_sum = np.zeros((6, 6, splitParts * splitNumber))
for col in range(splitParts):
    start = col * splitNumber
    end = (col + 1) * splitNumber
    cov_stored_n_sum[:, :, start:end] = np.cumsum(cov_stored_n[:, :, start:end], axis=2)

sigma_stored_n_sum = np.sqrt(cov_stored_n_sum)
sigma_stored_n_sum[3:, 3:, :] *= 180 / np.pi

dx_est_sum_n = 1000 * reshapematrix(dx_est_stored_n)
dx_sum_n = 1000 * reshapematrix(dx_stored_n)
eta2_n = 180 / np.pi * reshapematrix(eta_est2_stored_n)
eta_true_n = 180 / np.pi * reshapematrix(eta_true_stored_n)

for indx in range(3):
    eta2_n[indx, :] = np.rad2deg(np.unwrap(np.deg2rad(eta2_n[indx, :]), discont=np.pi))
    eta_true_n[indx, :] = np.rad2deg(np.unwrap(np.deg2rad(eta_true_n[indx, :]), discont=np.pi))

threshold = 30
for t in range(1, eta_true_n.shape[1]):
    if eta_true_n[1, t] * eta_true_n[1, t - 1] < 0:
        eta_true_n[1, t] *= -1
    if eta2_n[1, t] * eta2_n[1, t - 1] < 0:
        eta2_n[1, t] *= -1
    for indx in range(3):
        if abs(eta_true_n[indx, t] - eta_true_n[indx, t - 1]) > threshold:
            eta_true_n[indx, t] = eta_true_n[indx, t - 1]

dx_sum_n_plot_tot = np.cumsum(dx_sum_n, axis=1)
dx_sum_n_plot_tot -= np.mean(dx_sum_n_plot_tot, axis=1, keepdims=True)



fig = plt.figure(figsize=(24, 8))
gs = gridspec.GridSpec(splitParts, 7, width_ratios=[3] + [1] * 6, wspace=0.3, hspace=0.6)

''' Left plot '''
ax_left = fig.add_subplot(gs[:, 0])
sc = ax_left.scatter(posStoredBig[0, :], posStoredBig[1, :],
                     c=np.linalg.norm(yStoredBig[:, :], axis=0), cmap='viridis', s=100)
cbar = fig.colorbar(sc, ax=ax_left, orientation='horizontal', pad=0.1, aspect=40)
cbar.set_label(r'Norm of magnetic field [$\mu$T]', fontsize=12)

ax_left.plot([], [], color='black', label='Ground Truth Trajectory')
ax_left.plot([], [], color=cmap[1], label='Estimated Trajectory')
ax_left.scatter([], [], color='red', s=100, marker='o', label='Start estimated segment')
ax_left.scatter([], [], color='red', s=100, marker='x', label='End estimated segment')
ax_left.fill_between([], [], [], color=cmap[1], alpha=0.3, label='1 std')

ax_left.legend(loc='lower center', bbox_to_anchor=(0.5, -0.425), fontsize=10, ncol=3)

segmentPoints = ['A', 'B', 'C', 'D', 'E']
for col in range(splitParts):
    start = col * splitNumber
    end = (col + 1) * splitNumber
    horiz = 500 * (0 if col == 0 else (1 if col == 1 else (0 if col == 2 else -1)))
    vert = 500 * (1 if col == 0 else (0 if col == 1 else (-1 if col == 2 else 0)))

    initPos = dx_sum_n_plot_tot[:, start:start+1]
    ax_left.text(initPos[0, 0]/1000 + horiz/1000, initPos[1, 0]/1000 + vert/1000,
                 segmentPoints[col], fontsize=20, color='black', ha='center', va='center')
    
    if col == splitParts - 1:
        ax_left.text(dx_sum_n_plot_tot[0, -1]/1000, dx_sum_n_plot_tot[1, -1]/1000 + 0.5,
                     segmentPoints[col + 1], fontsize=20, color='black', ha='center', va='center')

    ax_left.scatter(initPos[0, 0]/1000, initPos[1, 0]/1000, color='red', s=100, marker='o')

    dx_sum_n_plot = initPos + np.cumsum(dx_sum_n[:, start:end], axis=1)
    dx_est_sum_n_plot = initPos + np.cumsum(dx_est_sum_n[:, start:end], axis=1)
    dx_est_sum_n_plot = dx_est_sum_n_plot - dx_est_sum_n_plot[:, [0]] + dx_sum_n_plot[:, [0]]

    ax_left.plot(dx_sum_n_plot[0, :]/1000, dx_sum_n_plot[1, :]/1000, color=cmap[0])
    ax_left.plot(dx_est_sum_n_plot[0, :]/1000, dx_est_sum_n_plot[1, :]/1000, color=cmap[1])
    ax_left.scatter(dx_est_sum_n_plot[0, -1]/1000, dx_est_sum_n_plot[1, -1]/1000, color='red', s=100, marker='x')

ax_left.set_xlim(-4.5, 4.5)
ax_left.set_ylim(-4.5, 4.5)
ax_left.set_xlabel(r'Horizontal position $p_1$ [m]', fontsize=12)
ax_left.set_ylabel(r'Horizontal position $p_2$ [m]', fontsize=12)
ax_left.set_title("Trajectory `LP-1`" if scenario == 'bigsquare1' else "Trajectory `NP-1`", fontsize=14, fontweight='bold')

''' Right subplots '''
labels = [r'$p_1$ [m]', r'$p_2$ [m]', r'$p_3$ [m]', r'Roll [$\degree$]', r'Pitch [$\degree$]', r'Yaw [$\degree$]']
segmentTitles = ['Segment A-B', 'Segment B-C', 'Segment C-D', 'Segment D-E']

for row in range(splitParts):
    start = row * splitNumber
    end = (row + 1) * splitNumber
    initPos = dx_sum_n_plot_tot[:, start:start+1]

    dx_sum_n_plot = initPos + np.cumsum(dx_sum_n[:, start:end], axis=1)
    dx_est_sum_n_plot = initPos + np.cumsum(dx_est_sum_n[:, start:end], axis=1)
    dx_est_sum_n_plot = dx_est_sum_n_plot - dx_est_sum_n_plot[:, [0]] + dx_sum_n_plot[:, [0]]

    sigma_plot = sigma_stored_n_sum[:3, :3, start:end]
    sigma_eta_plot = sigma_stored_n_sum[3:, 3:, start:end]
    eta2_n_plot = eta2_n[:, start:end]
    eta_true_n_plot = eta_true_n[:, start:end]

    t = np.linspace(0, dx_sum_n_plot.shape[1], dx_sum_n_plot.shape[1]) / 10

    for col in range(6):
        ax = fig.add_subplot(gs[row, col + 1])

        if col < 3:
            idx = col
            ax.plot(t, dx_sum_n_plot[idx, :] / 1000, color=cmap[0])
            ax.plot(t, dx_est_sum_n_plot[idx, :] / 1000, color=cmap[1])
            ax.fill_between(t,
                            dx_est_sum_n_plot[idx, :] / 1000 - sigma_plot[idx, idx, :],
                            dx_est_sum_n_plot[idx, :] / 1000 + sigma_plot[idx, idx, :],
                            color=cmap[1], alpha=0.3)
        else:
            eta_idx = col - 3
            ax.plot(t, eta_true_n_plot[eta_idx, :], color=cmap[0])
            ax.plot(t, eta2_n_plot[eta_idx, :], color=cmap[1])
            ax.fill_between(t,
                            eta2_n_plot[eta_idx, :] - sigma_eta_plot[eta_idx, eta_idx, :],
                            eta2_n_plot[eta_idx, :] + sigma_eta_plot[eta_idx, eta_idx, :],
                            color=cmap[1], alpha=0.3)

        ax.set_ylabel(labels[col], fontsize=10, fontweight='bold')
        ax.set_title(segmentTitles[row], fontsize=10, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax.grid(True)
        ax.tick_params(labelsize=8)

fig.tight_layout()
plt.show()

''' Save figure '''
my_figures = 'Figures'
my_file = 'WalkPath2D_MAINS' + scenario + '.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')
