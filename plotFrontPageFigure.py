import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
import os
import GP as GP
import linAlg as linAlg
import magArray as magArray
import helper as helper
from scipy.linalg import LinAlgError
from scipy.interpolate import CubicSpline, griddata
import matplotlib.colors as mcolors
from scipy import linalg as alg
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import scipy.optimize

''' Initialise parameters '''

np.random.seed(1)
theta_init = np.array([0.0225, 25, .1, 225])
theta = theta_init

magnetometers = list(range(30))
scenario = 'tinySquare3noRotLow'

Narray = len(magnetometers)
Rho = magArray.magArrayPos(magnetometers)

if scenario == 'bigsquare1':
    TrimStart, TrimEnd = 5000, 21000 - 1
elif scenario == 'bigsquare1normalheight':
    TrimStart, TrimEnd = 5000, 9500
elif scenario == 'bigsquare1tilted':
    TrimStart, TrimEnd = 5000, 10750
elif scenario in ['tinySquare1noRotLow', 'tinySquare2noRotLow', 'tinySquare3noRotLow',
                   'tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot',
                   'tinySquare1Rot', 'tinySquare2Rot', 'tinySquare3Rot']:
    TrimStart, TrimEnd = 2000, 2000

TrimSlice = 150
TakeSlices = [400, 1250]

m = {
    "Narray": Narray,
    "magnetometers": magnetometers,
    "theta": theta,
    "Rho": Rho,
    "TrimStart": TrimStart,
    "TrimEnd": TrimEnd,
    "TrimSlice": TrimSlice,
    "TakeSlices": TakeSlices,
}

''' Load and preprocess data '''
my_path = os.getcwd()
my_data = 'ArrayData'

calib_file = 'calibrationBefore2_results.mat'
Data = scipy.io.loadmat(os.path.join(my_path, my_data, calib_file))
m['mag_D'], m['mag_o'], m['SigmaY'] = magArray.processCalibrationData2(Data)

scenario_file = scenario + '.mat'
Data = scipy.io.loadmat(os.path.join(my_path, my_data, scenario_file))
ydata_b, ydata_n, Xcenter, Xdata, Rdata = magArray.preprocessMagData2(Data, m)

ynorm = np.sqrt(np.sum(ydata_b**2, axis=0))
plt.figure(figsize=(12,4))
plt.scatter(Xdata[0], -Xdata[1], c=ynorm)
cbar = plt.colorbar(orientation='vertical')
cbar.set_label(r'$[\mu T]$')
plt.show()

''' Slice data for two timestep '''
timestep = 0
Step = 500
indx1, indx2 = timestep, timestep + Step

slice_indices = lambda i: slice(i*Narray, (i+1)*Narray)
ydata1_b, ydata2_b = ydata_b[:, slice_indices(indx1)], ydata_b[:, slice_indices(indx2)]
ydata1_n, ydata2_n = ydata_n[:, slice_indices(indx1)], ydata_n[:, slice_indices(indx2)]

Ydata_b = np.hstack((ydata1_b, ydata2_b))
Ydata_n = np.hstack((ydata1_n, ydata2_n))
Xdata_1, Xdata_2 = Xdata[:, slice_indices(indx1)], Xdata[:, slice_indices(indx2)]
XdataStack = np.hstack((Xdata_1, Xdata_2)) - np.mean(np.hstack((Xdata_1, Xdata_2)), axis=1, keepdims=True)

''' Prediction '''
AngleRotation = 0.4
Length1, Length2 = 0.8, 0.4
PlotX1Start, PlotX1End = -Length1, Length1
PlotX2Start, PlotX2End = -Length2, Length2
distanceBetween = 0.425

Xpred = helper.gridpoints2(25, 2, 3, -Length1, Length1, -Length2, Length2)
Rz = lambda angle: linAlg.Rz(angle)
Xarray1 = Rz(-AngleRotation) @ Rho * 1.5 + np.array([[-distanceBetween], [0], [0]])
Xarray2 = Rz(AngleRotation) @ Rho * 1.5 + np.array([[distanceBetween], [0], [0]])
Xarray1Edge = Rz(-AngleRotation) @ (1.75 * Rho) + np.array([[-distanceBetween], [0], [0]])
Xarray2Edge = Rz(AngleRotation) @ (1.75 * Rho) + np.array([[distanceBetween], [0], [0]])


K11 = GP.kernelCurlFree(Xarray1, Xarray1, m) + GP.kernelConstant(Xarray1, Xarray1, m)
K21 = GP.kernelCurlFree(Xpred, Xarray1, m) + GP.kernelConstant(Xpred, Xarray1, m)
K22 = GP.kernelCurlFree(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)

L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
alpha = np.linalg.solve(L.T, np.linalg.solve(L, ydata1_b.T.reshape(-1, 1)))
f = K21 @ alpha
v = np.linalg.solve(L, K21.T)
covf = K22 - v.T @ v
f = f.reshape(-1, 3).T
fnorm = np.linalg.norm(f, axis=0)
covtraced = np.diag(covf).reshape(-1, 1)
alphas = covtraced[::3] + covtraced[1::3] + covtraced[2::3]


Nplot = 1000
X0 = np.linspace(PlotX1Start, PlotX1End, Nplot)
X1 = np.linspace(PlotX2Start, PlotX2End, Nplot)
XX0, XX1 = np.meshgrid(X0, X1)
Alphas = griddata((Xpred[0], Xpred[1]), alphas, (XX0, XX1), method='cubic')
Alphas = np.squeeze((Alphas - Alphas.min()) / (Alphas.max() - Alphas.min()))


K11 = GP.kernelCurlFree(Xdata, Xdata, m) + GP.kernelConstant(Xdata, Xdata, m)
K21 = GP.kernelCurlFree(Xpred, Xdata, m) + GP.kernelConstant(Xpred, Xdata, m)
K22 = GP.kernelCurlFree(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)

L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
alpha = np.linalg.solve(L.T, np.linalg.solve(L, ydata_n.T.reshape(-1, 1)))
f = K21 @ alpha
v = np.linalg.solve(L, K21.T)
covf = K22 - v.T @ v
f = f.reshape(-1, 3).T
fnorm = np.linalg.norm(f, axis=0)
F = griddata((Xpred[0], Xpred[1]), fnorm, (XX0, XX1), method='cubic')

''' Plot results '''
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
contour = ax.contourf(XX0, -XX1, F, levels=30, cmap='gray', antialiased=True)
for c in contour.collections:
    c.set_rasterized(True)

Layers = 25
vmin, vmax = np.min(F), np.max(F)
for i in range(Layers):
    threshold_max = (i + 1) / Layers
    mask = np.zeros_like(Alphas)
    mask[Alphas**(1/15) <= threshold_max] = 1
    masked_F = np.ma.masked_where(mask == 0, F)
    contour = ax.contourf(XX0, -XX1, masked_F, levels=30, cmap='viridis', vmin=vmin, vmax=vmax, alpha=1-threshold_max, antialiased=True)
    for c in contour.collections:
        c.set_rasterized(True)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.125)
cbar.set_label(r'Magnetic field norm [$\mu$ T]')

plt.xlabel(r'$p_1$ [m]')
plt.ylabel(r'$p_2$ [m]')


for edge, style in zip([Xarray1Edge, Xarray2Edge], ['-', ':']):
    ax.plot(edge[0, [0, 5]], -edge[1, [0, 5]], 'white', linewidth=2, linestyle=style)
    ax.plot(edge[0, [5, -1]], -edge[1, [5, -1]], 'white', linewidth=2, linestyle=style)
    ax.plot(edge[0, [-1, -6]], -edge[1, [-1, -6]], 'white', linewidth=2, linestyle=style)
    ax.plot(edge[0, [-6, 0]], -edge[1, [-6, 0]], 'white', linewidth=2, linestyle=style)

ax.scatter(Xarray1[0], -Xarray1[1], c='red', s=50, edgecolors='white', label='Magnetometer on array')
ax.scatter(Xarray2[0], -Xarray2[1], facecolors='none', edgecolors='white', linestyle=':', s=50)

Xarray1c = np.mean(Xarray1, axis=1)
Xarray2c = np.mean(Xarray2, axis=1)
ax.plot([Xarray1c[0], Xarray2c[0] - 0.015], [Xarray1c[1], Xarray2c[1]], linestyle=':', color='white', linewidth=2)
arrow = patches.FancyArrowPatch((Xarray2c[0] - 0.015, Xarray1c[1]), (Xarray2c[0], Xarray1c[1]),
                                 arrowstyle='-|>', color='white', mutation_scale=15)
ax.add_patch(arrow)
ax.text((Xarray1c[0] + Xarray2c[0]) / 2, (Xarray1c[1] + Xarray2c[1]) / 2 + 0.025, 'Pose change',
        color='white', ha='center', va='center', fontsize=12)
ax.text((Xarray1c[0] + Xarray2c[0]) / 2, (Xarray1c[1] + Xarray2c[1]) / 2 - 0.025,
        r'$\Delta \mathbf{p}, \Delta \mathbf{R}$', color='white', ha='center', va='center', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.show()


''' Save the figure '''
fig_path = os.path.join(my_path, 'Figures', 'FieldNormFrontPage.pdf')
fig.savefig(fig_path, format='pdf', bbox_inches='tight')