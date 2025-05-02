import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
import os
import GP as GP
import linAlg as linAlg
import helper as helper
import magArray as magArray
from scipy.interpolate import griddata
import matplotlib.colors as mcolors


# ________________________________Initialise parameters________________________________#
np.random.seed(1)
theta_init = np.array([0.0225, 25, .1, 225])
theta = theta_init

magnetometers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#magnetometers = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]

scenario = 'tinySquare3noRotLow'

Narray = len(magnetometers)
Rho = magArray.magArrayPos(magnetometers)
# TrimStart = 10000
# TrimEnd = 18500
if scenario == 'bigsquare1':
    TrimStart = 5000
    # #TrimEnd = 17000-1
    TrimEnd = 21000-1
    # TrimStart = 10000
    #TrimEnd = 17000-1
    # TrimEnd = 16000-1
elif scenario == 'bigsquare1normalheight':
    TrimStart = 5000
    TrimEnd = 9500
elif scenario == 'bigsquare1tilted':
    TrimStart = 5000
    TrimEnd = 10750
elif scenario == 'tinySquare1noRotLow':
    TrimStart = 2550
    TrimEnd = 6000
elif scenario == 'tinySquare1noRot':
    TrimStart = 2000
    TrimEnd = 2000
elif scenario == 'tinySquare2noRotLow':
    TrimStart = 2000
    TrimEnd = 2000
elif scenario == 'tinySquare2noRot':
    TrimStart = 2000
    TrimEnd = 2000
elif scenario == 'tinySquare3noRotLow':
    TrimStart = 2000
    TrimEnd = 2000
elif scenario == 'tinySquare3noRot':
    TrimStart = 2000
    TrimEnd = 2000
elif scenario == 'tinySquare1Rot':
    TrimStart = 5000
    TrimEnd = 4000
elif scenario == 'tinySquare2Rot':
    TrimStart = 3000
    TrimEnd = 3000
elif scenario == 'tinySquare3Rot':
    TrimStart = 3000
    TrimEnd = 3000

TrimSlice = 150
TakeSlices = [400, 1250] #Visual inspection

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


my_path = os.getcwd()
my_data = 'ArrayData'
#my_file = 'CalibrationChuan.mat'
my_file = 'calibrationBefore2_results.mat'
Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
m['mag_D'], m['mag_o'], m['SigmaY'] = magArray.processCalibrationData2(Data)

my_file = scenario + '.mat'
Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
ydata_b, ydata_n, Xcenter, Xdata, Rdata = magArray.preprocessMagData2(Data, m)

ynorm = np.sqrt(ydata_b[0, :]**2 + ydata_b[1, :]**2 + ydata_b[2, :]**2)
fig = plt.figure(figsize=(12,4))
plt.scatter(Xdata[0, :], -Xdata[1, :], c=ynorm)
cbar = plt.colorbar(orientation='vertical')
cbar.set_label(r'$[\mu T]$') 
plt.show()

timestep = 0
Step = 500

indx1 = timestep
indx2 = indx1+Step


ydata1_b = ydata_b[:, indx1*Narray:(indx1+1)*Narray]
ydata2_b = ydata_b[:, indx2*Narray:(indx2+1)*Narray]
Ydata_b = np.hstack((ydata1_b, ydata2_b))
ydata1_n = ydata_n[:, indx1*Narray:(indx1+1)*Narray]
ydata2_n = ydata_n[:, indx2*Narray:(indx2+1)*Narray]
Ydata_n = np.hstack((ydata1_n, ydata2_n))
Xdata_1 = Xdata[:, indx1*Narray:(indx1+1)*Narray]
Xdata_2 = Xdata[:, indx2*Narray:(indx2+1)*Narray]
XdataStack = np.hstack((Xdata_1, Xdata_2))
XdataStack -= np.mean(XdataStack, 1).reshape(3, 1)

AngleRotation = 0.4
Length1 = .8
Length2 = Length1/2
PlotX1Start = -Length1
PlotX1End = Length1
PlotX2Start = -Length2
PlotX2End = Length2
distanceBetween = 0.425
Xpred = helper.gridpoints2(25, 2, 3, -Length1, Length1, -Length2, Length2)
Xarray1 = linAlg.Rz(-AngleRotation) @ Rho *1.5 + np.array([[-distanceBetween], [0], [0]])
Xarray2 = linAlg.Rz(AngleRotation) @ Rho *1.5 + np.array([[distanceBetween], [0], [0]])
Xarray1Edge = linAlg.Rz(-AngleRotation) @ (1.75*Rho) + np.array([[-distanceBetween], [0], [0]])
Xarray2Edge = linAlg.Rz(AngleRotation) @ (1.75*Rho) + np.array([[distanceBetween], [0], [0]])


K11 = GP.kernelCurlFree(Xarray1, Xarray1, m) + GP.kernelConstant(Xarray1, Xarray1, m)
K21 = GP.kernelCurlFree(Xpred, Xarray1, m) + GP.kernelConstant(Xpred, Xarray1, m)
K22 = GP.kernelCurlFree(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)

L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
Inner = np.linalg.solve(L, ydata1_b.T.reshape(-1, 1))
alpha = np.linalg.solve(L.T, Inner)
f = K21 @ alpha
v = np.linalg.solve(L, K21.T)
covf = K22 - v.T @ v
f = f.reshape(-1, 3).T
fnorm = np.sqrt(f[0, :]**2 + f[1, :]**2 + f[2, :]**2)

covtraced = np.diag(covf).reshape(-1, 1)
alphas = covtraced[::3, :] + covtraced[1::3, :] + covtraced[2::3, :]

# Define parameters
Nplot = 250


# Create grid for interpolation
X0 = np.linspace(PlotX1Start, PlotX1End, Nplot)
X1 = np.linspace(PlotX2Start, PlotX2End, Nplot)
XX0, XX1 = np.meshgrid(X0, X1)

# Interpolate fnorm
# F = griddata((Xpred[0, :], Xpred[1, :]), fnorm, (XX0, XX1), method='cubic')

# Interpolate alphas
Alphas = griddata((Xpred[0, :], Xpred[1, :]), alphas, (XX0, XX1), method='cubic')
Alphas = ((Alphas - Alphas.min()) / (Alphas.max() - Alphas.min()))  # Scale to 0-1
Alphas = np.squeeze(Alphas)


K11 = GP.kernelCurlFree(Xdata, Xdata, m) + GP.kernelConstant(Xdata, Xdata, m)
K21 = GP.kernelCurlFree(Xpred, Xdata, m) + GP.kernelConstant(Xpred, Xdata, m)
K22 = GP.kernelCurlFree(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)

L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
Inner = np.linalg.solve(L, ydata_n.T.reshape(-1, 1))
alpha = np.linalg.solve(L.T, Inner)
f = K21 @ alpha
v = np.linalg.solve(L, K21.T)
covf = K22 - v.T @ v
f = f.reshape(-1, 3).T
fnorm = (np.sqrt(f[0, :]**2 + f[1, :]**2 + f[2, :]**2))
F = griddata((Xpred[0, :], Xpred[1, :]), fnorm, (XX0, XX1), method='cubic')



# Creating a contour plot
fig, ax = plt.subplots(figsize=(6.6, 2.95), dpi=300)
# contour = ax.contourf(XX0, XX1, F, levels=30, cmap='viridis')

# Create filled contour plot
contour = ax.contourf(XX0, -XX1, F, levels=30, cmap='gray')
# Overlay the original data points
# scatter = ax.scatter(XX0, XX1, c='white', s=5, alpha = Alphas, marker = 's')
# Creating a scatter plot
# scatter = plt.scatter(XX0, XX1, c='white', s=50, alpha=Alphas, marker='X')

Layers = 5
vmin = np.min(F)  # Set the minimum value of the color range
vmax = np.max(F)  # Set the maximum value of the color range
for i in range(Layers):
    threshold_min = i / Layers
    threshold_max = (i + 1) / Layers
    
    Alphas2 = Alphas**(1/15)
    # Defining the threshold and creating the mask
    mask = np.zeros_like(Alphas)
    # mask[(Alphas >= threshold_min) & (Alphas <= threshold_max)] = 1
    mask[Alphas2 <= threshold_max] = 1

    # Masking the contour plot
    masked_F = np.ma.masked_where(mask == 0, F)
    contour = ax.contourf(XX0, -XX1, masked_F, levels=30, cmap='viridis', vmin=vmin, vmax=vmax, alpha = 1-threshold_max)
  


# Define the colormap and the norm
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # Set your desired vmin and vmax

# Create a ScalarMappable object
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Create the colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical') #, pad=0.125)
cbar.set_label(r'Magnetic field norm [$\mu$ T]')
# Create the colorbar
# cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
# cbar.set_label(r'$[\mu T]$')
plt.xlabel(r'$p_1$ [m]')
plt.ylabel(r'$p_2$ [m]')

plt.plot(Xarray1Edge[0, [0, 5]], -Xarray1Edge[1, [0, 5]], c='white', linewidth=2)
plt.plot(Xarray1Edge[0, [5, -1]], -Xarray1Edge[1, [5, -1]], c='white', linewidth=2)
plt.plot(Xarray1Edge[0, [-1, -6]] , -Xarray1Edge[1, [-1, -6]], c='white', linewidth=2)
plt.plot(Xarray1Edge[0, [-6, 0]], -Xarray1Edge[1, [-6, 0]], c='white', linewidth=2)

plt.plot(Xarray2Edge[0, [0, 5]], -Xarray2Edge[1, [0, 5]], c='white', linewidth=2, linestyle = ':')
plt.plot(Xarray2Edge[0, [5, -1]], -Xarray2Edge[1, [5, -1]], c='white', linewidth=2, linestyle = ':')
plt.plot(Xarray2Edge[0, [-1, -6]] , -Xarray2Edge[1, [-1, -6]], c='white', linewidth=2, linestyle = ':')
plt.plot(Xarray2Edge[0, [-6, 0]], -Xarray2Edge[1, [-6, 0]], c='white', linewidth=2, linestyle = ':')

plt.scatter(Xarray1[0, :], -Xarray1[1, :], c='red', s=20, edgecolors='white', label = 'Magnetometer on array')
plt.scatter(Xarray2[0, :], -Xarray2[1, :], facecolors='none', edgecolors='white', linestyle = ':', s=20)

Xarray1c = np.mean(Xarray1, 1).reshape(3, 1)
Xarray2c = np.mean(Xarray2, 1).reshape(3, 1)

# Plot the dotted line for the body
plt.plot([Xarray1c[0, 0], Xarray2c[0, 0]-0.015], [Xarray1c[1, 0], Xarray2c[1, 0]], linestyle=':', color='white', linewidth=2)

# Create an arrowhead at the end of the line
arrow = patches.FancyArrowPatch((Xarray2c[0, 0]-0.015, Xarray1c[1, 0]), 
                                (Xarray2c[0, 0], Xarray1c[1, 0]),
                                arrowstyle='-|>', color='white', mutation_scale=15)

# Add the arrow to the plot
plt.gca().add_patch(arrow)

plt.text((Xarray1c[0, 0] + Xarray2c[0, 0]) / 2, (Xarray1c[1, 0] + Xarray2c[1, 0]) / 2  + 0.35, 'Pose change', color='white', ha='center', va='center', fontsize=12)
plt.text((Xarray1c[0, 0] + Xarray2c[0, 0]) / 2, (Xarray1c[1, 0] + Xarray2c[1, 0]) / 2  + 0.025, r'${\Delta \mathbf{p}, \Delta \mathbf{R}}$', color='white', ha='center', va='center', fontsize=10)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2)
# plt.legend()
plt.show()

# Save the figure
my_path = os.getcwd()
my_figures = 'Figures'
my_file = 'abstractImage.png'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='png', bbox_inches='tight')
