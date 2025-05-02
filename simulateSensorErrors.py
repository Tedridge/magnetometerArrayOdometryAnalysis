import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as alg
import os

import magArray as magArray
import linAlg as linAlg
import helper as helper
import motionModels as motionModels
import GP as GP

np.random.seed()

''' Constants ''' 
Din = 3
Narray = 25
Array_Shape = 'Square'

steps = 25
stepSize = np.sqrt(0.15**2) / 20
Nmc = 3
sensorErrorLevels = [0, 1, 2, 3]
sensorErrorLabels = ['None', 'Low', 'Medium', 'High']
sensorErrorSettings = ['Bias', 'Misalignment Rotation', 'Misalignment Position']
Colours = plt.cm.viridis(np.linspace(0, 1, len(sensorErrorLevels)))

'''  Hyperparameters and Input Locations '''
theta = np.array([0.15**2, 5**2, 0.0012, 15**2])
RhoBase = magArray.shape(Array_Shape, Narray, np.sqrt(theta[0]))
Rho = RhoBase

SigmaY = linAlg.identityArray(3, (Narray,)) * theta[2]

m = {
    "theta": theta,
    "Rho": Rho,
    "Din": Din,
    "Narray": Narray,
    "SigmaY": SigmaY,
}

''' Storage '''
XestStored = np.zeros((3, steps + 1, len(sensorErrorLevels), len(sensorErrorSettings), Nmc))
XestStored2 = np.zeros((3, steps + 1, len(sensorErrorLevels), len(sensorErrorSettings), Nmc))
dXStored = np.zeros((3, steps + 1, len(sensorErrorLevels), len(sensorErrorSettings), Nmc))
Rstored = np.zeros((3, 3, steps + 1, len(sensorErrorLevels), len(sensorErrorSettings), Nmc))
etaStored = np.zeros((3, steps + 1, len(sensorErrorLevels), len(sensorErrorSettings), Nmc))

''' Simulation '''
for mndx in np.array([0]):
    for nndx in range(Nmc):
        Xcenter, Xarray, Rb2n = motionModels.motionModelLshape(steps, stepSize, mndx, m)
        ydatacurl, fdatacurl, _ = GP.datagenCurlFree(Xarray, np.zeros((3, 1)), Rb2n, np.zeros_like(Xarray), m)

        biasErrorsBase = np.kron(np.ones((1, steps + 1)), np.random.normal(0, 0.25, (3, Narray)))
        misalignmentsPosBase = np.random.normal(0, 0.0005, (3, Narray))
        misalignmentsRotBase = np.kron(np.ones((1, steps + 1)), np.random.normal(0, 0.5 * np.pi / 180, (3, Narray)))

        for lndx, sensorErrorSetting in enumerate(sensorErrorSettings):
            for kndx, sensorErrorLevel in enumerate(sensorErrorLevels):
                print(f'MC simulation number: {nndx} - {sensorErrorLabels[kndx]} {sensorErrorSetting}')

                biasErrors = misalignmentsRot = misalignmentsPos = np.zeros_like(biasErrorsBase)
                Ycurl = ydatacurl

                if lndx == 0:
                    biasErrors = biasErrorsBase * sensorErrorLevel
                elif lndx == 1:
                    misalignmentsRot = misalignmentsRotBase * sensorErrorLevel
                elif lndx == 2:
                    misalignmentsPos = misalignmentsPosBase * sensorErrorLevel
                    Ycurl = GP.datagenCurlFreePosteriorSim(Xarray + np.kron(np.ones((1, steps+1)), misalignmentsPos), Xarray, fdatacurl, Rb2n, m)

                Xest = np.zeros((3, 1))
                Xest2 = np.zeros((3, 1))
                R = np.eye(3)
                for indx in range(Xcenter.shape[1] - 1):
                    y0 = linAlg.expR(misalignmentsRot[:, indx:indx+1]) @ (Ycurl[:, indx*Narray:(indx+1)*Narray] + biasErrors[:, indx*Narray:(indx+1)*Narray])
                    y1 = linAlg.expR(misalignmentsRot[:, indx+1:indx+2]) @ (Ycurl[:, (indx+1)*Narray:(indx+2)*Narray] + biasErrors[:, (indx+1)*Narray:(indx+2)*Narray])
                    Y = np.hstack((y0, y1))

                    init = np.zeros((6, 1))
                    dx, dR, _ = magArray.poseEstArrayWLS(init, Y, m)
                    Xest += R @ dx
                    Xest2 += dx
                    R = dR @ R

                    XestStored[:, indx + 1, kndx, lndx, nndx] = Xest[:, 0]
                    dXStored[:, indx + 1, kndx, lndx, nndx] = dx[:, 0]
                    Rstored[:, :, indx + 1, kndx, lndx, nndx] = R
                    etaStored[:, indx + 1, kndx, lndx, nndx] = linAlg.R2eta(R)[:, 0]

    ''' plot '''
    def make_plots_summary():
        fig, axes = plt.subplots(6, 3, figsize=(10, 10))
        for j, setting in enumerate(sensorErrorSettings):
            for i, label in reversed(list(enumerate(sensorErrorLabels))):
                for m in range(3):
                    XestMean = np.mean(np.abs(XestStored[m, :, i, j, :] - Xcenter[m, :][:, None]), axis=1)
                    etaMean = np.mean(180 / np.pi * np.abs(etaStored[m, :, i, j, :]), axis=1)
                    axes[m, j].fill_between(np.linspace(0, (steps+1)*stepSize, steps+1), XestMean, 0, color=Colours[i], label=label)
                    axes[m+3, j].fill_between(np.linspace(0, (steps+1)*stepSize, steps+1), etaMean, 0, color=Colours[i], label=label)
        for ax_col in axes.T:
            for ax in ax_col:
                ax.set_xlabel('Displacement in $x_1$ [m]')
        for row, ylabel in enumerate(['$\\epsilon_1$ [m]', '$\\epsilon_2$ [m]', '$\\epsilon_3$ [m]', '$\\eta_1$ [°]', '$\\eta_2$ [°]', '$\\eta_3$ [°]']):
            axes[row, 0].set_ylabel(f'Mean error {ylabel}')
        for j, setting in enumerate(sensorErrorSettings):
            axes[0, j].set_title(setting)
        fig.legend([Patch(color=Colours[i]) for i in range(len(sensorErrorLabels))], sensorErrorLabels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(sensorErrorLabels))
        plt.tight_layout()
        plt.show()
        return fig

    fig = make_plots_summary()
    fig.savefig(os.path.join(os.getcwd(), 'Figures', f'SensorErrors{mndx}.pdf'), format='pdf', bbox_inches='tight')

    ''' Save data '''
    np.savez(
        os.path.join(os.getcwd(), 'Data', f'MC_errors_{"xyz"[mndx]}.npz'),
        Xcenter=Xcenter,
        XestStored=XestStored,
        dXStored=dXStored,
        Rstored=Rstored,
        etaStored=etaStored,
        theta=theta,
        steps=steps,
        stepSize=stepSize,
        Nmc=Nmc,
        sensorErrorLevels=sensorErrorLevels,
        sensorErrorLabels=sensorErrorLabels,
        sensorErrorSettings=sensorErrorSettings
    )
