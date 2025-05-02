import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

import linAlg as linAlg
import magArray as magArray
import motionModels as motionModels
import helper as helper
import GP as GP

from joblib import Parallel, delayed
from scipy.linalg import LinAlgError


''' To be executed in parallel '''
def process_timestep(timestep):
    print(str(round(timestep/TimeSteps*100, 1)) + "%")
    dx_stored_local = np.zeros((3, Steps))
    dx_est_stored_local = np.zeros((3, Steps))
    eta_error_stored_local = np.zeros((3, Steps))
    StoreLinAlgError_local = np.zeros((Steps))
    cov_stored_local = np.zeros((6, 6, Steps))

    for i in range(Steps):
        indx1 = int(Steps1[-(i+1)]) + timestep*2*Steps
        indx2 = int(Steps2[i]) + timestep*2*Steps

        R_true = Rdata[:, :, indx1].T @ Rdata[:, :, indx2]
        dx_true = Rdata[:, :, indx1].T @ (Xcenter[:, indx2:indx2+1] - Xcenter[:, indx1:indx1+1])
        eta_true = linAlg.R2eta(R_true)
        dx_stored_local[:, i:i+1] = dx_true

        ydata1_b = ydata_b[:, indx1*Narray:(indx1+1)*Narray]
        ydata2_b = ydata_b[:, indx2*Narray:(indx2+1)*Narray]
        Ydata_b = np.hstack((ydata1_b, ydata2_b))

        if initial_condition == 'Zero':
            init = np.vstack((dx_true, eta_true)) * 0
        elif initial_condition == 'True':
            init = np.vstack((dx_true, eta_true))

        try:
            dx_est, R_est, cov = magArray.poseEstArrayWLS(init, Ydata_b, m)
        except LinAlgError:
            dx_est = np.zeros(dx_est.shape)
            R_est = np.eye(3)
            cov = np.ones(cov.shape)
            StoreLinAlgError_local[i] = 1

        cov_stored_local[:, :, i] = cov
        dx_est_stored_local[:, i:i+1] = dx_est
        eta_error_stored_local[:, i:i+1] = linAlg.R2eta(R_true.T @ R_est)

    return dx_stored_local, dx_est_stored_local, eta_error_stored_local, StoreLinAlgError_local, cov_stored_local


''' Initialise parameters '''
np.random.seed(0)
theta_init = np.array([0.15**2, 5**2, 0.0012, 15**2])
theta = theta_init

magnetometerNumbers = [
    list(range(30)),
    list(range(0, 30, 2)),
    list(range(1, 30, 2))
]
magNames = ['Full', 'Even', 'Odd']
Scenarios = ['tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']

for magnetometers, magName in zip(magnetometerNumbers, magNames):
    for scenario in Scenarios:
        Narray = len(magnetometers)
        Rho = magArray.magArrayPos(magnetometers)

        ''' Set trim indices depending on the scenario '''
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
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare1noRot':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare2noRotLow':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare2noRot':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare3noRotLow':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare3noRot':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare1Rot':
            TrimStart = 3000
            TrimEnd = 3000
        elif scenario == 'tinySquare2Rot':
            TrimStart = 3000
            TrimEnd = 3000
        elif scenario == 'tinySquare3Rot':
            TrimStart = 3000
            TrimEnd = 3000

        TrimSlice = 1
        TakeSlices = [400, 1250]  # Visual inspection

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

        ''' Load and (pre)process data '''
        my_path = os.getcwd()
        my_data = 'ArrayData'
        my_file = 'calibrationBefore2_results.mat'

        Data = np.load(os.path.join(my_path, my_data, scenario + 'SigmaY.npy'))
        m['SigmaY'] = Data[:, :, magnetometers]

        Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
        m['mag_D'], m['mag_o'], SimgaYOld = magArray.processCalibrationData2(Data)

        my_file = scenario + '.mat'
        Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))

        ydata_b, ydata_n, Xcenter, Xdata, Rdata = magArray.preprocessMagData2(Data, m)

        Steps = 100
        TimeSteps = int(np.floor(Xcenter.shape[1]/(2*Steps)))
        Steps1 = np.linspace(0, Steps-1, Steps)
        Steps2 = np.linspace(Steps, 2*Steps-1, Steps)

        print(TimeSteps)

        '''  Visualise magnetic field magnitude '''
        ynorm = np.sqrt(ydata_b[0, :]**2 + ydata_b[1, :]**2 + ydata_b[2, :]**2)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Xdata[0, :], -Xdata[1, :], c=ynorm)
        cbar = plt.colorbar(orientation='vertical')
        cbar.set_label(r'$[\mu T]$') 
        plt.show()

        ''' Run parallel estimation over time '''
        num_jobs = 3

        Initial_conditions = ['Zero']
        for initial_condition in Initial_conditions:
            cov_stored = np.zeros((6, 6, Steps, TimeSteps))
            dx_est_stored = np.zeros((3, Steps, TimeSteps))
            eta_error_stored = np.zeros((3, Steps, TimeSteps))
            dx_stored = np.zeros((3, Steps, TimeSteps))
            StoreLinAlgError = np.zeros((Steps, TimeSteps))

            results = Parallel(n_jobs=num_jobs)(
                delayed(process_timestep)(timestep) for timestep in range(TimeSteps)
            )

            for i, (dx_stored_local, dx_est_stored_local, eta_error_stored_local, StoreLinAlgError_local, cov_stored_local) in enumerate(results):
                eta_error_stored[:, :, i] = eta_error_stored_local
                dx_est_stored[:, :, i] = dx_est_stored_local
                dx_stored[:, :, i] = dx_stored_local
                cov_stored[:, :, :, i] = cov_stored_local
                StoreLinAlgError[:, i] = StoreLinAlgError_local

            sigma_dx = np.zeros((3, TimeSteps))
            sigma_eta = np.zeros((3, TimeSteps))
            for timestep in range(TimeSteps):
                for i in range(3):
                    errors_dx = dx_stored[i, :, timestep] - dx_est_stored[i, :, timestep]
                    errors_eta = eta_error_stored[i, :, timestep]
                    sigma_dx[i, timestep] = np.sqrt(np.sum(errors_dx**2)/len(errors_dx))
                    sigma_eta[i, timestep] = np.sqrt(np.sum(errors_eta**2)/len(errors_eta))

            ''' Plot results '''
            cmap = plt.cm.viridis(np.linspace(0, 1, TimeSteps))
            fig, axs = plt.subplots(3, 2, figsize=(12, 6))

            for timestep, color in zip(range(TimeSteps), cmap):
                dx_error = np.abs(dx_stored[:, :, timestep] - dx_est_stored[:, :, timestep])
                for i, ax in enumerate(axs[:, 0]):
                    if i == 2:
                        ax.set_xlabel(r"$||\Delta \mathbf{p}||_2/l [-]$")
                    ax.set_ylabel(f"$\epsilon_{i+1} [mm]$")
                    dx_plot = np.sqrt(np.sum(dx_stored[:, :, timestep]**2, axis=0))
                    ax.scatter(dx_plot/np.sqrt(theta[0]), dx_error[i, :]*1000, s=15, alpha=0.75, c=[color])
                    ax.set_xscale('log')
                    ax.set_xlim([np.min(dx_plot/np.sqrt(theta[0])), 5])
                    ax.set_yscale('log')

            for timestep, color in zip(range(TimeSteps), cmap):
                eta_error = np.abs(eta_error_stored[:, :, timestep])
                for i, ax in enumerate(axs[:, 1]):
                    if i == 2:
                        ax.set_xlabel(r"$||\Delta \mathbf{p}||_2/l [-]$")
                    ax.set_ylabel(f"$\eta_{i+1} [°]$")
                    dx_plot = np.sqrt(np.sum(dx_stored[:, :, timestep]**2, axis=0))
                    ax.scatter(dx_plot/np.sqrt(theta[0]), eta_error[i, :]*180/np.pi, s=15, alpha=0.75, c=[color])
                    ax.set_xscale('log')
                    ax.set_xlim([np.min(dx_plot/np.sqrt(theta[0])), 5])
                    ax.set_yscale('log')

            plt.tight_layout()

            ''' Save results '''
            my_file = scenario + initial_condition + magName + '.npz'
            np.savez(
                os.path.join(my_path, my_data, my_file),
                dx_stored=dx_stored,
                dx_est_stored=dx_est_stored,
                eta_error_stored=eta_error_stored,
                cov_stored=cov_stored,
                StoreLinAlgError=StoreLinAlgError
            )