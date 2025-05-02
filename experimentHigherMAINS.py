import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import magArray as magArray
import linAlg as linAlg
import helper as helper
import GP as GP
from joblib import Parallel, delayed
from scipy.linalg import LinAlgError



''' Define a function to be executed in parallel '''
def processTimestep(timestep):
    dx_stored_local = np.zeros((3, Steps))
    dx_est_stored_local = np.zeros((3, Steps))
    R_est_stored_local = np.zeros((3, 3, Steps))
    R_true_stored_local = np.zeros((3, 3, Steps))

    eta_error_stored_local = np.zeros((3, Steps))
    StoreLinAlgError_local = np.zeros((Steps))
    cov_stored_local = np.zeros((6, 6, Steps))

    posStored = np.zeros((3, Narray, Steps))
    yStored = np.zeros((3, Narray, Steps))

    for i in range(Steps):
        indx1 = timestep*Steps+i
        indx2 = indx1+1

        R_true = Rdata[:, :, indx1].T @ Rdata[:, :, indx2]
        dx_true = Rdata[:, :, indx1].T @ (Xcenter[:, indx2:indx2+1] - Xcenter[:, indx1:indx1+1])
        eta_true = linAlg.R2eta(R_true)
        dx_stored_local[:, i:i+1] = dx_true
        R_true_stored_local[:, :, i] = R_true
        ydata1_b = ydata_b[:, indx1*Narray:(indx1+1)*Narray]
        ydata2_b = ydata_b[:, indx2*Narray:(indx2+1)*Narray]
        Ydata_b = np.hstack((ydata1_b, ydata2_b))

        posStored[:, :, i] = Xdata[:, indx1*Narray:(indx1+1)*Narray] 
        yStored[:, :, i] = ydata_b[:, indx1*Narray:(indx1+1)*Narray]

        if initial_condition == 'Zero':
            init = np.vstack((dx_true, eta_true))*0
        elif initial_condition == 'True':
            init = np.vstack((dx_true, eta_true))
        try:
            dx_est, R_est, cov = magArray.poseEstArrayWLS(init, Ydata_b, m)
        except LinAlgError:
            dx_est = np.zeros((3, 1))
            R_est = np.eye(3)
            cov = np.ones((6, 6))
            StoreLinAlgError_local[i] = 1

        R_est_stored_local[:, :, i] = R_est
        cov_stored_local[:, :, i] = cov
        dx_est_stored_local[:, i:i+1] = dx_est
        eta_error_stored_local[:, i:i+1] = linAlg.R2eta(R_true.T @ R_est)

    return dx_stored_local, dx_est_stored_local, eta_error_stored_local, R_est_stored_local, R_true_stored_local, StoreLinAlgError_local, cov_stored_local, posStored, yStored




''' Initialise parameters '''
np.random.seed(0)
theta = np.array([.15**2, 1**2, 0.0012, 15**2])

magmagnetometersNumbers = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
# magmagnetometersNumbers = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#                         [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
#                         [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]]
# magNames = ['Full', 'Even', 'Odd']
magNames = ['Full']
# magnetometers = 
# magnetometers = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# magnetometers = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]
# magnetometers = [0, 1, 2, 6, 7, 8, 12, 13, 14]
# magnetometers = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29]

# Scenarios = ['tinySquare1noRotLow', 'tinySquare2noRotLow', 'tinySquare3noRotLow', 'tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']
# Scenarios = ['bigsquare1normalheight']
Scenarios = ['bigsquare1']
# Scenarios = ['tinySquare3noRotLow', 'tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']
# Scenarios = ['tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']
# Scenarios = ['tinySquare1noRot', 'tinySquare1noRotLow', 'tinySquare1Rot']
#Scenarios = ['bigsquare1', 'bigsquare1normalheight', 'bigsquare1tilted']
# Scenarios = ['tinySquare1noRot', 'tinySquare2noRot', 'tinySquare3noRot']

for magnetometers, magName in zip(magmagnetometersNumbers, magNames):
    for scenario in Scenarios:
        Narray = len(magnetometers)
        Rho = magArray.magArrayPos(magnetometers)
        print(Narray)
        # TrimStart = 10000
        # TrimEnd = 18500
        if scenario == 'bigsquare1':
            TrimStart = 6250
            # #TrimEnd = 17000-1
            TrimEnd = 15750
            # TrimStart = 10000
            #TrimEnd = 17000-1
            # TrimEnd = 16000-1
        elif scenario == 'bigsquare2':
            TrimStart = 6000
        # #TrimEnd = 17000-1
            TrimEnd = 18000-1
            # TrimStart = 10000
            #TrimEnd = 17000-1
            # TrimEnd = 16000-1
        elif scenario == 'bigsquare3':
            TrimStart = 6000
        # #TrimEnd = 17000-1
            TrimEnd = 22000-1
            # TrimStart = 10000
            #TrimEnd = 17000-1
            # TrimEnd = 16000-1
        elif scenario == 'bigsquare1normalheight':
            TrimStart = 4000
            TrimEnd = 10500
        elif scenario == 'bigsquare1tilted':
            TrimStart = 5000
            TrimEnd = 10750
        elif scenario == 'tinySquare1noRotLow':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare1noRot':
            TrimStart = 2000
            TrimEnd = 2000
        elif scenario == 'tinySquare2noRotLow':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare2noRot':
            TrimStart = 2000
            TrimEnd = 2000
        elif scenario == 'tinySquare3noRotLow':
            TrimStart = 2500
            TrimEnd = 2000
        elif scenario == 'tinySquare3noRot':
            TrimStart = 2000
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

        TrimSlice = 10
        TakeSlices = [400, 1250] #Visual inspection

        m = {
            "Narray": Narray,
            "magnetometers": magnetometers,
            "theta": theta,
            "thetaInit": theta,
            "Rho": Rho,
            "TrimStart": TrimStart,
            "TrimEnd": TrimEnd,
            "TrimSlice": TrimSlice,
            "TakeSlices": TakeSlices,
            }


        my_path = os.getcwd()
        my_data = 'ArrayData'
        my_file = 'CalibrationMAINS.mat'
        Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
        m['mag_D'], m['mag_o'], _ = magArray.processCalibrationData(Data)
        m['SigmaY'] = linAlg.identityArray(3, np.array([Narray]))*theta[2]

        my_file = scenario + '.mat'
        Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
        # ydata_b, ydata_n, Xcenter, Xdata, Rdata = preprocess_magdata(Data, m)

        # my_path = os.getcwd()
        # my_data = 'ArrayData'
        # #my_file = 'CalibrationMAINS.mat'
        # my_file = 'calibrationBefore2_results.mat'
        
        # Data = np.load(os.path.join(my_path, my_data, scenario + 'SigmaY.npy'))
        # # m['SigmaY'] = Data[:, :, magnetometers]
        # m['SigmaY'] = linAlg.identityArray(3, np.array([Narray]))*theta[2]

        # Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
        # m['mag_D'], m['mag_o'], SigmaOLD = magArray.processCalibrationData2(Data)

        my_file = scenario + '.mat'
        Data = scipy.io.loadmat(os.path.join(my_path, my_data, my_file))
        ydata_b, ydata_n, Xcenter, Xdata, Rdata = magArray.preprocessMagData(Data, m)



        Steps = 50
        TimeSteps = int(np.floor(Xcenter.shape[1]/(Steps)))
        Steps1 = np.linspace(0, Steps-1, Steps)
        Steps2 = np.linspace(Steps, 2*Steps-1, Steps)
        print(TimeSteps)

        ynorm = np.sqrt(ydata_b[0, :]**2 + ydata_b[1, :]**2 + ydata_b[2, :]**2)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Xdata[0, :], -Xdata[1, :], c=ynorm)
        cbar = plt.colorbar(orientation='vertical')
        cbar.set_label(r'$[\mu T]$') 
        plt.show()
        print(Xcenter.shape)


        ''' Define the number of parallel jobs to run ''' 
        num_jobs = 3  # Use # available CPU cores

        # Initial_conditions = ['True']
        Initial_conditions = ['Zero']
        for initial_condition in Initial_conditions:
            cov_stored = np.zeros((6, 6, Steps, TimeSteps))
            dx_est_stored = np.zeros((3, Steps, TimeSteps))
            eta_error_stored = np.zeros((3, Steps, TimeSteps))
            dx_stored = np.zeros((3, Steps, TimeSteps))
            R_true_stored = np.zeros((3, 3, Steps, TimeSteps))
            R_est_stored = np.zeros((3, 3, Steps, TimeSteps))
            StoreLinAlgError = np.zeros((Steps, TimeSteps))
            posStoredBig = np.zeros((3, Narray, Steps, TimeSteps))
            yStoredBig = np.zeros((3, Narray, Steps, TimeSteps))

            ''' Run the function in parallel on the range of timesteps '''
            results = Parallel(n_jobs=num_jobs)(delayed(processTimestep)(timestep) for timestep in range(TimeSteps))
            for i, (dx_stored_local, dx_est_stored_local, eta_error_stored_local, R_est_stored_local, R_true_stored_local, StoreLinAlgError_local, cov_stored_local, posStored, yStored) in enumerate(results):
                eta_error_stored[:, :, i] = eta_error_stored_local
                dx_est_stored[:, :, i] = dx_est_stored_local
                dx_stored[:, :, i] = dx_stored_local
                R_true_stored[:, :, :, i] = R_true_stored_local
                R_est_stored[:, :, :, i] = R_est_stored_local
                cov_stored[:, :, :, i] = cov_stored_local
                StoreLinAlgError[:, i] = StoreLinAlgError_local
                posStoredBig[:, :, :, i] = posStored
                yStoredBig[:, :, :, i] = yStored

            sigma_dx = np.zeros((3, TimeSteps))
            sigma_eta = np.zeros((3, TimeSteps))
            for timestep in range (TimeSteps):
                for i in range(3):
                    errors_dx = dx_stored[i, :, timestep] - dx_est_stored[i, :, timestep]
                    errors_eta = eta_error_stored[i, :, timestep]
                    sigma_dx[i, timestep] = np.sqrt(np.sum(errors_dx**2)/len(errors_dx))
                    sigma_eta[i, timestep] = np.sqrt(np.sum(errors_eta**2)/len(errors_eta))

            cmap = plt.cm.viridis(np.linspace(0, 1, TimeSteps))
            fig, axs = plt.subplots(3, 2, figsize=(12, 6))

            ''' Plot positional error data '''
            for timestep, color in zip(range(TimeSteps), cmap):
                dx_error = dx_stored[:, :, timestep] - dx_est_stored[:, :, timestep]
                for i, ax in enumerate(axs[:, 0]):
                    if i == 2:
                        ax.set_xlabel(r"$||\Delta \mathbf{p}||_2 [-]$")
                    ax.set_ylabel(f"$\epsilon_{i+1} [mm]$")  # Add timestep to ylabel
                    dx_plot = np.sqrt(dx_stored[0, :, timestep]**2 + dx_stored[1, :, timestep]**2 + dx_stored[2, :, timestep]**2)
                    ax.scatter(dx_plot, dx_error[i, :]*1000, s=15, alpha=0.75, c=[color])
                    ax.set_xscale('log')
                    #ax.set_xlim([np.min(dx_plot/np.sqrt(theta[0])), 5])
                    # ax.set_yscale('log')

            ''' Plot rotational error data '''
            for timestep, color in zip(range(TimeSteps), cmap):
                eta_error = eta_error_stored[:, :, timestep]
                for i, ax in enumerate(axs[:, 1]):
                    if i == 2:
                        ax.set_xlabel(r"$||\Delta \mathbf{p}||_2 [-]$")
                    ax.set_ylabel(f"$\eta_{i+1} [°]$")  # Add timestep to ylabel
                    dx_plot = np.sqrt(dx_stored[0, :, timestep]**2 + dx_stored[1, :, timestep]**2 + dx_stored[2, :, timestep]**2)
                    ax.scatter(dx_plot, eta_error[i, :]*180/np.pi, s=15, alpha=0.75, c=[color])
                    ax.set_xscale('log')
                    #ax.set_xlim([np.min(dx_plot/np.sqrt(theta[0])), 5])
                    # ax.set_yscale('log')

            plt.tight_layout()

            # # Show the plot
            # plt.show()
            # my_path = os.getcwd()
            # my_figures = 'Figures'
            # my_file = 'ArrayNewErrordxdRInitTrue.pdf'
            # fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf')

            my_file = 'WalkPath' + scenario + initial_condition + magName + '.npz'
            np.savez(os.path.join(my_path, my_data, my_file),Xcenter=Xcenter, Xdata=Xdata, Rdata=Rdata, dx_stored=dx_stored, dx_est_stored=dx_est_stored, R_est_stored=R_est_stored, R_true_stored=R_true_stored, eta_error_stored=eta_error_stored, cov_stored=cov_stored, StoreLinAlgError=StoreLinAlgError, ydata_n=ydata_n, posStoredBig=posStoredBig, yStoredBig=yStoredBig)




