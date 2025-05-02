import numpy as np
import matplotlib.pyplot as plt
import os

import magArray as magArray
import linAlg as linAlg
import helper as helper
import motionModels as motionModels
import GP as GP

from scipy import linalg as alg

np.random.seed()

''' Settings '''
Din = 3
Narray = 64
Nsize = 50

''' GP Hyperparameters '''
theta = np.array([0.15**2, 5**2, 0.0012, 15**2])

''' Monte Carlo Simulation Settings '''
Nmc = 50
Ndirection = 2
Nsim = 3

cov_stored = np.zeros((6, 6, Nmc, Nsim, Nsize, Ndirection))
cov_stored_single = np.zeros((6, 6, Nmc, Nsim))
arraySizes = np.logspace(-4, 2, Nsize, base=10)

Rbn = linAlg.identityArray(3, (2,))

''' Main Simulation Loop '''
for direction in range(Ndirection):
    for arraySizeNumber, arraySize in enumerate(arraySizes):
        
        ''' Step direction '''
        step = np.zeros((3, 1))
        if direction == 0:
            step[0, 0] = 0.05
        elif direction == 1:
            step[2, 0] = 0.05

        ''' Create dictionary of parameters '''
        m = {
            "theta": theta,
            "Din": Din,
            "Narray": Narray,
            "step": step,
        }

        ''' Monte Carlo and Simulation Runs '''
        for MonteCarloNumber in range(Nmc):
            for simulationNumber in range(Nsim):
                
                ''' Different sensor configurations '''
                if simulationNumber == 0:
                    Rho = magArray.shape('Cube', Narray, arraySize * 3 / 7)
                elif simulationNumber == 1:
                    Rho = magArray.shape('Square', Narray, arraySize)
                elif simulationNumber == 2:
                    Rho = magArray.shape('Line', Narray, arraySize * 9)

                m["Rho"] = Rho

                ''' Generate measurements '''
                XarrayTrue = np.hstack((Rho, step + Rbn[:, :, 1].T @ Rho))
                ydatacurl, fdatacurl, fcrosscurl = GP.datagenCurlFreeSim(XarrayTrue, np.zeros((3, 1)), m)

                ''' Normal to body frame '''
                ydatacurl[:, :Narray] = Rbn[:, :, 0] @ ydatacurl[:, :Narray]
                ydatacurl[:, Narray:] = Rbn[:, :, 1] @ ydatacurl[:, Narray:]

                ''' Estimate pose covariance '''
                init = np.vstack((step, linAlg.R2eta(Rbn[:, :, 1])))
                cov_pose = magArray.poseEstArraySim(init, ydatacurl, m)

                ''' Store covariance results '''
                cov_stored[:, :, MonteCarloNumber, simulationNumber, arraySizeNumber, direction] = cov_pose

''' Save data '''
my_path = os.getcwd()
my_data = 'Data'
my_file = 'MC_configuration_3d.npz'
np.savez(os.path.join(my_path, my_data, my_file), cov_stored=cov_stored, sensorSeparation=arraySizes/4)
