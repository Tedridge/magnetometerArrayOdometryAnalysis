import numpy as np
import os

import linAlg as linAlg
import magArray as magArray
import helper as helper
import motionModels as motionModels
import GP as GP

from scipy import linalg as alg

np.random.seed(0)

''' Settings '''
Din = 3
Ncenter = 2
Narray = 25
SizeArray = 4
SensorSeparation = SizeArray / 4

Array_Shape = 'Square'
Rho_base = magArray.shape(Array_Shape, Narray, SensorSeparation)
Rho = Rho_base

Rbn = linAlg.identityArray(3, (2,))

''' GP Hyperparameters '''
theta = np.array([0.15**2, 5**2, 0.0012, 15**2])

''' Create dictionary of parameters '''
m = {
    "theta": theta,
    "Rho": Rho,
    "Din": Din,
    "Ncenter": Ncenter,
    "Narray": Narray,
    "SizeArray": SizeArray,
}

''' Simulation Settings '''
Nsim = 50
Nmc = 5
Xrange = np.logspace(-4, 3, Nsim, base=10.0)

''' Main Simulation Loop '''
for direction in np.array([0, 2]):
    cov_stored = np.zeros((6, 6, Nmc, Nsim))
    cov_stored_single_fused = np.zeros((6, 6, Nmc, Nsim))
    cov_stored_single_mean = np.zeros((6, 6, Nmc, Nsim))

    ''' Set step direction '''
    step = np.zeros((3, 1))
    step[direction, 0] = 0.005
    m['step'] = step

    for SimulationNumber in range(1, Nsim + 1):
        for MonteCarloNumber in range(Nmc):
            
            ''' Rescale sensor array '''
            Rho = np.sqrt(Xrange[SimulationNumber-1]) * Rho_base
            m['Rho'] = Rho

            ''' Generate measurements '''
            XarrayTrue = np.hstack((Rho, step + Rbn[:, :, 1].T @ Rho))
            ydatacurl, fdatacurl, fcrosscurl = GP.datagenCurlFreeSim(XarrayTrue, np.zeros((3, 1)), m)

            ''' Normal to body frame '''
            ydatacurl[:, :Narray] = Rbn[:, :, 0] @ ydatacurl[:, :Narray]
            ydatacurl[:, Narray:] = Rbn[:, :, 1] @ ydatacurl[:, Narray:]

            ''' Estimate pose covariance '''
            init = np.vstack((step, linAlg.R2eta(Rbn[:, :, 1])))
            cov_pose = magArray.poseEstArraySim(init, ydatacurl, m)
            cov_stored[:, :, MonteCarloNumber, SimulationNumber-1] = cov_pose

            (cov_pose_single_fused, cov_pose_single_mean) = magArray.poseEstSingle(init, ydatacurl, m)
            cov_stored_single_fused[:, :, MonteCarloNumber, SimulationNumber-1] = cov_pose_single_fused
            cov_stored_single_mean[:, :, MonteCarloNumber, SimulationNumber-1] = cov_pose_single_mean

    ''' Save data '''
    my_path = os.getcwd()
    my_data = 'Data'

    if direction == 0:
        my_file = 'MC_step_x.npz'
    elif direction == 1:
        my_file = 'MC_step_y.npz'
    else:
        my_file = 'MC_step_z.npz'

    print(direction, my_file)

    np.savez(
        os.path.join(my_path, my_data, my_file),
        cov_stored=cov_stored,
        cov_stored_single_fused=cov_stored_single_fused,
        cov_stored_single_mean=cov_stored_single_mean,
        Xrange=Xrange/np.sqrt(m['Narray']-1)/np.sqrt(theta[0]),
        theta=theta
    )
