import numpy as np
import matplotlib.pyplot as plt
import os

import magArray as magArray
import GP as GP
import motionModels as motionModels
import linAlg as linAlg
import motionModels as motionModels

np.random.seed()

''' Settings '''
Din = 3
Din2 = 2
Dx = 6
Ncenter = 2
Narray = 25
Npred = 1
Ncross = 1
yawrate = 0
Iterations = 1

SizeArray = 0.15
SensorSeparation = SizeArray/4

Array_Shape = 'Square'
Rho = magArray.shape(Array_Shape, Narray, SensorSeparation)

theta = np.array([0.15**2, 5**2, 0.0012, 15**2])  

''' Create dictionary to store parameters '''
m = {
    "theta": theta,
    "Rho": Rho,
    "Din": Din,
    "Din2": Din2,
    "Dx": Dx,
    "Ncenter": Ncenter,
    "Narray": Narray,
    "Npred": Npred,
    "Ncross": Ncross,
    "SizeArray": SizeArray,
    "yawrate": yawrate,
    "Iterations": Iterations,
}

''' Simulation setup '''
Nsim = 50
Nmc = 100

cov_stored = np.zeros((6, 6, Nmc, Nsim))
cov_stored_single_fused = np.zeros((6, 6, Nmc, Nsim))
cov_stored_single_mean = np.zeros((6, 6, Nmc, Nsim))

Sigmax = np.zeros((3, 3))
SigmaR = np.zeros((3, 3))

Xrange = np.logspace(-5, np.log10(1), Nsim, base=10)

Rbn = linAlg.identityArray(3, (2,))

for direction in np.array([0, 2]):
    stepBase = np.zeros((3, 1))
    stepBase[direction, 0] = 1
    for SimulationNumber in range(1, Nsim+1):  
        for MonteCarloNumber in range(Nmc):
            step = stepBase*Xrange[SimulationNumber-1]
            m['step'] = step
            
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

    ''' Save the results '''
    my_path = os.getcwd()
    my_data = 'Data'
    if direction == 0: 
        my_file = 'MC_step_distance_x.npz'
    elif direction == 1: 
        my_file = 'MC_step_distance_y.npz'
    else: 
        my_file = 'MC_step_distance_z.npz'

    np.savez(
        os.path.join(my_path, my_data, my_file),
        cov_stored=cov_stored,
        cov_stored_single_fused=cov_stored_single_fused,
        cov_stored_single_mean=cov_stored_single_mean,
        Xrange=Xrange,
        theta=theta
    )