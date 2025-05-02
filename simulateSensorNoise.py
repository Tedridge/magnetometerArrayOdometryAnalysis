import numpy as np
import os

import magArray as magArray
import linAlg as linAlg
import helper as helper
import motionModels as motionModels
import GP as GP

np.random.seed(0)

''' GP and system parameters '''
Din = 3
Ncenter = 2
Narray = 25

Start = -4
End = 4
Distance = 3

SizeArray = End
SensorSeparation = SizeArray / 4
Array_Shape = 'Square'

theta = np.array([0.15**2, 5**2, 0.0012, 15**2])
Rho = magArray.shape(Array_Shape, Narray, np.sqrt(theta[0]))

Rbn = linAlg.identityArray(3, (2,))

''' Store parameters in dictionary '''
m = {
    "theta": theta,
    "Rho": Rho,
    "Din": Din,
    "Ncenter": Ncenter,
    "Narray": Narray,
    "Start": Start,
    "End": End,
    "SizeArray": SizeArray,
}

''' Simulation settings '''
Nsim = 30
Nmc = 50
Xrange = np.logspace(-2, 2, Nsim, base=10)

''' Loop over x and z direction '''
for direction in np.array([0, 2]):
    cov_stored = np.zeros((6, 6, Nmc, Nsim))
    cov_stored_single_fused = np.zeros((6, 6, Nmc, Nsim))
    cov_stored_single_mean = np.zeros((6, 6, Nmc, Nsim))
    cov_stored_single_median = np.zeros((6, 6, Nmc, Nsim))

    step = np.zeros((3, 1))
    step[direction, 0] = 0.05
    m['step'] = step

    for SimulationNumber in range(1, Nsim + 1):
        theta[2] = Xrange[SimulationNumber - 1]**2
        m['theta'] = theta

        for MonteCarloNumber in range(Nmc):
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
        my_file = 'MC_step_noise_x.npz'
    elif direction == 1:
        my_file = 'MC_step_noise_y.npz'
    else:
        my_file = 'MC_step_noise_z.npz'

    print(direction, my_file)

    np.savez(
        os.path.join(my_path, my_data, my_file),
        cov_stored=cov_stored,
        cov_stored_single_fused=cov_stored_single_fused,
        cov_stored_single_mean=cov_stored_single_mean,
        Xrange=np.sqrt(theta[1]) / Xrange,
        theta=theta,
    )
