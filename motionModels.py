import numpy as np
import linAlg as linAlg
import magArray as magArray

def motionModelStep(Sigmax, SigmaR, m):
    e_x, e_eta = magArray.errorVectors(Sigmax, SigmaR, m)
    Ncenter = m["Ncenter"]
    stepsize = m["stepsize"]
    x = np.zeros((3, Ncenter))
    xerror = np.zeros((3, Ncenter))
    Sx_R = np.zeros((3, 3, e_eta.shape[1]))
    SR = np.zeros((3, 3, e_eta.shape[1]))
    R = np.zeros((3, 3, Ncenter))
    R[:, :, 0] = np.eye(3)
    step = np.array([[stepsize], [0], [0]])
    yaw = 0
    for i in range(1, Ncenter):
        R[:, :, i] = linAlg.Rz(yaw)
        dx = R[:, :, i] @ step
        x[:, i] = x[:, i - 1] + dx[:, 0]

        Re = linAlg.so3Rodrigues(e_eta[:, i]) @ R[:, :, i]
        dxe = Re @ step + e_x[:, i]

        xerror[:, i] = xerror[:, i - 1] + dxe[:, 0]
        A = linAlg.crossVector(dx)
        Sx_R[:, :, i] = A @ SigmaR @ A.T + Sx_R[:, :, i - 1] + Sigmax
        SR[:, :, i] = SigmaR

    e_x = x - xerror
    return x, xerror, R, SR, Sx_R, e_x, e_eta

def motionModel1Step(Sigmax, SigmaR, m):    
    e_x, e_eta = magArray.errorVectors(Sigmax, SigmaR, m)
    Ncenter = m["Ncenter"]
    step = m["step"]
    x = np.zeros((3, Ncenter))
    xerror = np.zeros((3, Ncenter))
    Sx_R = np.zeros((3, 3, e_eta.shape[1]))
    SR = np.zeros((3, 3, e_eta.shape[1]))
    R = np.zeros((3, 3, Ncenter))
    for i in range(Ncenter):
        R[:, :, i] = np.eye(3)
        SR[:, :, 1] = SigmaR
    Re = linAlg.so3Rodrigues(e_eta[:, 1]) @ R[:, :, 0]
    
    dx = step
    dxe = dx #Re @ dx

    e_x[:, 1:2] += dxe - dx

    x[:, 1] = x[:, 0] + dx[:, 0]
    

    xerror[:, 1] = xerror[:, 0] + dxe[:, 0] 
    A = linAlg.crossVector(dx)
    Sx_R[:, :, 1] = Sigmax #+ A @ SigmaR @ A.T

    return x, xerror, R, SR, Sx_R, e_x, e_eta

def motionModel(Sigmax, SigmaR, m):
    e_x, e_eta = magArray.errorVectors(Sigmax, SigmaR, m)
    Ncenter = m["Ncenter"]
    yawrate = m["yawrate"]
    stepsize = m["stepsize"]
    x = np.zeros((3, Ncenter))
    xerror = np.zeros((3, Ncenter))
    yaw = 0
    Sx_R = np.zeros((3, 3, e_eta.shape[1]))
    SR = np.zeros((3, 3, e_eta.shape[1]))
    R = np.zeros((3, 3, Ncenter))
    R[:, :, 0] = np.eye(3)
    step = np.array([[stepsize], [0], [0]])
    for i in range(1, Ncenter):
        #yaw += yawrate
        R[:, :, i] = linAlg.Rz(yaw)
        dx = R[:, :, i] @ step
        x[:, i] = x[:, i - 1] + dx[:, 0]

        Re = linAlg.so3Rodrigues(e_eta[:, i]) @ R[:, :, i]
        dxe = Re @ step + e_x[:, i]

        xerror[:, i] = xerror[:, i - 1] + dxe[:, 0]
        A = linAlg.crossVector(dx)
        #Sx_R[:, :, i] = A @ SigmaR @ A.T + Sx_R[:, :, i - 1] + Sigmax
        Sx_R[:, :, i] = Sigmax
        SR[:, :, i] = SigmaR

    return x, xerror, R, SR, Sx_R, e_x, e_eta

def motionModelOnlyRotation(Sigmax, SigmaR, m):
    e_x, e_eta = magArray.errorVectors(Sigmax, SigmaR, m)
    e_x = e_x * 0
    Ncenter = m["Ncenter"]
    yawrate = m["yawrate"]
    stepsize = m["stepsize"]
    x = np.zeros((3, Ncenter))
    xerror = np.zeros((3, Ncenter))
    yaw = 0
    Sx_R = np.zeros((3, 3, e_eta.shape[1]))
    SR = np.zeros((3, 3, e_eta.shape[1]))
    R = np.zeros((3, 3, Ncenter))
    R[:, :, 0] = np.eye(3)
    step = np.array([[stepsize], [0], [0]])
    for i in range(1, Ncenter):
        yaw += yawrate
        R[:, :, i] = linAlg.Rz(yaw)
        dx = R[:, :, i] @ step
        x[:, i] = x[:, i - 1] + dx[:, 0]

        Re = R[:, :, i]
        dxe = Re @ step

        xerror[:, i] = xerror[:, i - 1] + dxe[:, 0]
        A = linAlg.crossVector(dx)
        SR[:, :, i] = SigmaR

    return x, xerror, R, SR, Sx_R, e_x, e_eta


def motionModelLshape(steps, stepSize, indxNumber, m):
    Rho = m['Rho']
    Narray = m['Narray']
    Xcenter = np.zeros((3, steps+1))
    Rb2n = np.zeros((3, 3, steps+1))  

    x = np.zeros((3, 1))
    for indx in np.array([indxNumber]):
        for jndx in range(steps):
            step = np.zeros((3, 1))
            if indx == 0:
                step[0] = stepSize
            elif indx == 1:
                step[1] = stepSize
            elif indx == 2:
                step[2] = stepSize
            x += step
            Xcenter[:, jndx + 1] = x[:, 0]
            Rb2n[:, :, jndx + 1] = np.eye(3)
    Rb2n[:, :, 0] = np.eye(3)
    Xarray = np.kron(Xcenter, np.ones((1, Narray))) + np.kron(np.ones((1, steps+1)), Rho)
    return Xcenter, Xarray, Rb2n
