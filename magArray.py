import numpy as np
import linAlg as linAlg
import helper as helper
import GP as GP
import math
import sys
import matplotlib.pyplot as plt
import scipy.linalg as alg
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


def rho_n(Rdata, m):
    Rho = m["Rho"]
    Rhodata = np.zeros((3, 1))
    for i in range(Rdata.shape[2]):
        Rhodata = np.hstack((Rhodata, Rdata[:, :, i].T @ Rho))
    return Rhodata[:, 1:]

def Rho_n(Rdata, m):
    Rho = m["Rho"]
    Rhodata = np.zeros((3, 1))

    for i in range(Rdata.shape[2]):
        Rhodata = np.hstack((Rhodata, Rdata[:, :, i] @ Rho))
    return Rhodata[:, 1:]

''' Magnetometer array shapes '''
def GenerateArray(m):
    Narray = m["Narray"]
    SizeArray = m["SizeArray"]
    Din = m["Din"]

    # Generate the desired array
    if Din == 1:
        H = np.linspace(-SizeArray / 2, SizeArray / 2, Narray).reshape(1, -1)
    else:
        H = helper.gridpoints(SizeArray[1], -SizeArray[0], Narray, Din)
    return H

def shape(ArrayShape, Narray, SizeArray):
    if ArrayShape == 'Triangle':
        if Narray == 3:
            Rho = shapeTriangle(SizeArray)
            return Rho
        else:
            print('Array number =/= 3')
            sys.exit()
            return None
        
    elif ArrayShape == 'Square':
        square_root = math.sqrt(Narray)
        if square_root.is_integer():
            Rho = shapeSquare(SizeArray, Narray)
            return Rho
        else:
            print('Array number =/= square')
            sys.exit()
            return None    

    elif ArrayShape == 'Line':
        Rho = shapeLine(SizeArray, Narray)
        return Rho
    
    elif ArrayShape == 'Circle':
        Rho = shapeCircle(SizeArray, Narray)
        return Rho

    elif ArrayShape == 'Cube':
        Rho = ArrayCube(SizeArray, Narray)
        return Rho
    
def shapeTriangle(SizeArray):
    Rho = np.zeros((3, 3))
    theta = 0
    for i in range(3):
        Rho[:, i:i+1] = SizeArray*np.array([[np.sin(theta)], [np.cos(theta)], [0]])
        theta += np.pi*2/3
    return Rho

def shapeLine(SizeArray, Narray):
    Rho = np.zeros((3, Narray))
    Rho[0, :] = np.linspace(-SizeArray/2, SizeArray/2, Narray)
    return Rho

def shapeSquare(SizeArray, Narray):
    Rho = np.zeros((3, Narray))
    Rho = helper.gridpoints(int(np.sqrt(Narray)), 2, 3, -SizeArray/2, SizeArray/2)
    return Rho

def shapeCircle(Radius, Narray):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = Radius * np.cos(angles)
    y = Radius * np.sin(angles)
    z = np.zeros(N)
    points = np.array([x, y, z])
    return points

def shapeSphere(N, Radius):
    Ncount = 0
    a = 4 * np.pi ** 2 / N
    d = np.sqrt(a)
    M_theta = round(np.pi / d)
    d_theta = np.pi / M_theta
    d_phi = a / d_theta
    
    points = np.zeros((3, N))
    
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = round(2 * np.pi * np.sin(theta) / d_phi)
        
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            x = Radius * np.sin(theta) * np.cos(phi)
            y = Radius * np.sin(theta) * np.sin(phi)
            z = Radius * np.cos(theta)
            points[:, Ncount] = [x, y, z]
            Ncount += 1
            if Ncount == N:
                return points
    
    return points

def ArrayCube(SizeArray, Narray):
    Rho = helper.gridpoints(int(np.round(Narray**(1/3), 0)), 3, 3, -SizeArray/2, SizeArray/2)
    return Rho

def magArrayLocations(Din, Narray, SizeArray):
    S = helper.gridpoints(int(np.sqrt(Narray)), 2, Din, -SizeArray / 2, SizeArray / 2)
    return S

def magArrayReal():
    Rho = np.zeros((3, 30))
    side1 = 0.34
    side2 = 0.24
    x1 = np.linspace(-side1/2, side1/2, 6)
    x2 = np.linspace(-side2/2, side2/2, 5)
    X1, X2 = np.meshgrid(x1, x2)

    Rho[0:1, :] = X1.reshape(1, -1)
    Rho[1:2, :] = X2.reshape(1, -1)
    return Rho

def magArrayPos(magnetometers):
    #h1 = np.array([[-.16, -.096, -.032, .032, .096, .16]])
    #h2 = np.array([[.110, .055, 0, -.055, -.110]])
    #h1 = np.array((-.160, -.096, -.032, .032, .096, .160)).reshape(1, -1)
    #h2 = np.array((.110, .055, 0, -.055, -.110)).reshape(1, -1)
    h1 = np.array((-.150, -.1, -.05, .05, .1, .150)).reshape(1, -1)
    h2 = np.array((.1, .05, 0, -.05, -.1)).reshape(1, -1)
    
    H1 = np.kron(np.ones((1, 5)), h1)
    H2 = np.kron(h2, np.ones((1, 6)))
    H3 = np.zeros((1, 30))
    
    H4 = np.row_stack((H1, H2, H3))
    H = H4[:, magnetometers]
    return H

def magArray30():
    Rho = np.zeros((3, 30))
    l = 0.15
    side1 = 0.34/l
    side2 = 0.24/l
    x1 = np.linspace(-side1/2, side1/2, 6)
    x2 = np.linspace(-side2/2, side2/2, 5)
    X1, X2 = np.meshgrid(x1, x2)

    Rho[0:1, :] = X1.reshape(1, -1)
    Rho[1:2, :] = X2.reshape(1, -1)
    return Rho

''' (Pre)process data '''
def preprocessMagData(Data, m):
    mag_data = Data['data']['mag_array'][0,0]['field'][0][0]
    rot_data = Data['data']['gt'][0,0]['ori'][0][0]
    pos_data = Data['data']['gt'][0,0]['pos'][0][0]
    gt_t = Data['data']['gt'][0,0]['t'][0][0]
    mag_t = Data['data']['mag_array'][0,0]['t'][0][0]
    mag_data_int = np.zeros((90, gt_t.shape[1]))
    for i in range(90):
        cs = CubicSpline(mag_t[0, :], mag_data[:, i])
        mag_data_int[i, :] = cs(gt_t[0, :])
        #mag_data_int[i, :] = np.interp(gt_t[0, :], mag_t[0, :], mag_data[:, i])
    #mag_data_int[:, -6:] = np.tile(mag_data_int[:, -6], (6, 1)).T

    mag_D = m['mag_D']
    mag_o = m['mag_o']
    #m_n = data['m_n']
    Narray = m['Narray']
    #magnetometers = m['magnetometers']
    TrimStart = m['TrimStart']
    TrimEnd = m['TrimEnd']
    TrimSlice = m['TrimSlice']

    mag_data_int = mag_data_int[:, TrimStart:-TrimEnd]
    mag_data_int = mag_data_int[:, ::TrimSlice]
    pos_data = pos_data[:, TrimStart:-TrimEnd]
    pos_data = pos_data[:, ::TrimSlice]
    rot_data = rot_data[:, :, TrimStart:-TrimEnd]
    rot_data = rot_data[:, :, ::TrimSlice]
    
    ydata_b = np.zeros((3, Narray*pos_data.shape[1]))
    ydata_n = np.zeros((3, Narray*pos_data.shape[1]))

    for i in range(Narray):
        Ones = np.zeros((1, Narray))
        Ones[0, i] = 1
        y_i = linAlg.sinv(mag_D[:, :, i]) @ (mag_data_int[i*3:(i+1)*3, :] - mag_o[:, 0, i:i+1])
        #y_i = (mag_data_int[i*3:(i+1)*3, :] - mag_o[:, 0, i:i+1])
        ydata_b += np.kron(y_i, Ones)

    for i in range(rot_data.shape[2]):
        ydata_n[:, i*Narray:(i+1)*Narray] = rot_data[:, :, i] @ ydata_b[:, i*Narray:(i+1)*Narray]
    Rho_normal = Rho_n(rot_data, m)

    # data_mean = np.mean(ydata_n, 1).reshape(-1, 1)
    # ydata_n -= data_mean
    # for i in range(rot_data.shape[2]):
    #    ydata_b[:, i*Narray:(i+1)*Narray] = rot_data[:, :, i].T @ ydata_n[:, i*Narray:(i+1)*Narray]

    Xdata = np.kron(pos_data, np.ones((1, Narray)))
    Xdata += Rho_normal
    Xdata -= np.mean(Xdata, 1).reshape(-1, 1)
    return ydata_b, ydata_n, pos_data, Xdata, rot_data

def preprocessMagData2(Data, m):
    mag_data = Data['data']['mag_array'][0,0]['y'][0][0]
    rot_data = Data['data']['gt'][0,0]['Rb2n'][0][0]
    pos_data = Data['data']['gt'][0,0]['pos'][0][0]

    # gt_t = Data['data']['gt'][0,0]['t'][0][0]
    # mag_t = Data['data']['mag_array'][0,0]['t'][0][0]
    # mag_data_int = np.zeros((90, gt_t.shape[1]))
    # for i in range(90):
    #     cs = CubicSpline(mag_t[0, :], mag_data[:, i])
    #     mag_data_int[i, :] = cs(gt_t[0, :])
    #     #mag_data_int[i, :] = np.interp(gt_t[0, :], mag_t[0, :], mag_data[:, i])
    #mag_data_int[:, -6:] = np.tile(mag_data_int[:, -6], (6, 1)).T

    mag_D = m['mag_D']
    mag_o = m['mag_o']
    #m_n = data['m_n']
    Narray = m['Narray']
    magnetometers = m['magnetometers']
    TrimStart = m['TrimStart']
    TrimEnd = m['TrimEnd']
    TrimSlice = m['TrimSlice']

    mag_data_int = mag_data[:90, TrimStart:-TrimEnd]
    mag_data_int = mag_data_int[:, ::TrimSlice]
    pos_data = pos_data[:, TrimStart:-TrimEnd]
    pos_data = pos_data[:, ::TrimSlice]
    rot_data = rot_data[:, :, TrimStart:-TrimEnd]
    rot_data = rot_data[:, :, ::TrimSlice]
    
    ydata_b = np.zeros((3, Narray*pos_data.shape[1]))
    ydata_n = np.zeros((3, Narray*pos_data.shape[1]))

    indx = 0
    for magndx in magnetometers:
        Ones = np.zeros((1, Narray))
        Ones[0, indx] = 1
        y_i = linAlg.sinv(mag_D[:, :, magndx]) @ (mag_data_int[magndx*3:(magndx+1)*3, :] - mag_o[:, 0, magndx:magndx+1])
        #y_i = (mag_data_int[i*3:(i+1)*3, :] - mag_o[:, 0, i:i+1])
        ydata_b += np.kron(y_i, Ones)
        indx += 1

    for i in range(rot_data.shape[2]):
        ydata_n[:, i*Narray:(i+1)*Narray] = rot_data[:, :, i] @ ydata_b[:, i*Narray:(i+1)*Narray]
    Rho_normal = Rho_n(rot_data, m)

    # data_mean = np.mean(ydata_n, 1).reshape(-1, 1)
    # ydata_n -= data_mean
    # for i in range(rot_data.shape[2]):
    #    ydata_b[:, i*Narray:(i+1)*Narray] = rot_data[:, :, i].T @ ydata_n[:, i*Narray:(i+1)*Narray]

    Xdata = np.kron(pos_data, np.ones((1, Narray)))
    Xdata += Rho_normal
    Xdata -= np.mean(Xdata, 1).reshape(-1, 1)
    return ydata_b, ydata_n, pos_data, Xdata, rot_data

def preprocessMagdataOldArray(Data, m):
    mag_data = Data['mag_raw']
    rot_data = np.zeros((3, 3, 2))
    for i in range(2):
        rot_data[:, :, i] = np.eye(3)
    pos_data = np.zeros((3, 2))
    pos_data[:, 1:2] = np.array([[0.055], [0], [0]])
    #rot_data = Data['gyr_raw']
    #pos_data = Data['acc_raw']

    mag_D = m['mag_D']
    mag_o = m['mag_o']
    #m_n = data['m_n']
    Narray = m['Narray']
    magnetometers = m['magnetometers']
    TrimStart = m['TrimStart']
    TrimEnd = m['TrimEnd']

    mag_data = mag_data[:, m['TakeSlices']]
    #mag_data_int = mag_data_int[:, TrimStart:-TrimEnd]
    #mag_data_int = mag_data_int[:, ::TrimSlice]
    # pos_data = pos_data[:, TrimStart:-TrimEnd]
    # pos_data = pos_data[:, ::TrimSlice]
    # rot_data = rot_data[:, :, TrimStart:-TrimEnd]
    # rot_data = rot_data[:, :, ::TrimSlice]
    
    ydata_b = np.zeros((3, Narray*pos_data.shape[1]))
    ydata_n = np.zeros((3, Narray*pos_data.shape[1]))

    j = 0
    for i in magnetometers:
        Ones = np.zeros((1, Narray))
        Ones[0, j] = 1
        y_i = linAlg.sinv(mag_D[:, :, i]) @ (mag_data[i*3:(i+1)*3, :] - mag_o[:, 0, i:i+1])
        ydata_b += np.kron(y_i, Ones)
        j += 1

    for i in range(rot_data.shape[2]):
        ydata_n[:, i*Narray:(i+1)*Narray] = rot_data[:, :, i] @ ydata_b[:, i*Narray:(i+1)*Narray]
    Rho_normal = Rho_n(rot_data, m)

    data_mean = np.mean(ydata_n, 1).reshape(-1, 1)
    ydata_n -= data_mean
    for i in range(rot_data.shape[2]):
       ydata_b[:, i*Narray:(i+1)*Narray] = rot_data[:, :, i].T @ ydata_n[:, i*Narray:(i+1)*Narray]

    Xdata = np.kron(pos_data, np.ones((1, Narray)))
    Xdata += Rho_normal
    Xdata -= np.mean(Xdata, 1).reshape(-1, 1)

    return ydata_b, ydata_n, pos_data, Xdata, rot_data

def processCalibrationData(Data):
    Ds = Data['theta_all'][0:9, :]
    os = Data['theta_all'][9:12, :]
    #mag_noise = Data['setting']['magNoiseDensity'][0][0]
    mag_noise = 1.2*10**-2
    SigmaY = np.zeros((3, 3, 30))
    for i in range(30):
        SigmaY[:, :, i] = np.eye(3)*mag_noise
    D = np.zeros((3, 3, 30))
    o = np.zeros((3, 1, 30))
    for i in range(30):
        D[:, :, i] = Ds[:, i:i+1].reshape(3, 3).T
        o[:, :, i] = os[:, i:i+1].reshape(3, 1)
    
    Ddet = 0
    for i in range(30):
        Ddet += np.linalg.det(D[:, :, i])

    for i in range(30):
        D[:, :, i] = D[:, :, i]/((Ddet/30)**(1/3))
    return D, o, SigmaY

def processCalibrationData2(Data):
    #mag_noise = Data['setting']['magNoiseDensity'][0][0]
    mag_noise = 0.025**2
    SigmaY = np.zeros((3, 3, 30))
    for i in range(30):
        SigmaY[:, :, i] = np.eye(3)*mag_noise
    D = np.zeros((3, 3, 30))
    o = np.zeros((3, 1, 30))
    for i in range(30):
        D[:, :, i] = Data['D'][0][i]
        o[:, :, i] = Data['o'][0][i].reshape(3, 1)
    
    Ddet = 0
    for i in range(30):
        Ddet += np.linalg.det(D[:, :, i])

    for i in range(30):
        D[:, :, i] = D[:, :, i]/((Ddet/30)**(1/3))
    return D, o, SigmaY

def processCalibrationOldArray(Data):
    n = Data['D'].shape[1]
    D = np.zeros((3, 3, n))
    o = np.zeros((3, 1, n))
    for i in range(n):
        D[:, :, i] = Data['D'][0, i]
        o[:, :, i] = Data['o'][0, i]
    return D, o


''' Odometry '''
def poseEstArrayWLS(init, Y, m):
    Rho = m["Rho"]
    Narray = m["Narray"]

    y0 = Y[:, :Narray] 
    y1_b = Y[:, Narray:]
    Xdata = Rho
    Niterations = 5
    R_est = np.zeros((3, 3, Niterations))
    dx_est = np.zeros((3, Niterations))

    R_est[:, :, 0] = linAlg.expR(init[3:, 0:1]).T
    dx_est[:, 0:1] = init[:3, 0:1]
    beta = np.zeros((6, 1))
    StepSize = 1
    CostFunctionValue = np.array([])
    dxCostFunction = np.zeros((6, Niterations))
    for i in range(Niterations):
        if i > 0:
            R_est[:, :, i] = linAlg.expR(beta[3:, :]).T @ R_est[:, :, i-1]
            dx_est[:, i:i+1] = dx_est[:, i-1:i] + beta[:3, 0:1]
        dxCostFunction[:, i:i+1] = np.vstack((dx_est[:, i:i+1],  linAlg.R2eta(R_est[:, :, i])))

        Xpred = np.kron(np.ones((1, Narray)), dx_est[:, i:i+1]) + R_est[:, :, i] @ Rho

        K11 = GP.kernelCurlFreeLoop(Xdata, Xdata, m) + GP.kernelConstant(Xdata, Xdata, m)
        K21 = GP.kernelCurlFreeLoop(Xpred, Xdata, m) + GP.kernelConstant(Xpred, Xdata, m)
        K22 = GP.kernelCurlFreeLoop(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)
        dKx21, dKR21 = GP.nablaKernelCurlFreeLoop(Xpred, Xdata, R_est[:, :, i].reshape(3, 3, 1), m)
        
        SigmaY = linAlg.long3Dto2Ddiagonal(m['SigmaY'])

        L = linAlg.chol(K11 + SigmaY)
        Inner = np.linalg.solve(L, y0.T.reshape(-1, 1))
        alpha = np.linalg.solve(L.T, Inner)
        f = K21 @ alpha
        v = np.linalg.solve(L, K21.T)
        covf = K22 - v.T @ v
        E = (R_est[:, :, i] @ y1_b).T.reshape(-1, 1) - f

        bigR = np.kron(np.eye(Narray), R_est[:, :, i])
        covf_inv = linAlg.sinv(covf + bigR @ SigmaY @ bigR.T)
      
        dfx = np.squeeze(dKx21 @ alpha)
        dfR = np.squeeze(dKR21 @ alpha)
        dYR = linAlg.long3Dto2Dvertical(linAlg.crossMatrix((R_est[:, :, i] @ y1_b)))
        df = np.hstack((-dfx, -dfR+dYR))

        CostFunctionValue = np.append(CostFunctionValue, np.squeeze(E.T @ covf_inv @ E))# + np.log(np.linalg.det(covf + bigR @ SigmaY @ bigR.T)))   

        StepDirection = -linAlg.sinv(df.T @ covf_inv @ df) @ df.T @ covf_inv @ E
        beta = StepSize*StepDirection
        cov = CostFunctionValue[-1]/(Narray*3) * linAlg.sinv(df.T @ covf_inv @ df)
        if len(CostFunctionValue) > 1:
            if CostFunctionValue[-1] - CostFunctionValue[-2] > 0:
                R_est[:, :, i] = R_est[:, :, i-1]
                dx_est[:, i:i+1] = dx_est[:, i-1:i] 
                StepSize = StepSize/2
                CostFunctionValue = CostFunctionValue[:-1]
                dxCostFunction = dxCostFunction[:, :-1]
    
    return dx_est[:, -1].reshape(3, -1), R_est[:, :, -1], cov

def poseEstArraySim(init, Y, m):    
    theta = m["theta"]
    Narray = m['Narray']
    Rho = m['Rho']
    y0 = Y[:, :Narray] 
    y1_b = Y[:, Narray:]
    Xdata = Rho

    R_est = linAlg.expR(init[3:, 0:1]).T
    dx_est = init[:3, 0:1]

    Xpred = np.kron(np.ones((1, Narray)), dx_est) + R_est @ Rho

    K11 = GP.kernelCurlFreeLoop(Xdata, Xdata, m) + GP.kernelConstant(Xdata, Xdata, m)
    K21 = GP.kernelCurlFreeLoop(Xpred, Xdata, m) + GP.kernelConstant(Xpred, Xdata, m)
    K22 = GP.kernelCurlFreeLoop(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)
    dKx21, dKR21 = GP.nablaKernelCurlFreeLoop(Xpred, Xdata, R_est.reshape(3, 3, 1), m)

    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
    Inner = np.linalg.solve(L, y0.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    f = K21 @ alpha
    v = np.linalg.solve(L, K21.T)
    covf = K22 - v.T @ v
    E = (R_est @ y1_b).T.reshape(-1, 1) - f

    SigmaY = np.zeros((3*Narray, 3*Narray))
    for ind in range(Narray):
        SigmaY[ind*3:(ind+1)*3, ind*3:(ind+1)*3] = np.eye(3)*theta[2]

    bigR = np.kron(np.eye(Narray), R_est)
    covf_inv = linAlg.sinv(covf + bigR @ SigmaY @ bigR.T)
    
    dfx = np.squeeze(dKx21 @ alpha)
    dfR = np.squeeze(dKR21 @ alpha)
    dYR = linAlg.long3Dto2Dvertical(linAlg.crossMatrix((R_est @ y1_b)))
    df = np.hstack((-dfx, -dfR+dYR))

    lambdaCov = np.squeeze(E.T @ covf_inv @ E)/(3*Narray)# + np.log(np.linalg.det(covf + bigR @ SigmaY @ bigR.T)))   
    cov = lambdaCov * linAlg.sinv(df.T @ covf_inv @ df)    
    return cov

def poseEstArrayCov(init, Y, m):
    Rho = m["Rho"]
    Narray = m["Narray"]

    y0 = Y[:, :Narray] 
    y1_b = Y[:, Narray:]
    Xdata = Rho

    R_est = linAlg.expR(init[3:, 0:1]).T
    dx_est = init[:3, 0:1]

    CostFunctionValue = np.array([])
    dxCostFunction = np.zeros((6, 1))
    for i in range(1):
        dxCostFunction[:, i:i+1] = np.vstack((dx_est,  linAlg.R2eta(R_est)))

        Xpred = np.kron(np.ones((1, Narray)), dx_est) + R_est @ Rho

        K11 = GP.kernelCurlFreeLoop(Xdata, Xdata, m) + GP.kernelConstant(Xdata, Xdata, m)
        K21 = GP.kernelCurlFreeLoop(Xpred, Xdata, m) + GP.kernelConstant(Xpred, Xdata, m)
        K22 = GP.kernelCurlFreeLoop(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)
        dKx21, dKR21 = GP.nablaKernelCurlFreeLoop(Xpred, Xdata, R_est.reshape(3, 3, 1), m)
        
        SigmaY = linAlg.long3Dto2Ddiagonal(m['SigmaY'])

        L = linAlg.chol(K11 + SigmaY)
        Inner = np.linalg.solve(L, y0.T.reshape(-1, 1))
        alpha = np.linalg.solve(L.T, Inner)
        f = K21 @ alpha
        v = np.linalg.solve(L, K21.T)
        covf = K22 - v.T @ v
        E = (R_est @ y1_b).T.reshape(-1, 1) - f

        bigR = np.kron(np.eye(Narray), R_est)
        covf_inv = linAlg.sinv(covf + bigR @ SigmaY @ bigR.T)
      
        dfx = np.squeeze(dKx21 @ alpha)
        dfR = np.squeeze(dKR21 @ alpha)
        dYR = linAlg.long3Dto2Dvertical(linAlg.crossMatrix((R_est @ y1_b)))
        df = np.hstack((-dfx, -dfR+dYR))

        CostFunctionValue = np.append(CostFunctionValue, np.squeeze(E.T @ covf_inv @ E))# + np.log(np.linalg.det(covf + bigR @ SigmaY @ bigR.T)))   

        cov = CostFunctionValue[-1]/(Narray*3) * linAlg.sinv(df.T @ covf_inv @ df)
    return cov

def poseEstSingle(init, Y, m):
    theta = m["theta"]
    Narray = m['Narray']
    Rho = m['Rho']
    covMean = 0
    lambdaMean = 0
    covFused = 0
    R_est = linAlg.expR(init[3:, 0:1]).T
    dx_est = init[:3, 0:1]
    for i in range(Narray):
        Xdata = Rho[:, i:i+1]
        y0 = Y[:, i:i+1]
        y1_b = Y[:, Narray+i:Narray+i+1] 
        Xpred = dx_est + R_est @ Rho[:, i:i+1]

        K11 = GP.kernelCurlFreeLoop(Xdata, Xdata, m) + GP.kernelConstant(Xdata, Xdata, m)
        K21 = GP.kernelCurlFreeLoop(Xpred, Xdata, m) + GP.kernelConstant(Xpred, Xdata, m)
        K22 = GP.kernelCurlFreeLoop(Xpred, Xpred, m) + GP.kernelConstant(Xpred, Xpred, m)
        dKx21, dKR21 = GP.nablaKernelCurlFreeLoop(Xpred, Xdata, R_est.reshape(3, 3, 1), m)

        L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
        Inner = np.linalg.solve(L, y0.T.reshape(-1, 1))
        alpha = np.linalg.solve(L.T, Inner)
        f = K21 @ alpha
        v = np.linalg.solve(L, K21.T)
        covf = K22 - v.T @ v
        E = (R_est @ y1_b).T.reshape(-1, 1) - f

        SigmaY = np.eye(3)*theta[2]
        bigR = R_est
        covf_inv = linAlg.sinv(covf + bigR @ SigmaY @ bigR.T)
        
        dfx = np.squeeze(dKx21 @ alpha)
        lambdaMean += np.squeeze(E.T @ covf_inv @ E)/(3*Narray)
        covMean += linAlg.sinv(dfx.T @ covf_inv @ dfx)
        covFused += dfx.T @ covf_inv @ dfx

    A = np.zeros((6, 6))
    A[:3, :3] = lambdaMean*linAlg.sinv(covFused)
    B = np.zeros((6, 6))
    B[:3 ,:3] = lambdaMean*covMean/Narray
    return A, B

def blockRotation(Rdata, m):
    Rho = m["Rho"]
    k = 0
    A = Rdata.shape[2]
    B = Rho.shape[1]
    dRs = np.zeros((3, 3, A * B))
    for i in range(0, B):
        for j in range(0, A):
            dRs[:, :, k] = linAlg.crossVector(Rdata[:, :, j] @ Rho[:, i]).T
            k += 1
    return linAlg.long3Dto2Ddiagonal(dRs)

def addMagneticFieldDelft(f):
    inclination = 67.15*np.pi/180
    declination = 2.333*np.pi/180
    fconst = 49.388*np.array([[np.cos(inclination)*np.cos(declination)], [np.cos(inclination)*np.sin(declination)], [np.sin(inclination)]])
    if f.shape[1] > f.shape[0]:
        return f + fconst
    elif f.shape[1] == 1:
        return f + np.kron(np.ones((int(f.shape[0]/3), 1)), fconst)