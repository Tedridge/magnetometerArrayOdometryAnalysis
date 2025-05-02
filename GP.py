import numpy as np
import linAlg as linAlg
import magArray as magArray

''' Kernel functions '''
def nablaKernelCurlFreeLoop(X, X2, Rdata, m):
    theta = m["theta"]
    dR = np.zeros((X.shape[0] * X.shape[1], 3, X2.shape[0] * X2.shape[1]))
    dx = np.zeros((X.shape[0] * X.shape[1], 3, X2.shape[0] * X2.shape[1]))
    Rho_n = magArray.rho_n(Rdata, m)

    for i in range(X.shape[1]):
        for j in range(X2.shape[1]):
            ind1 = i * 3
            ind2 = ind1 + 3
            jnd1 = j * 3
            jnd2 = jnd1 + 3
            for k in range(3):
                one = np.zeros((3, 1))
                one[k] = 1

                diff = (X[:, i] - X2[:, j]).reshape(-1, 1)
                cross_s = linAlg.crossVector(Rho_n[:, i])

                d = cross_s.T @ diff

                pre_calc = (
                    theta[1] / theta[0] ** 2
                    * np.exp(np.sum(diff**2) / (-2 * theta[0]))
                )

                pre_calc_1 = pre_calc * (diff @ diff.T / theta[0] - np.eye(3))

                dx[ind1:ind2, k, jnd1:jnd2] = pre_calc_1 * diff[k : k + 1, 0]
                dR[ind1:ind2, k, jnd1:jnd2] = pre_calc_1 * d[k : k + 1, 0]
                
                dx[ind1:ind2, k, jnd1:jnd2] -= pre_calc * (one @ diff.T + diff @ one.T)
                dR[ind1:ind2, k, jnd1:jnd2] -= pre_calc * (cross_s[:, k : k + 1] @ diff.T + diff @ cross_s[:, k : k + 1].T)
    return dx, dR

def nablaKernelCurlFree(X, X2, Rdata, modelParameters):
    theta = modelParameters['theta']
    Nd = X.shape[0]
    expArg = np.zeros((X.shape[1], X2.shape[1]))
    diff = np.zeros((X.shape[1], X2.shape[1], Nd))
    diffLeft = np.zeros((Nd * X.shape[1], X2.shape[1]))
    diffRight = np.zeros((X.shape[1], Nd * X2.shape[1]))
    diffLong = np.zeros((Nd * X.shape[1], Nd * X2.shape[1], Nd))
    diffLongCross = np.zeros((Nd * X.shape[1], Nd * X2.shape[1], Nd))
    diffBig = np.zeros((Nd * X.shape[1], Nd * X2.shape[1], Nd))
    diffCross = np.zeros((X.shape[1], X2.shape[1], Nd))
    diffBigCross = np.zeros((Nd * X.shape[1], Nd * X2.shape[1], Nd))
    dKx = np.zeros((Nd * X.shape[1], Nd, Nd * X2.shape[1]))
    dKR = np.zeros((Nd * X.shape[1], Nd, Nd * X2.shape[1]))

    Rho_n = magArray.rho_n(Rdata, modelParameters)
    rhoCross = linAlg.matrix3DTo2DVertical(linAlg.crossMatrix(Rho_n))

    for i in range(Nd):
        expArg -= 1/2* np.subtract.outer(X[i, :], X2[i, :])**2 / theta[0]
        One = np.zeros((Nd, 1))
        One[i] = 1

        diff[:, :, i] = np.subtract.outer(X[i, :], X2[i, :])
        diffLeft += np.kron(diff[:, :, i], One)
        diffRight += np.kron(diff[:, :, i], One.T)

        diffBig[:, :, i] = np.kron(diff[:, :, i], np.ones((Nd, Nd)))
        diffCross[:, :, i] = np.kron(Rho_n[i:i+1, :].T, np.ones((1, X2.shape[1])))

    diffBigCross[:, :, 0] = np.kron(diffCross[:, :, 1]*diff[:, :, 2] - diffCross[:, :, 2]*diff[:, :, 1], np.ones((Nd, Nd)))
    diffBigCross[:, :, 1] = np.kron(diffCross[:, :, 2]*diff[:, :, 0] - diffCross[:, :, 0]*diff[:, :, 2], np.ones((Nd, Nd)))
    diffBigCross[:, :, 2] = np.kron(diffCross[:, :, 0]*diff[:, :, 1] - diffCross[:, :, 1]*diff[:, :, 0], np.ones((Nd, Nd)))

    expResult = np.kron(np.exp(expArg), np.ones((Nd, Nd)))
    eyeLong = np.kron(np.ones((X.shape[1], X2.shape[1])), np.eye(Nd))

    for i in range(Nd):
        One = np.zeros((Nd, 1))
        One[i] = 1

        diffLong[:, :, i] = np.kron(diffLeft, One.T) + np.kron(diffRight, One)
        diffLongCross[:, :, i] = np.kron(diffRight, np.ones((3, 1))) * np.kron(rhoCross[:, i:i+1], np.ones((1, Nd*X2.shape[1])))
        diffLongCross[:, :, i] += np.kron(diffLeft, np.ones((1, 3))) * np.kron(np.ones((1, X2.shape[1])), np.kron(-rhoCross[i::3, :], np.ones((Nd, 1))))
        diffProductLong = np.kron(diffLeft, np.ones((1, Nd))) * np.kron(diffRight, np.ones((Nd, 1))) 
 
        dKx[:, i, :] = theta[1]/theta[0]**2 * expResult * ((diffProductLong / theta[0] - eyeLong)*diffBig[:, :, i] - diffLong[:, :, i]) 
        dKR[:, i, :] = theta[1]/theta[0]**2 * expResult * ((diffProductLong / theta[0] - eyeLong)*diffBigCross[:, :, i] - diffLongCross[:, :, i]) 
    
    return dKx, dKR

def kernelCurlFreeLoop(X, X2, m):
    theta = m["theta"]
    k = np.zeros((X.shape[1], X2.shape[1]))
    for n in range(0, X.shape[0]):
        k -= 1 / 2 * np.subtract.outer(X[n, :], X2[n, :]) ** 2 / theta[0]

    A = np.zeros((X.shape[0] * X.shape[1], X2.shape[0] * X2.shape[1]))
    for i in range(0, X.shape[1]):
        for j in range(0, X2.shape[1]):
            diff_matrix = np.outer(X[:, i] - X2[:, j], X[:, i] - X2[:, j])
            A[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = (
                np.eye(3) - diff_matrix / theta[0]
            )

    k2 = np.kron(np.exp(k), np.ones((3, 3)))
    K = A * k2 * theta[1] / theta[0]
    return K

def kernelCurlFree(X, X2, m):
    theta = m["theta"]
    Nd = X.shape[0]
    exp_arg = np.zeros((X.shape[1], X2.shape[1]))
    diffleft = np.zeros((Nd * X.shape[1], X2.shape[1]))
    diffright = np.zeros((X.shape[1], Nd * X2.shape[1]))

    for i in range(Nd):
        exp_arg -= 1 / 2 * np.subtract.outer(X[i, :], X2[i, :]) ** 2 / theta[0]
        One = np.zeros((Nd, 1))
        One[i] = 1
        diff = np.subtract.outer(X[i, :], X2[i, :])

        diffleft += np.kron(diff, One)
        diffright += np.kron(diff, One.T)

    diffleft_long = np.kron(diffleft, np.ones((1, Nd)))
    diffright_long = np.kron(diffright, np.ones((Nd, 1)))

    Identity_matrices = np.kron(np.ones((X.shape[1], X2.shape[1])), np.eye(Nd))
    cross_term = Identity_matrices - diffleft_long * diffright_long / theta[0]

    exp_result = np.kron(np.exp(exp_arg), np.ones((Nd, Nd)))
    K = cross_term * exp_result * theta[1]
    return K

def kernelConstant(X, X2, m):
    theta = m["theta"]
    eye3 = np.eye(3)*theta[3]
    K = np.ones((X.shape[1], X2.shape[1]))
    Kcon = np.kron(K, eye3)
    return Kcon

def kernelConstantOpt(X, X2, theta):
    eye3 = np.eye(3)*theta[3]
    K = np.ones((X.shape[1], X2.shape[1]))
    Kcon = np.kron(K, eye3)
    return Kcon

def kernelSquaredExponential(X, X2, m):
    theta = m["theta"]
    # Squared exponential kernel
    k = np.zeros((X.shape[1], X2.shape[1]))
    for n in range(0, X.shape[0]):
        k -= 1 / 2 * np.subtract.outer(X[n, :], X2[n, :]) ** 2 / theta[0]
    K = theta[1] * np.exp(k)
    return K

def nablaKernelSquaredExponential(X, X2, m):
    theta = m["theta"]
    # First derivative of the Squared exponential kernel
    k = np.zeros((X.shape[1], X2.shape[1]))
    kdiff = np.zeros((X.shape[1], X2.shape[1], X.shape[0]))
    for n in range(0, X.shape[0]):
        kdiff[:, :, n] = np.subtract.outer(X[n, :], X2[n, :])
        k -= 1 / 2 * kdiff[:, :, n] ** 2 / theta[0]

    K = theta[1] * np.exp(k)
    dK = np.zeros((kdiff.shape[0] * X.shape[0], kdiff.shape[1]))
    for n in range(0, X.shape[0]):
        kdiff[:, :, n] = kdiff[:, :, n] * K / theta[0]
        one = np.zeros((X.shape[0], 1))
        one[n] = 1
        dK -= np.kron(kdiff[:, :, n], one)
    return dK

def nablaKernelSquaredExponentialRot(X, X2, Rdata, m):
    theta = m["theta"]
    Rho = m["Rho"]

    exp_arg = np.zeros((X.shape[1], X2.shape[1]))
    Xdiff = np.zeros((X.shape[1], X2.shape[1], X.shape[0]))
    Rho_cross_big = np.zeros((X.shape[1], X2.shape[1], X.shape[0]))
    diff_cross_big = np.zeros((X.shape[1], X2.shape[1], X.shape[0]))

    Rhon = np.zeros((3, 1))
    for i in range(m["Ncenter"]):
        Rhon = np.hstack((Rhon, Rdata[:, :, i] @ Rho))
    Rhon = Rhon[:, 1:]

    for n in range(X.shape[0]):
        Xdiff[:, :, n] = np.subtract.outer(X[n, :], X2[n, :])
        exp_arg -= 1 / 2 * Xdiff[:, :, n] ** 2 / theta[0]
        AAA = np.kron(Rhon[n : n + 1, :], np.ones((X.shape[1], 1)))
        Rho_cross_big[:, :, n] = np.kron(Rhon[n : n + 1, :], np.ones((X.shape[1], 1)))

    K = np.exp(exp_arg)
    dKx = np.zeros((Xdiff.shape[0] * X.shape[0], Xdiff.shape[1]))
    dKR = np.zeros((Xdiff.shape[0] * X.shape[0], Xdiff.shape[1]))

    diff_cross_big[:, :, 0] = (
        Rho_cross_big[:, :, 1] * Xdiff[:, :, 2]
        - Rho_cross_big[:, :, 2] * Xdiff[:, :, 1]
    )
    diff_cross_big[:, :, 1] = (
        Rho_cross_big[:, :, 2] * Xdiff[:, :, 0]
        - Rho_cross_big[:, :, 0] * Xdiff[:, :, 2]
    )
    diff_cross_big[:, :, 2] = (
        Rho_cross_big[:, :, 0] * Xdiff[:, :, 1]
        - Rho_cross_big[:, :, 1] * Xdiff[:, :, 0]
    )

    for n in range(X.shape[0]):
        one = np.zeros((X.shape[0], 1))
        one[n] = 1
        dKx -= theta[1] / theta[0] * np.kron(Xdiff[:, :, n] * K, one)
        dKR -= theta[1] / theta[0] * np.kron(diff_cross_big[:, :, n] * K, one)

    return dKx, dKR


''' Optimisation '''
def expLogMLCurlFree(s0, *args):
    (Xopt, Yopt) = args
    Y = Yopt.T.reshape(-1, 1)
    X = Xopt
    #Xopt = Rho

    exp_s = np.exp(s0)
    #exp_s = np.exp(s0)
    K11 = kernelCurlFreeOpt(X, X, exp_s) + kernelConstantOpt(X, X, exp_s)
    Ky = K11 + np.eye(len(K11)) * exp_s[2]
    
    L = linAlg.chol(Ky)
    Inner = linAlg.sinv(L, Y)
    alpha = linAlg.sinv(L.T, Inner)
    
    Ylog = 0.5 * Y.T @ alpha
    Ylog += 0.5 * np.log(np.linalg.det(Ky))
    Ylog += 0.5 * Y.shape[1] * np.log(2 * np.pi)
    return Ylog

def expLogMLCurlFree2(s0, Xopt, Yopt):
    Y = Yopt.T.reshape(-1, 1)
    X = Xopt
    #Xopt = Rho

    exp_s = np.exp(s0)
    #exp_s = np.exp(s0)
    K11 = kernelCurlFreeOpt(X, X, exp_s)
    Ky = K11 + np.eye(len(K11)) * exp_s[2]
    
    L = linAlg.chol(Ky)
    Inner = linAlg.sinv(L, Y)
    alpha = linAlg.sinv(L.T, Inner)
    
    Ylog = 0.5 * Y.T @ alpha
    Ylog += 0.5 * np.log(np.linalg.det(Ky))
    Ylog += 0.5 * Y.shape[1] * np.log(2 * np.pi)
    return Ylog

def expGradLogMLCurlFree(s0, *args):
    (Xopt, Yopt, theta) = args
     
    exp_s = np.exp(s0)
    Ydata = Yopt.T.reshape(-1, 1)

    Kcf = kernelCurlFreeOpt(Xopt, Xopt, exp_s)
    Kse = kernelSquaredExponentialOpt(Xopt, Xopt, exp_s)
    In = np.eye(len(Kcf))
    Ky = Kcf + In * exp_s[2]
    
    L = linAlg.chol(Ky)
    Inner = linAlg.sinv(L, Ydata)
    alpha = linAlg.sinv(L.T, Inner)
    beta = np.multiply.outer(alpha[:, 0], alpha[:, 0].T) - linAlg.sinv(Ky)
    
    diffleft = np.zeros((3 * Xopt.shape[1], Xopt.shape[1]))
    diffright = np.zeros((Xopt.shape[1], 3 * Xopt.shape[1]))

    k = np.zeros((Xopt.shape[1], Xopt.shape[1]))
    for n in range(0, Xopt.shape[0]):
        k += np.subtract.outer(Xopt[n, :], Xopt[n, :])**2
        One = np.zeros((3, 1))
        One[n] = 1
        diff = np.subtract.outer(Xopt[n, :], Xopt[n, :])

        diffleft += np.kron(diff, One)
        diffright += np.kron(diff, One.T)

    diffleft_long = np.kron(diffleft, np.ones((1, 3)))
    diffright_long = np.kron(diffright, np.ones((3, 1)))


    xdiffsquared = k
    dKdl = 0.5 / exp_s[0] * Kcf * np.kron(xdiffsquared, np.eye(3))
    dKdl += exp_s[0] / exp_s[1] * np.kron(Kse, np.eye(3)) * diffleft_long * diffright_long
    dKdf = Kcf
    dkdn = In * exp_s[2]

    dYl = -0.5 * np.trace(beta.dot(dKdl)) 
    dYf = -0.5 * np.trace(beta.dot(dKdf))
    # dYn = -0.5 * np.trace(beta.dot(dkdn))
    

    # return np.array([dYl, dYf, dYn])
    return np.array([dYl, dYf])

def kernelSquaredExponentialOpt(X, X2, theta):
    # Squared exponential kernel
    k = np.zeros((X.shape[1], X2.shape[1]))
    for n in range(0, X.shape[0]):
        k -= 1 / 2 * np.subtract.outer(X[n, :], X2[n, :]) ** 2 / theta[0]
    K = theta[1] * np.exp(k)
    return K

def expLogML(s0, *args):
    (Xopt, Yopt) = args
    exp_s = np.exp(s0)
    K11 = kernelCurlFreeOpt(Xopt, Xopt, exp_s)
    Ky = K11 + np.eye(len(K11)) * exp_s[2]
    
    L = linAlg.chol(Ky)
    Inner = linAlg.sinv(L, Yopt.T.reshape(-1, 1))
    alpha = linAlg.sinv(L.T, Inner)
    
    Y = 0.5 * Yopt.T.reshape(-1, 1).T @ alpha
    Y += 0.5 * np.log(np.linalg.det(Ky)) 
    #Y += 0.5 * ydata1.shape[0] * np.log(2 * np.pi)
    return Y

def expGradLogML(s0, *args):
    #Derivative of the Log marginal likelihood of a GP for optimisation, with the kernel trick
    (XdataOpt, YdataOpt) = args
    exp_s = np.exp(s0)
    
    K11 = kernelSquaredExponential(XdataOpt, XdataOpt, exp_s)
    In = np.eye(len(K11))
    Ky = K11 + In * exp_s[2]
    
    L = linAlg.chol(Ky)
    Inner = linAlg.sinv(L, YdataOpt)
    alpha = linAlg.sinv(L.T, Inner)
    beta = np.multiply.outer(alpha[:,0], alpha[:,0].T) - linAlg.sinv(Ky)
    
    dKdl = np.zeros(len(s0)-2)
    dY =  np.zeros(len(s0))

    k = np.zeros((XdataOpt.shape[0], XdataOpt.shape[0]))
    for n in range(0,XdataOpt.shape[1]):
        k += np.subtract.outer(XdataOpt[:,n], XdataOpt[:,n])**2
        
    xdiffsquared = k
    dKdl = 0.5 / exp_s[0] * K11 * xdiffsquared
    
    dKdf = K11
    dkdn = In * exp_s[2]

    dYl = -0.5 * np.trace(beta.dot(dKdl)) 
    dYf = -0.5 * np.trace(beta.dot(dKdf))
    dYn = -0.5 * np.trace(beta.dot(dkdn))
    
    dY[0] = dYl
    dY[1] = dYf
    dY[2] = dYn
    return dY



''' Data generation '''
def datagenSquaredExponential(Xdata, Xcross, m):
    # Create data from a GP prior
    theta = m["theta"]
    Xn = Xdata.shape[1]
    X = np.hstack((Xdata, Xcross))
    K = kernelSquaredExponential(X, X, m)
    Z = np.random.normal(0, 1, size=(len(K)))
    f = linAlg.chol(K) @ Z
    ftrue = f[:Xn]
    fcross = f[Xn:]
    ydata = ftrue + np.random.normal(0, np.sqrt(theta[2]), size=(ftrue.shape))
    return ydata.reshape(-1, 1), ftrue.reshape(-1, 1), fcross.reshape(-1, 1)

def datagenSquaredExponentialPrior(Xprior, m):
    # Create data from a GP posterior
    K = kernelSquaredExponential(Xprior, Xprior, m)
    Z = np.random.normal(0, 1, size=(len(K)))
    fprior = linAlg.chol(K) @ Z
    return fprior

def datagenSquaredExponentialPosterior(Xdata, Xprior, fprior, m):
    # Create data from a GP posterior
    K11 = kernelSquaredExponential(Xprior, Xprior, m)
    K21 = kernelSquaredExponential(Xdata, Xprior, m)
    L = linAlg.chol(K11)
    Inner = np.linalg.solve(L, fprior)
    alpha = np.linalg.solve(L.T, Inner)
    ftrue = K21.dot(alpha)
    return ftrue.reshape(-1, 1)

def datagenCurlFree(Xdata, Xcross, Rdata, e_eta, m):
    theta = m["theta"]
    X0, X1 = Xdata.shape
    Xn = X0 * X1
    X = np.hstack((Xdata, Xcross))
    K = kernelCurlFree(X, X, m)
    Z = np.random.normal(0, 1, size=(len(K)))
    f = linAlg.chol(K) @ Z
    ftrue = f[:Xn].reshape(-1, 3).T
    fcross = f[Xn:].reshape(-1, 3).T

    Narray = m['Narray']
    ydata = np.zeros(ftrue.shape)
    for i in range(Rdata.shape[2]):
        ydata[:, i*Narray:(i+1)*Narray] = linAlg.so3Rodrigues(e_eta[:, i]) @ Rdata[:, :, i] @ ftrue[:, i*Narray:(i+1)*Narray]

    ydata += np.random.normal(0, np.sqrt(theta[2]), size=(f[:Xn].shape)).reshape(-1, 3).T
    return ydata, ftrue, fcross

def datagenCurlFreeSim(Xdata, Xcross, m):
    theta = m["theta"]
    X0, X1 = Xdata.shape
    Xn = X0 * X1
    X = np.hstack((Xdata, Xcross))
    K = kernelCurlFree(X, X, m)
    Z = np.random.normal(0, 1, size=(len(K)))
    f = linAlg.chol(K) @ Z
    inclination = 67.15*np.pi/180
    declination = 2.333*np.pi/180
    fconst = 49.388*np.array([[np.cos(inclination)*np.cos(declination)], [np.cos(inclination)*np.sin(declination)], [np.sin(inclination)]])

    ftrue = f[:Xn].reshape(-1, 3).T + fconst
    fcross = f[Xn:].reshape(-1, 3).T + fconst

    ydata = ftrue.copy()
    ydata += np.random.normal(0, np.sqrt(theta[2]), size=(f[:Xn].shape)).reshape(-1, 3).T
    return ydata, ftrue, fcross

def datagenCurlFreePosteriorSim(Xdata, Xprior, fprior, Rdata, m):
    K11 = kernelCurlFree(Xprior, Xprior, m)
    K21 = kernelCurlFree(Xdata, Xprior, m)
    L = linAlg.chol(K11)
    Inner = np.linalg.solve(L, fprior.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    f = K21 @ alpha
    Narray = m['Narray']

    inclination = 67.15*np.pi/180
    declination = 2.333*np.pi/180
    fconst = 49.388*np.array([[np.cos(inclination)*np.cos(declination)], [np.cos(inclination)*np.sin(declination)], [np.sin(inclination)]])
    ftrue = f.reshape(-1, 3).T + fconst

    ydata = np.zeros(ftrue.shape)
    for i in range(Rdata.shape[2]):
        ydata[:, i*Narray:(i+1)*Narray] = Rdata[:, :, i] @ ftrue[:, i*Narray:(i+1)*Narray]
    
    return ydata

def datagenCurlFreePosterior(Xdata, Xcross, Rdata, e_eta, fprior, m):
    # Create data from a GP posterior
    K11 = kernelCurlFree(Xcross, Xcross, m)
    K21 = kernelCurlFree(Xdata, Xcross, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * m["theta"][2])
    Inner = np.linalg.solve(L, fprior.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    ftrue = K21.dot(alpha).reshape(-1, 3).T

    Narray = m['Narray']
    ydata = np.zeros(ftrue.shape)
    for i in range(Rdata.shape[2]):
        ydata[:, i*Narray:(i+1)*Narray] = linAlg.so3Rodrigues(e_eta[:, i]) @ Rdata[:, :, i] @ ftrue[:, i*Narray:(i+1)*Narray]

    return ydata, ftrue

''' Predictions '''
def predictSquaredExponential(Xdata, Xpred, ydata, m):
    theta = m["theta"]

    K11 = kernelSquaredExponential(Xdata, Xdata, m)
    K21 = kernelSquaredExponential(Xpred, Xdata, m)
    K22 = kernelSquaredExponential(Xpred, Xpred, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
    Inner = np.linalg.solve(L, ydata)
    alpha = np.linalg.solve(L.T, Inner)
    v = np.linalg.solve(L, K21.T)
    f = K21.dot(alpha)
    cov = K22 - v.T.dot(v)
    return f, cov

def predict3SquaredExponential(Xdata, Xpred, ydata, m):
    theta = m["theta"]
    N = ydata.shape[1]
    y = ydata.T.reshape(-1, 3)
    f = np.zeros((3, Xpred.shape[1]))

    K11 = kernelSquaredExponential(Xdata, Xdata, m)
    K21 = kernelSquaredExponential(Xpred, Xdata, m)
    K22 = kernelSquaredExponential(Xpred, Xpred, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
    v = np.linalg.solve(L, K21.T)
    cov = 0

    for i in range(3):
        Inner = np.linalg.solve(L, y[:, i:i+1])
        alpha = np.linalg.solve(L.T, Inner)
        f[i:i+1, :] = (K21 @ alpha).T
        One = np.zeros((3, 3))
        One[i, i] = 1
        cov +=  np.kron(K22 - v.T @ v, One)

    return f, cov

def predictCurlFree(Xdata, Xpred, ydata, m, mu=0):
    theta = m["theta"]
    K11 = kernelCurlFree(Xdata, Xdata, m)
    K21 = kernelCurlFree(Xpred, Xdata, m)
    K22 = kernelCurlFree(Xpred, Xpred, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
    Inner = np.linalg.solve(L, ydata.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    v = np.linalg.solve(L, K21.T)
    f = K21.dot(alpha)
    cov = K22 - v.T.dot(v)
    f = f.reshape(-1, 3).T
    return f, cov

def predictCurlFreeCon(Xdata, Xpred, ydata, m):
    theta = m["theta"]
    K11 = kernelCurlFree(Xdata, Xdata, m) + kernelConstant(Xdata, Xdata, m)
    K21 = kernelCurlFree(Xpred, Xdata, m) + kernelConstant(Xpred, Xdata, m)
    K22 = kernelCurlFree(Xpred, Xpred, m) + kernelConstant(Xpred, Xpred, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
    Inner = np.linalg.solve(L, ydata.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    v = np.linalg.solve(L, K21.T)
    f = K21.dot(alpha)
    cov = K22 - v.T.dot(v)
    f = f.reshape(-1, 3).T
    return f, cov

def predictCurlFreeMagArray(Xcenter, Xpred, ydata, Rdata, m):
    theta = m["theta"]
    Narray = m['Narray']
    Rho = m['Rho']
    Xdata = np.kron(Xcenter, np.ones((1, Narray))) + Rho
    Xpred = np.kron(Xpred, np.ones((1, Narray))) + Rdata @ Rho
    K11 = kernelCurlFree(Xdata, Xdata, m)
    K21 = kernelCurlFree(Xpred, Xdata, m)
    K22 = kernelCurlFree(Xpred, Xpred, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2])
    Inner = np.linalg.solve(L, ydata.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    v = np.linalg.solve(L, K21.T)
    f = K21.dot(alpha)
    cov = K22 - v.T.dot(v)
    f = f.reshape(-1, 3).T
    return f, cov

def predictNablaCurlFree(Xdata, Xpred, ydata, m, dfblock, mu=0):
    theta = m["theta"]

    K11 = kernelSquaredExponential(Xdata, Xdata, m)
    K21 = nablaKernelSquaredExponential(Xdata, Xpred, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2] + dfblock)
    Inner = np.linalg.solve(L, ydata)
    alpha = np.linalg.solve(L.T, Inner)
    df = K21.dot(alpha)
    return df

def predictNablaSquaredExponentialRot(Xdata, Xpred, ydata, Rdata, dfxblock, dfRblock, m):
    theta = m["theta"]
    K11 = kernelSquaredExponential(Xdata, Xdata, m)
    K21x, K21R = nablaKernelSquaredExponential(Xpred, Xdata, Rdata, m)
    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2] + dfxblock + dfRblock)
    Inner = np.linalg.solve(L, ydata.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    dfx = K21x @ alpha
    dfR = K21R @ alpha
    return dfx, dfR

def predictNablaSquaredExponentialRot3D(Xdata, Xpred, ydata, Rdata, dfxblock, dfRblock, m):
    theta = m["theta"]
    K11 = kernelSquaredExponential(Xdata, Xdata, m)
    dfx = 0
    dfR = 0
    K21x, K21R = nablaKernelSquaredExponential(Xpred, Xdata, Rdata, m)
    for i in range(3):
        L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2] + dfxblock + dfRblock)
        Inner = np.linalg.solve(L, ydata[i:i+1].T)
        alpha = np.linalg.solve(L.T, Inner)
        One = np.zeros((1, 3))
        One[0, i] = 1
        dfx += np.kron(One, K21x @ alpha)
        dfR += np.kron(One, K21R @ alpha)
    return dfx, dfR

def predictNablaSquaredExponentialMagArray(Xdata, Xpred, ydata, m, Sdata, Spred, mu=0):
    theta = m["theta"]

    Sdata = linAlg.long3dto2Ddiagonal(Sdata)
    Spred = linAlg.long3dto2Ddiagonal(Spred)

    K11 = kernelSquaredExponential(Xdata, Xdata, m)
    K21 = kernelSquaredExponential(Xpred, Xdata, m)
    K22 = kernelSquaredExponential(Xpred, Xpred, m)
    InputError = magArray.inputErrorMatrix(Xdata, Xdata, ydata, m, Sdata)

    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2] + InputError)
    Inner = np.linalg.solve(L, ydata)
    alpha = np.linalg.solve(L.T, Inner)
    v = np.linalg.solve(L, K21.T)
    f = K21.dot(alpha)

    # InputErrorPred = magArray.inputErrorMatrix(Xdata, Xpred, ydata, theta, Spred, 1, DoTheoreticalGradient)
    cov = K22 - v.T.dot(v)  # + InputErrorPred
    return f, cov

def predictNablaSquaredExponentialMagArrayRot(Xdata, Xpred, y, Rdata, Sx, SR, m):
    theta = m["theta"]
    #f = np.zeros((3, Xpred.shape[1]))
    N = Xdata.shape[0]*Xdata.shape[1]

    K11 = np.kron(kernelSquaredExponential(Xdata, Xdata, m), np.eye(3))
    K21 = np.kron(kernelSquaredExponential(Xpred, Xdata, m), np.eye(3))
    K22 = np.kron(kernelSquaredExponential(Xpred, Xpred, m), np.eye(3))
    
    Px, PR = magArray.inputErrorMatrixRot(Xdata, Xdata, y, Rdata, Sx, SR, m)

    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2] + Px + PR)
    Inner = np.linalg.solve(L, y.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    f = K21 @ alpha
    v = np.linalg.solve(L, K21.T)
    cov = K22 - v.T @ v
    return f.reshape(-1, 3).T, cov

def predictNablaCurlFreeRotation(
    Xdata, Rdata, ydata, Xpred, m, Px, PR, Pf
):
    theta = m["theta"]
    Nd = Xdata.shape[0]
    Ncenter = Xdata.shape[1]

    Rbig = linAlg.bigRotationMatrix(Rdata, m)

    K11 = kernelCurlFree(Xdata, Xdata, m)
    K21 = kernelCurlFree(Xpred, Xdata, m)
    dKx21, dKR21 = nablaKernelCurlFreeLoop(Xpred, Xdata, Rdata, m)

    L = linAlg.chol(K11 + np.eye(len(K11)) * theta[2] + Px + PR + Pf)
    Inner = np.linalg.solve(L, ydata.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)
    f = K21 @ alpha

    fcrossed = linAlg.long3dto2Ddiagonal(linAlg.crossMatrix(f.reshape(-1, Nd).T))
    dfx = np.squeeze(dKx21 @ alpha)
    dfR = np.squeeze(dKR21 @ alpha)
    return dfx, dfR, fcrossed

def predictCurlFreeArray(Xdata, Rdata, ydata, Xpred, Sigmax, SigmaR, m):
    theta = m["theta"]

    K11 = kernelCurlFree(Xdata, Xdata, m)
    K21 = kernelCurlFree(Xpred, Xdata, m)
    K22 = kernelCurlFree(Xpred, Xpred, m)
    (
        Px,
        PR,
        Pf
    ) = magArray.inputErrorMatrixCurlFreeRotation(Xdata, ydata, Xpred, Rdata, Sigmax, SigmaR, m)

    L = linAlg.chol(
        K11 + np.eye(len(K11)) * theta[2] + Px + PR + Pf
    )
    Inner = np.linalg.solve(L, ydata.T.reshape(-1, 1))
    alpha = np.linalg.solve(L.T, Inner)

    v = np.linalg.solve(L, K21.T)
    f = K21.dot(alpha)
    cov = K22 - v.T.dot(v)  # + InputErrorPred
    return (
        f.reshape(-1, 3).T,
        cov,
    )
