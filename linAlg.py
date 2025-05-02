import numpy as np
from scipy import linalg as alg

def diag(x):
    # Create diagonal matrix
    return np.diag(x[:, 0])

def sinv(M, M2=0):
    # Stable inverse function
    if np.isscalar(M2) == True:
        M2 = np.eye(len(M))
    return np.linalg.solve(M, M2)

def jitter(M):
    # Add jitter for numerical stabilisation
    return np.eye(len(M)) * 10**-9

def chol(M):
    # Cholesky decomposition + jitter for numerical stabilisation
    return np.linalg.cholesky(M + jitter(M))

def normaliseVector(vector):
    minVal = np.min(vector)
    maxVal = np.max(vector)
    scaledVector = (vector - minVal) / (maxVal - minVal)
    return scaledVector

def vector2Scalar(y):
    y2 = np.zeros((y.shape[1], 1))
    for i in range(y.shape[1]):
        y2[i, 0] = np.linalg.norm(y[:, i])
    return y2


def matrix3DTo2DVertical(M3D):
    Nx, Ny, Nz = M3D.shape
    M2D = np.zeros((Nx * Nz, Ny))
    for kndx in range(0, M3D.shape[2]):
        M2D[kndx*Nx:(kndx+1)*Nx, :] = M3D[:, :, kndx]
    return M2D

''' Error metrics '''
def MSLL(f, P, ftrue, m):
    theta = m["theta"]
    MSLL = 0
    ftrue = ftrue.T.reshape(-1, 1)
    f = f.T.reshape(-1, 1)

    for i in range(len(f)):
        MSLL += ((ftrue[i] - f[i]) ** 2 / (P[i, i] + theta[2])) + np.log(2 * np.pi * (P[i, i] + theta[2]))

    return MSLL / (2 * len(f))

def NMSE(f, P, ftrue, m):
    ftrue = ftrue.T.reshape(-1, 1)
    f = f.T.reshape(-1, 1)
    P = np.diag(P)
    N = len(ftrue)
    return np.sum((ftrue - f) ** 2 / P) / N

def RMSE(fhat, P, ftrue, m):
    ftrue = ftrue.T.reshape(-1, 1)
    fhat = fhat.T.reshape(-1, 1)
    N = len(ftrue)
    return np.sqrt(np.sum((ftrue - fhat) ** 2) / N)

def MAE(f, P, ftrue, m):
    ftrue = ftrue.T.reshape(-1, 1)
    f = f.T.reshape(-1, 1)
    N = len(ftrue)
    return np.sum(np.abs(ftrue - f)) / N

def cov3DTrace(cov):
    n = int(cov.shape[0] / 3)
    trace_cov = np.zeros((1, n))
    for i in range(n):
        for j in range(3):
            trace_cov[0, i] += cov[3 * i + j, 3 * i + j]
    return trace_cov

def cov3DNorm(cov):
    n = int(cov.shape[0] / 3)
    norm_cov = np.zeros((1, n))
    a = np.zeros((3, 1))
    for i in range(n):
        # for j in range(3):
        #    a[j, 0] = cov[3 * i + j, 3 * i + j]
        a = cov[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)]
        norm_cov[0, i] += np.linalg.norm(a)
    return norm_cov

def cov3DDimension(cov, k):
    n = int(cov.shape[0] / 3)
    cov2 = np.zeros((1, n))
    for i in range(n):
        cov2[0, i] = cov[3 * i + k, 3 * i + k]
    return cov2

def cov3DMax(cov):
    n = int(cov.shape[0] / 3)
    max_cov = np.zeros((1, n))
    a = np.zeros((3, 1))
    for i in range(n):
        for j in range(3):
            a[j, 0] = cov[3 * i + j, 3 * i + j]
        max_cov[0, i] = np.max(a)
    return max_cov

def cov2Scalar(cov):
    N = int(cov.shape[0] / 3)
    a = np.zeros((1, N))
    for i in range(N):
        a[0, i] = np.trace(cov[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)]) / 3
    return a

def sumCov(M):
    A = int(M.shape[0] / 3)
    B = int(M.shape[1] / 3)

    SumM2 = np.zeros((A, B * 3))
    SumM = np.zeros((A, B))
    for i in range(int(A / 3)):
        ind1 = i * 3
        ind2 = ind1 + 3
        SumM2[ind1:ind2, :] += M[ind1:ind2, :]
        SumM2[ind1:ind2, :] += M[ind1 + 3 : ind2 + 3, :]
        SumM2[ind1:ind2, :] += M[ind1 + 6 : ind2 + 6, :]

    for j in range(int(B / 3)):
        jnd1 = j * 3
        jnd2 = jnd1 + 3
        SumM[:, jnd1:jnd2] += SumM2[:, jnd1:jnd2]
        SumM[:, jnd1:jnd2] += SumM2[:, jnd1 + 3 : jnd2 + 3]
        SumM[:, jnd1:jnd2] += SumM2[:, jnd1 + 6 : jnd2 + 6]
    return SumM


''' Matrix functions '''
def crossMatrix(a):
    A = np.zeros((3, 3, a.shape[1]))
    for i in range(0, a.shape[1]):
        A[:, :, i] = crossVector(a[:, i])
    return A

def crossVector(a):
    a = a.reshape(-1, 1)
    A = np.zeros((3, 3))
    A[1, 0] = a[2, 0]
    A[2, 0] = -a[1, 0]
    A[2, 1] = a[0, 0]
    A -= A.T
    return A

def long3Dto2Dhorizontal(A):
    B = np.zeros((3, 1))
    for i in range(0, A.shape[2]):
        B = np.hstack((B, A[:, :, i]))
    return B[:, 1:]

def long3Dto2Dvertical(A):
    B = np.zeros((1, 3))
    for i in range(0, A.shape[2]):
        B = np.vstack((B, A[:, :, i]))
    return B[1:, :]

def long3Dto2Ddiagonal(S3D):
    S2D = []
    for i in range(0, S3D.shape[2]):
        S2D = alg.block_diag(S2D, S3D[:, :, i])
    return S2D[1:, :]

def Long3dto2Ddiagonal(S3D):
    S2D = []
    for i in range(0, S3D.shape[2]):
        S2D = alg.block_diag(S2D, S3D[:, :, i])
    return S2D[1:, :]


def diagdf(df):
    N = int(df.shape[0] / 3)
    blockdf = np.zeros((df.shape[0], df.shape[0]))
    for i in range(N):
        ind1 = i * 3
        ind2 = ind1 + 3
        blockdf[ind1:ind2, ind1:ind2] = df[ind1:ind2, :]
    return blockdf

def df2blockdf3D(df, Xdata):
    blockdf = np.zeros((Xdata.shape[0]*Xdata.shape[1], Xdata.shape[0]*Xdata.shape[1]))
    for i in range(Xdata.shape[1]):
        blockdf[i*3:(i+1)*3, i*3:(i+1)*3] = df[i*3:(i+1)*3, :].T
    #df = diag(df)
    #blockdf = df.dot(np.kron(np.eye(Xdata.shape[1]), np.ones((Xdata.shape[0], 1))))
    return blockdf

def df2blockdf(df, Xdata):
    df.reshape(-1, 3)
    blockdf = np.zeros((Xdata.shape[1], Xdata.shape[0]*Xdata.shape[1]))
    for i in range(Xdata.shape[1]):
        blockdf[i, i*3:(i+1)*3] = df[i, :]
    #df = diag(df)
    #blockdf = df.dot(np.kron(np.eye(Xdata.shape[1]), np.ones((Xdata.shape[0], 1))))
    return blockdf

def df2blockdfCurlFree(df, Xdata):
    Din = m["Din"]
    df = diag(df)
    blockdf = df @ np.kron(
        np.eye(Xdata.shape[1]), np.kron(np.ones((Din, 1)), np.eye(Din))
    )
    return blockdf

def derivative3Dto2D(df, m):
    Din = m["Din"]
    N = int(df.shape[0] / 3)
    blockdf = np.zeros((df.shape[0] * Din, df.shape[0]))
    print(diag(df[:, :, 0]).shape)
    for i in range(Din):
        One = np.zeros((Din, 1))
        One[i] = 1
        blockdf += np.kron(np.eye(N), np.kron(One, np.eye(3))) @ diag(df[:, :, i])
    return blockdf

def bigRotationMatrix(Rdata, m):
    Narray = m["Narray"]
    Din = m["Din"]
    RdataArray = np.zeros((Din * Narray, Din * Narray, Rdata.shape[2]))
    for i in range(Rdata.shape[2]):
        RdataArray[:, :, i] = np.kron(np.eye(Narray), Rdata[:, :, i].T)
    Rbig = long3dto2Ddiagonal(RdataArray)
    return Rbig

def identityArray(sizeMatrix, D):
    matrix = np.stack([np.eye(sizeMatrix) for _ in range(D[0])], axis=2)
    for indx in range(1, len(D)):
        matrix = np.expand_dims(matrix, axis=2+indx)
        matrix = np.repeat(matrix, D[indx], axis=2+indx)
    return matrix



''' Rotation functions '''
def expR(eta):
    R = np.eye(3)
    if eta[0] == 0 and eta[1] == 0 and eta[2] == 0:
        return R
    else:
        eta_norm = np.sqrt(eta[0]**2 + eta[1]**2 + eta[2]**2)
        eta_cross = crossVector(eta/eta_norm)
        R += np.sin(eta_norm) * eta_cross
        R += (1-np.cos(eta_norm)) * eta_cross @ eta_cross 
        return R

def expq(eta):
    q = np.zeros((4, 1))
    q[0, 0] = 1
    if eta[0] == 0 and eta[1] == 0 and eta[2] == 0:
        return q
    else:
        eta_norm = np.sqrt(eta[0]**2 + eta[1]**2 + eta[2]**2)
        q[0, 0] = np.cos(eta_norm)
        q[1:4, 0:1] = eta / eta_norm * np.sin(eta_norm)
        return q

def qL(q):
    qL = np.zeros((4, 4))
    qL[0:4, 0:1] = q[0:4, 0:1]
    qL[0:1, 1:4] = -q[1:4, 0:1].T
    qL[1:4, 1:4] = np.eye(3) * q[0] + crossVector(q[1:])
    return qL

def q2R(q):
    c_v = crossVector(q[1:])
    R = np.outer(q[1:], q[1:]) + q[0]**2 * np.eye(3) + 2 * q[0] * c_v + np.dot(c_v, c_v)
    return R

def R2q(R):
    r = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    s = 0.5 / r
    q = np.array([[0.5 * r],
                  [(R[2, 1] - R[1, 2]) * s],
                  [(R[0, 2] - R[2, 0]) * s],
                  [(R[1, 0] - R[0, 1]) * s]])
    return q

# def R2eta(R):

#     eta = np.zeros((3, 1))
#     # eta[0, 0] = R[2, 1]
#     # eta[1, 0] = R[0, 2]
#     # eta[2, 0] = R[1, 0]
#     if np.array_equal(R, np.eye(3)):
#         return eta
#     else:
#         theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
#         # theta = np.arccos((np.trace(R) - 1) / 2)
#         if theta != 0:
#             R2 = theta/(2*np.sin(theta)) * (R - R.T)
#             eta[0, 0] = R2[2, 1]
#             eta[1, 0] = R2[0, 2]
#             eta[2, 0] = R2[1, 0]

#         return eta
def R2eta(R):

    eta = np.zeros((3, 1))
    # eta[0, 0] = R[2, 1]
    # eta[1, 0] = R[0, 2]
    # eta[2, 0] = R[1, 0]
    if np.array_equal(R, np.eye(3)):
        return eta
    else:
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        # theta = np.arccos((np.trace(R) - 1) / 2)
        if theta != 0:
            R2 = theta/(2*np.sin(theta)) * (R - R.T)
            eta[0, 0] = R2[2, 1]
            eta[1, 0] = R2[0, 2]
            eta[2, 0] = R2[1, 0]

        return eta

def so3Rodrigues(psi):
    if psi[0] == 0 and psi[1] == 0 and psi[2] == 0:
        n = psi
        psi_abs = 0
    else:
        psi_abs = np.linalg.norm(psi)
        n = psi / psi_abs
    R = np.zeros((3, 3))
    sin = np.sin(psi_abs)
    cos = np.cos(psi_abs)
    for i in range(0, 3):
        R[i, i] = cos + n[i] ** 2 * (1 - cos)

    R[0, 1] = n[0] * n[1] * (1 - cos) - n[2] * sin
    R[1, 0] = n[0] * n[1] * (1 - cos) + n[2] * sin

    R[0, 2] = n[0] * n[2] * (1 - cos) + n[1] * sin
    R[2, 0] = n[0] * n[2] * (1 - cos) - n[1] * sin

    R[1, 2] = n[1] * n[2] * (1 - cos) - n[0] * sin
    R[2, 1] = n[1] * n[2] * (1 - cos) + n[0] * sin
    return R

def yawRotations(N, m):
    yawrate = m["yawrate"]
    R = np.zeros((3, 3, N))
    yaw = 0
    for i in range(0, N):
        yaw += yawrate
        R[:, :, i] = Rz(yaw)
    return R

def Rx(alpha):
    R = np.eye(3)
    R[1, 1] = np.cos(alpha)
    R[2, 2] = np.cos(alpha)
    R[1, 2] = np.sin(alpha)
    R[2, 1] = -np.sin(alpha)
    return R

def Ry(beta):
    R = np.eye(3)
    R[0, 0] = np.cos(beta)
    R[2, 2] = np.cos(beta)
    R[0, 2] = np.sin(beta)
    R[2, 0] = -np.sin(beta)
    return R

def Rz(gamma):
    R = np.eye(3)
    R[0, 0] = np.cos(gamma)
    R[1, 1] = np.cos(gamma)
    R[0, 1] = -np.sin(gamma)
    R[1, 0] = np.sin(gamma)
    return R

def frameBody2Navigation(ya, Rba, m):
    Narray = m['Narray']
    yb = np.zeros(ya.shape)
    for i in range(Rba.shape[2]):
        yb[:, i*Narray:(i+1)*Narray] = Rba[:, :, i] @ ya[:, i*Narray:(i+1)*Narray] 
    return yb

def frameNavigation2Body(ya, Rba, m):
    Narray = m['Narray']
    yb = np.zeros(ya.shape)
    for i in range(Rba.shape[2]):
        yb[:, i*Narray:(i+1)*Narray] = Rba[:, :, i].T @ ya[:, i*Narray:(i+1)*Narray] 
    return yb
