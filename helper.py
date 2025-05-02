import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import linAlg as linAlg

''' Data generation functions '''
def gridpoints(N, D, Din, Start, End):
    # Create input points on a grid
    X1 = np.linspace(Start, End, N)

    if Din == 1:
        X = X1.reshape(D, -1)
    else:
        dict = {}
        inputs = []

        for i in range(0, D):
            inputs.append("x" + str(i))
            dict[inputs[i]] = X1

        X2 = np.array(np.meshgrid(*(v for _, v in sorted(dict.items()))))
        X = np.zeros((D, N**D))
        for i in range(0, D):
            X[i, :] = X2[i, :].reshape(1, N**D)[0, :]

    if Din > D:
        X3 = np.zeros((Din - D, X.shape[1]))
        X = np.vstack((X, X3))
    return X

def gridpoints2(N, D, Din, Start1, End1, Start2, End2):
    # Create input points on a grid
    X1 = np.linspace(Start1, End1, N)
    X2 = np.linspace(Start2, End2, N)

    Xa, Xb = np.meshgrid(X1, X2)

    Xa = Xa.reshape(1, N**D)
    Xb = Xb.reshape(1, N**D)
    Xc = np.zeros((Din - D, Xa.shape[1]))
    X = np.vstack((Xa, Xb, Xc))
    return X

def gridpoints3(N, D, Din, Start1, End1, Start2, End2, Start3, End3):
    # Create input points on a grid
    X1 = np.linspace(Start1, End1, N)
    X2 = np.linspace(Start2, End2, N)
    X3 = np.linspace(Start3, End3, N)

    Xa, Xb, Xc = np.meshgrid(X1, X2, X3)

    Xa = Xa.reshape(1, N**D)
    Xb = Xb.reshape(1, N**D)
    Xc = Xc.reshape(1, N**D)
    X = np.vstack((Xa, Xb, Xc))
    return X


def derivativesigmoid(X):
    # Create data that follows a sigmoid function derivative
    sigma = 1 / (1 + np.exp(-X))
    df = sigma * (1 - sigma)
    return df

''' Plotting Functions '''

def plotCovariancesScalar(X, Xdata, Covariance):
    Nsmooth = 100
    Xsmooth = gridpoints(Nsmooth, 2, 2, Start, End)
    cov_smooth = scipy.interpolate.griddata(
        X[0:2, :].T, np.diag(Covariance).T, Xsmooth.T, method="cubic"
    )

    plt.figure()
    pcm = plt.pcolormesh(
        Xsmooth[0, :].reshape(Nsmooth, Nsmooth),
        Xsmooth[1, :].reshape(Nsmooth, Nsmooth),
        cov_smooth.reshape(Nsmooth, Nsmooth),
    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    cbar = plt.colorbar(pcm)
    cbar.set_label("l")
    plt.scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
    plt.show()
    return

def plotCovariancesScalar2(X, Xdata, Covariance, Covariance2):
    # Nsmooth = 100
    # Xsmooth = gridpoints(Nsmooth, 2, 2, Start, End)
    Npred = int(np.sqrt(X.shape[1]))
    trace_cov = np.diag(Covariance)
    # cov_smooth = scipy.interpolate.griddata(X[0:2, :].T, trace_cov.T, Xsmooth.T, method='cubic')

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # First subplot
    pcm1 = axes[0].pcolormesh(
        X[0, :].reshape(Npred, Npred),
        X[1, :].reshape(Npred, Npred),
        trace_cov.reshape(Npred, Npred),
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Correlated noise - SE kernel")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
    # axes[0].scatter(np.zeros(1), np.zeros(1), marker='x', c='white')
    cbar1 = fig.colorbar(pcm1, ax=axes[0], orientation="vertical", pad=0.1)

    # Second subplot
    trace_cov2 = np.diag(Covariance2)
    # cov_smooth2 = scipy.interpolate.griddata(X[0:2, :].T, trace_cov2.T, X.T, method='cubic')

    pcm2 = axes[1].pcolormesh(
        X[0, :].reshape(Npred, Npred),
        X[1, :].reshape(Npred, Npred),
        trace_cov2.reshape(Npred, Npred),
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Uncorrelated noise - SE kernel")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
    cbar2 = fig.colorbar(pcm2, ax=axes[1], orientation="vertical", pad=0.1)

    plt.show()
    return

def plotCovariances(X, Xdata, Covariance):
    Nsmooth = 100
    Xsmooth = gridpoints(Nsmooth, 2, 2, Start, End)
    trace_cov = linAlg.cov3DNorm(Covariance)
    cov_smooth = scipy.interpolate.griddata(
        X[0:2, :].T, trace_cov.T, Xsmooth.T, method="cubic"
    )

    plt.figure()
    pcm = plt.pcolormesh(
        Xsmooth[0, :].reshape(Nsmooth, Nsmooth),
        Xsmooth[1, :].reshape(Nsmooth, Nsmooth),
        cov_smooth.reshape(Nsmooth, Nsmooth),
    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    cbar = plt.colorbar(pcm)
    cbar.set_label("l")
    plt.scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
    plt.show()
    return

def plotCovariances2(X, Xdata, Covariance, Covariance2):
    Nsmooth = 100
    Xsmooth = gridpoints(Nsmooth, 2, 2, Start, End)
    trace_cov = linAlg.cov3DNorm(Covariance)
    cov_smooth = scipy.interpolate.griddata(
        X[0:2, :].T, trace_cov.T, Xsmooth.T, method="cubic"
    )

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # First subplot
    pcm1 = axes[0].pcolormesh(
        Xsmooth[0, :].reshape(Nsmooth, Nsmooth),
        Xsmooth[1, :].reshape(Nsmooth, Nsmooth),
        cov_smooth.reshape(Nsmooth, Nsmooth),
    )
    axes[0].set_title("Correlated noise - Curlfree Kernel")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")

    # Second subplot
    trace_cov2 = linAlg.cov3DNorm(Covariance2)
    cov_smooth2 = scipy.interpolate.griddata(
        X[0:2, :].T, trace_cov2.T, Xsmooth.T, method="cubic"
    )

    pcm2 = axes[1].pcolormesh(
        Xsmooth[0, :].reshape(Nsmooth, Nsmooth),
        Xsmooth[1, :].reshape(Nsmooth, Nsmooth),
        cov_smooth2.reshape(Nsmooth, Nsmooth),
    )
    axes[1].set_title("Uncorrelated noise - Curlfree Kernel")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")

    # Colorbar
    cbar = fig.colorbar(pcm1, ax=axes, orientation="vertical", pad=0.1)
    cbar.set_label("l")

    plt.show()
    return

def plotCovariances3(X, Xdata, Covariance, Covariance2, Covariance3):
    Nsmooth = 100
    Xsmooth = gridpoints(Nsmooth, 2, 2, Start, End)
    Npred = int(np.sqrt(X.shape[1]))
    for i in range(3):
        trace_cov = linAlg.cov3DDimension(Covariance, i)
        # cov_smooth = scipy.interpolate.griddata(X[0:2, :].T, trace_cov.T, Xsmooth.T, method='cubic')

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 5))

        # First subplot
        pcm1 = axes[0].pcolormesh(
            X[0, :].reshape(Npred, Npred),
            X[1, :].reshape(Npred, Npred),
            trace_cov.reshape(Npred, Npred),
            vmin=0,
            vmax=1,
        )
        axes[0].set_title("Correlated noise - Curlfree Kernel")
        axes[0].set_xlabel(r"$x_1$")
        axes[0].set_ylabel(r"$x_2$")
        axes[0].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
        # axes[0].scatter(np.zeros(1), np.zeros(1), marker='x', c='white')
        cbar = fig.colorbar(pcm1, ax=axes[0], orientation="vertical", pad=0.1)

        # Second subplot
        trace_cov2 = linAlg.cov3DDimension(Covariance2, i)
        # cov_smooth2 = scipy.interpolate.griddata(X[0:2, :].T, trace_cov2.T, Xsmooth.T, method='cubic')

        pcm2 = axes[1].pcolormesh(
            X[0, :].reshape(Npred, Npred),
            X[1, :].reshape(Npred, Npred),
            trace_cov2.reshape(Npred, Npred),
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Uncorrelated noise - Curlfree Kernel")
        axes[1].set_xlabel(r"$x_1$")
        axes[1].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
        # axes[1].scatter(np.zeros(1), np.zeros(1), marker='x', c='white')
        # Colorbar
        cbar = fig.colorbar(pcm2, ax=axes[1], orientation="vertical", pad=0.1)
        # cbar.set_label("l")

        # Second subplot
        trace_cov3 = linAlg.cov3DDimension(Covariance3, i)
        # cov_smooth3 = scipy.interpolate.griddata(X[0:2, :].T, trace_cov3.T, Xsmooth.T, method='cubic')

        pcm3 = axes[2].pcolormesh(
            X[0, :].reshape(Npred, Npred),
            X[1, :].reshape(Npred, Npred),
            trace_cov3.reshape(Npred, Npred),
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("No noise - Curlfree Kernel")
        axes[2].set_xlabel(r"$x_1$")
        axes[2].scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
        # axes[1].scatter(np.zeros(1), np.zeros(1), marker='x', c='white')
        # Colorbar
        cbar = fig.colorbar(pcm3, ax=axes[2], orientation="vertical", pad=0.1)
        # cbar.set_label("l")

        plt.show()
    return


def make2DPlot(X, f, cov, Xdata, ydata, Start, End, opacity=0):
    # Nsmooth = 100
    # Xsmooth = gridpoints(Nsmooth, 2, 2, Start, End)
    Npred = int(np.sqrt(X.shape[1]))
    if f.shape[0] > 1 and f.shape[1] > 1:
        f = np.linalg.norm(f, axis=0)
        cov2 = linAlg.cov3DTrace(cov)
    else:
        cov2 = np.diag(cov)
        f = f.reshape(-1)
    # f_smooth = scipy.interpolate.griddata(X[0:2, :].T, f.T, Xsmooth.T, method = 'cubic')

    if opacity == 1:
        # cov_smooth = scipy.interpolate.griddata(X[0:2, :].T, cov2.T, Xsmooth.T, method = 'cubic')
        alpha_smooth = 1 - linAlg.normaliseVector(cov2)
    else:
        alpha_smooth = np.ones((Npred, Npred))

    plt.figure()
    pcm = plt.pcolormesh(
        X[0, :].reshape(Npred, Npred),
        X[1, :].reshape(Npred, Npred),
        f.reshape(Npred, Npred),
        alpha=alpha_smooth.reshape(Npred, Npred),
    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Ambient magnetic field")
    cbar = plt.colorbar(pcm)
    cbar.set_label("Magnetic field strength")
    plt.scatter(Xdata[0, :], Xdata[1, :], marker="x", c="red")
    plt.show()
    return

def makeSmooth(Xpred, Npred, X, f, cov, opacity):
    if f.shape[0] > 1 and f.shape[1] > 1:
        f = np.linalg.norm(f, axis=0)
        cov2 = linAlg.cov3DTrace(cov)
    else:
        cov2 = np.diag(cov)
        f = f.reshape(-1)
    # f_smooth = scipy.interpolate.griddata(X[0:2, :].T, f.T, Xpred.T, method = 'cubic')

    if opacity == 1:
        # cov_smooth = scipy.interpolate.griddata(X[0:2, :].T, cov2.T, Xpred.T, method = 'cubic')
        alpha = 1 - linAlg.normaliseVector(cov2)
    else:
        alpha = np.ones((Npred, Npred))

    return f, alpha

def makePlots(
    Xcross,
    fcross,
    Xpred,
    f1,
    cov1,
    f2,
    cov2,
    f3,
    cov3,
    Xdata,
    ydata,
    Start,
    End,
    opacity=0,
):
    # Nsmooth = 100
    Npred = int(np.sqrt(Xpred.shape[1]))
    f_smooth1, alpha_smooth1 = makeSmooth(Xpred, Npred, Xpred, f1, cov1, opacity)
    f_smooth2, alpha_smooth2 = makeSmooth(Xpred, Npred, Xpred, f2, cov2, opacity)
    f_smooth3, alpha_smooth3 = makeSmooth(Xpred, Npred, Xpred, f3, cov3, opacity)
    fcross, alpha_cross = makeSmooth(Xpred, Npred, Xcross, fcross, 0 * cov1, 0)

    min = np.min(np.vstack((fcross, f_smooth1, f_smooth2, f_smooth3)))
    max = np.max(np.vstack((fcross, f_smooth1, f_smooth2, f_smooth3)))
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # Adjust the figsize as needed
    pcm0 = axes[0, 0].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        fcross.reshape(Npred, Npred),
        alpha=alpha_cross.reshape(Npred, Npred),
        vmin=min,
        vmax=max,
    )
    axes[0, 0].set_xlabel(r"$x_1$")
    axes[0, 0].set_ylabel(r"$x_2$")
    axes[0, 0].set_title("True field")
    fig.colorbar(pcm0, ax=axes[0, 0])
    # cbar.set_label("Magnetic field strength")

    ##############################
    pcm1 = axes[1, 1].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth1.reshape(Npred, Npred),
        alpha=alpha_smooth1.reshape(Npred, Npred),
        vmin=min,
        vmax=max,
    )
    axes[1, 1].set_xlabel(r"$x_1$")
    axes[1, 1].set_ylabel(r"$x_2$")
    axes[1, 1].set_title("No input error")
    fig.colorbar(pcm1, ax=axes[1, 1])
    # axes[1, 1].scatter(Xdata[0, :], Xdata[1, :], marker='x', c='red')

    ##############################
    pcm2 = axes[1, 0].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth2.reshape(Npred, Npred),
        alpha=alpha_smooth2.reshape(Npred, Npred),
        vmin=min,
        vmax=max,
    )
    axes[1, 0].set_xlabel(r"$x_1$")
    axes[1, 0].set_ylabel(r"$x_2$")
    axes[1, 0].set_title("Uncorrelated error")
    fig.colorbar(pcm2, ax=axes[1, 0])
    # axes[1, 0].scatter(Xdata[0, :], Xdata[1, :], marker='x', c='red')

    ##############################
    pcm3 = axes[0, 1].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth3.reshape(Npred, Npred),
        alpha=alpha_smooth3.reshape(Npred, Npred),
        vmin=min,
        vmax=max,
    )
    axes[0, 1].set_xlabel(r"$x_1$")
    axes[0, 1].set_ylabel(r"$x_2$")
    axes[0, 1].set_title("Correlated error")
    fig.colorbar(pcm3, ax=axes[0, 1])
    # axes[0, 1].scatter(Xdata[0, :], Xdata[1, :], marker='x', c='red')

    plt.tight_layout()
    plt.show()
    return

def makePlotError(
    Xcross,
    fcross,
    Xpred,
    f1,
    cov1,
    f2,
    cov2,
    f3,
    cov3,
    Xdata,
    ydata,
    Start,
    End,
    opacity=0,
):
    # Nsmooth = 100
    Npred = int(np.sqrt(Xpred.shape[1]))
    f_smooth1, alpha_smooth1 = makeSmooth(Xpred, Npred, Xpred, f1-fcross, cov1, opacity)
    f_smooth2, alpha_smooth2 = makeSmooth(Xpred, Npred, Xpred, f2-fcross, cov2, opacity)
    f_smooth3, alpha_smooth3 = makeSmooth(Xpred, Npred, Xpred, f3-fcross, cov3, opacity)

    min_val = np.min(np.vstack((f_smooth1, f_smooth2, f_smooth3)))
    max_val = np.max(np.vstack((f_smooth1, f_smooth2, f_smooth3)))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.2]})

    ##############################
    pcm1 = axes[0].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth1.reshape(Npred, Npred),
        alpha=alpha_smooth1.reshape(Npred, Npred),
        vmin=min_val,
        vmax=max_val,
    )
    axes[0].set_ylabel(r"$x_2 [m]$")
    axes[0].set_title("Assumes no input error")

    ##############################
    pcm2 = axes[1].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth2.reshape(Npred, Npred),
        alpha=alpha_smooth2.reshape(Npred, Npred),
        vmin=min_val,
        vmax=max_val,
    )
    axes[1].set_xlabel(r"$x_1  [m]$")
    axes[1].set_title("Assumes uncorrelated error")

    ##############################
    pcm3 = axes[2].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth3.reshape(Npred, Npred),
        alpha=alpha_smooth3.reshape(Npred, Npred),
        vmin=min_val,
        vmax=max_val,
    )
    axes[2].set_title("Assumes correlated error")

    # Add a single colorbar
    cbar = plt.colorbar(pcm3, ax=axes[2])

    plt.tight_layout()
    plt.show()

    my_path =  os.getcwd() # Figures out the absolute path for you in case your working directory moves around.
    my_figures = 'Figures'
    my_file = 'Error1step3compared.pdf'
    fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf')  
    return

def makePlotError2(
    Xcross,
    fcross,
    Xpred,
    f1,
    cov1,
    f2,
    cov2,
    f3,
    cov3,
    Xdata,
    ydata,
    Start,
    End,
    opacity=0,
):
    # Nsmooth = 100
    Npred = int(np.sqrt(Xpred.shape[1]))
    f_smooth1, alpha_smooth1 = makeSmooth(Xpred, Npred, Xpred, f1-fcross, cov1, opacity)
    f_smooth2, alpha_smooth2 = makeSmooth(Xpred, Npred, Xpred, f2-fcross, cov2, opacity)
    f_smooth3, alpha_smooth3 = makeSmooth(Xpred, Npred, Xpred, f3-fcross, cov3, opacity)

    min_val = np.min(np.vstack((f_smooth1, f_smooth2, f_smooth3)))
    max_val = np.max(np.vstack((f_smooth1, f_smooth2, f_smooth3)))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 2), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1.2]})

    ##############################
    pcm1 = axes[0].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth1.reshape(Npred, Npred),
        alpha=alpha_smooth1.reshape(Npred, Npred),
        vmin=min_val,
        vmax=max_val,
    )
    #axes[0].set_ylabel(r"$x_2 [m]$")
    axes[0].set_title("Measurement equation (1)")

    ##############################
    pcm3 = axes[1].pcolormesh(
        Xpred[0, :].reshape(Npred, Npred),
        Xpred[1, :].reshape(Npred, Npred),
        f_smooth3.reshape(Npred, Npred),
        alpha=alpha_smooth3.reshape(Npred, Npred),
        vmin=min_val,
        vmax=max_val,
    )
    axes[1].set_title("Measurement equation (3)")

    # Add a single colorbar
    cbar = plt.colorbar(pcm3, ax=axes[1])

    plt.tight_layout()
    plt.show()


    my_path =  os.getcwd() # Figures out the absolute path for you in case your working directory moves around.
    my_figures = 'Figures'
    my_file = 'Error1step.pdf'
    fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf')  
    return
