import os
import copy
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import linalg as lin
from scipy.stats import ortho_group
from matplotlib import pyplot as plt

datadir = os.path.join("analysis", "benchmarks.csv")
outpath = os.path.join("media", "plots")
trials = pd.read_csv(datadir, index_col=0)

trials.groupby(["algorithm", "dimension"]).iterations.describe()

# Boxplot iteration:
fig = plt.figure(figsize=(10, 5))
sns.boxplot(x="dimension", y="iterations", hue="algorithm", data=trials)
plt.yscale("log")
plt.title("Iterations needed before Convergence")
plt.savefig(os.path.join(outpath, "iterations_boxplot.png"))
plt.show()
plt.close()

# Boxplot elapsed time:
fig = plt.figure(figsize=(10, 5))
sns.boxplot(x="dimension", y="time", hue="algorithm", data=trials)
plt.title("Time needed before Convergence")
plt.ylabel("time (sec)")
plt.yscale('log')
plt.savefig(os.path.join(outpath, "time_boxplot.png"))
plt.show()
plt.close()

# Visualize Algorithm-Progress:
np.random.seed(42)
size = 5
Lambda = np.diag(np.random.randint(low=0, high=10, size=size))
G = ortho_group.rvs(dim=size)
X = np.dot(G, Lambda.dot(G.T))


def plot_factory(func):
    def plotter(savepath, **fig_kw):
        def algorithm_generator(*args, **kwargs):
            return func(*args, **kwargs)

        fig, ax = plt.subplots(nrows=2, ncols=2, **fig_kw)
        algorithm_iterator = algorithm_generator()
        j = -1

        for i, A in enumerate(algorithm_iterator):
            if i in (0, 1, 10, 75):
                j += 1

                hm = ax[j // 2, j % 2].imshow(A,
                                              cmap=plt.get_cmap('seismic'),
                                              vmin=-X.max(),
                                              vmax=X.max())
                ax[j // 2, j % 2].set_yticks([])
                ax[j // 2, j % 2].set_xticks([])
                ax[j // 2, j % 2].set_title("Iteration: " + str(i))

                if i > 75:
                    break

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(hm, cax=cbar_ax)

        fig.suptitle("Demonstrate {} on a {}x{} matrix".format(func.__name__,
                                                               *X.shape))
        fig.savefig(savepath)

        return fig, ax

    return plotter


@plot_factory
def jacobi():
    """
    Compute Eigenvalues and Eigenvectors for symmetric matrices using the
    jacobi method.

    Yields:
        * A - 2D numpy array of current iteration step.
    """
    A = copy.deepcopy(X)
    U = np.eye(A.shape[0])
    L = np.array([1])
    iterations = 0

    while iterations < 5000:
        L = np.abs(np.tril(A, k=0) - np.diag(A.diagonal()))
        i, j = np.unravel_index(L.argmax(), L.shape)
        alpha = 0.5 * np.arctan(2*A[i, j] / (A[i, i]-A[j, j]))

        V = np.eye(A.shape[0])
        V[i, i], V[j, j] = np.cos(alpha), np.cos(alpha)
        V[i, j], V[j, i] = -np.sin(alpha), np.sin(alpha)

        A = np.dot(V.T, A.dot(V))
        U = U.dot(V)
        iterations += 1
        yield A


@plot_factory
def qrm1():
    """
    Create generator for transformed matrices after applying the QR-Method.

    Yields:
        - T: 2D-numpy array. Similar matrix to X.
    """
    # First stage: transform to upper Hessenberg-matrix.
    T = copy.deepcopy(X)

    k = 0
    # Second stage: perform QR-transformations.
    while k < 5000:
        k += 1
        Q, R = np.linalg.qr(T)
        T = R.dot(Q)
        yield T


@plot_factory
def qrm2():
    """
    Create generator for transformed matrices after applying the QR-Method.

    Yields:
        - T: 2D-numpy array. Similar matrix to X.
    """
    # First stage: transform to upper Hessenberg-matrix.
    T = lin.hessenberg(X)

    k = 0
    # Second stage: perform QR-transformations.
    while k < 5000:
        if k == 0:
            yield X
        k += 1
        Q, R = np.linalg.qr(T)
        T = R.dot(Q)
        yield T


@plot_factory
def qrm3():
    """
    First compute similar matrix in Hessenberg form, then compute the
    Eigenvalues and Eigenvectors using the accelerated QR-Method.

    Yields:
        * T - 2D numpy array of current iteration step.
    """
    # First stage: transform to upper Hessenberg-matrix.
    T = lin.hessenberg(X)
    k = 0
    n, _ = X.shape

    # Second stage: perform QR-transformations.
    while k < 5000:
        if k == 0:
            yield X
        k += 1
        Q, R = np.linalg.qr(T - T[n-1, n-1] * np.eye(n))
        T = R.dot(Q) + T[n-1, n-1] * np.eye(n)

        yield T


jacobi(os.path.join(outpath, "jacobi.png"))
qrm1(os.path.join(outpath, "qrm1.png"))
qrm2(os.path.join(outpath, "qrm2.png"))
qrm3(os.path.join(outpath, "qrm3.png"))

plt.show()
plt.close()
