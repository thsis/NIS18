"""
Animate the QR-Method.

Visualize the performed chasing accross iterations

Code for heatmap was inspired by: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
Code for animations was inspired by:
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
"""

import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy import linalg as lin
from scipy.stats import ortho_group

outpath = os.path.join("media", "animations")

np.random.seed(42)
size = 5
Lambda = np.diag(np.random.randint(low=0, high=10, size=size))
G = ortho_group.rvs(dim=size)
X = np.dot(G, Lambda.dot(G.T))


def setup(func):
    # Construct empty matrix for background.
    empty = np.empty(shape=X.shape)
    empty[:] = np.nan

    # Set up figure and axis
    fig = plt.figure()
    ax = plt.axes(xlim=(empty.shape[0]-0.5, -0.5),
                  ylim=(-0.5, empty.shape[1]-0.5))
    plt.ylabel("")
    plt.xlabel("")
    hm = ax.imshow(empty,
                   cmap=plt.get_cmap('seismic'),
                   vmin=-X.max(), vmax=X.max())
    ax.figure.colorbar(hm)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(True)
    fig.suptitle("Demonstrate {} on a {}x{} matrix".format(func.__name__,
                                                           *X.shape))
    return fig, ax, hm, empty


def animation_factory(func):
    def animator(savepath):
        def algorithm_generator(*args, **kwargs):
            return func(*args, **kwargs)

        fig, ax, hm, empty = setup(func)
        algorithm_iterator = algorithm_generator()

        def init():
            """Initialize fig."""
            hm.set_data(empty)
            return hm

        # Define animation behavior
        def animate(i):
            """
            Update axes of fig.
                - i: Step of the current iteration. Will be called repeatedly
                     by the FuncAnimation-class.
            """
            A = next(algorithm_iterator)
            ax.set_title("Iteration: " + str(i))
            hm.set_data(np.round(A, 5))
            return hm

        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=60,
                                       interval=20)
        anim.save(savepath, fps=3, extra_args=['-vcodec', 'libx264'])
    return animator


@animation_factory
def qrm1():
    """
    Create generator for transformed matrices after applying the QR-Method.

    Parameters:
        - X: 2D-numpy array. Matrix whose eigenvalues should be computed.
        - maxiter: maximum number of iterations to be performed.
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


qrm1(os.path.join(outpath, "qrm1.mp4"))


@animation_factory
def qrm2():
    """
    Create generator for transformed matrices after applying the QR-Method.

    Parameters:
        - X: 2D-numpy array. Matrix whose eigenvalues should be computed.
        - maxiter: maximum number of iterations to be performed.
    Yields:
        - T: 2D-numpy array. Similar matrix to X.
    """
    # First stage: transform to upper Hessenberg-matrix.
    T = lin.hessenberg(X)

    k = 0
    # Second stage: perform QR-transformations.
    while k < 5000:
        k += 1
        Q, R = np.linalg.qr(T)
        T = R.dot(Q)
        yield T


qrm2(os.path.join(outpath, "qrm2.mp4"))


@animation_factory
def qrm3():
    """
    First compute similar matrix in Hessenberg form, then compute the
    Eigenvalues and Eigenvectors using the QR-Method.

    Parameters:
        - X: square numpy ndarray.
    Returns:
        - Eigenvalues of A.
        - Eigenvectors of A.
    """
    # First stage: transform to upper Hessenberg-matrix.
    T = lin.hessenberg(X)
    k = 0
    n, _ = X.shape

    # Second stage: perform QR-transformations.
    while k < 5000:
        k += 1
        Q, R = np.linalg.qr(T - T[n-1, n-1] * np.eye(n))
        T = R.dot(Q) + T[n-1, n-1] * np.eye(n)

        yield T


qrm3(os.path.join(outpath, "qrm3.mp4"))


@animation_factory
def jacobi():
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


jacobi(os.path.join(outpath, "jacobi.mp4"))
