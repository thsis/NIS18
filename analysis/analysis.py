"""
Script that performs analysis of eigenvalue-algorithms.
    - Visual animation of performed chasing accross iterations
    - Comparison of algorithms:
        + Runtimes
        + Iterations needed until Convergence

Code for heatmap was found on: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""

import numpy as np
from algorithms import eigen, helpers
from matplotlib import pyplot as plt
from scipy import linalg as lin


def demonstrate_qrm(X, maxiter=5000):
    """
    INSERT DOCSTRING
    """
    n, m = X.shape
    assert n == m

    # First stage: transform to upper Hessenberg-matrix.
    T = lin.hessenberg(X)

    conv = False
    k = 0

    # Second stage: perform QR-transformations.
    while (not conv) and (k < maxiter):
        k += 1
        Q, R = helpers.qr_factorize(T - T[n-1, n-1] * np.eye(n))
        T = R.dot(Q) + T[n-1, n-1] * np.eye(n)

        conv = np.alltrue(np.isclose(np.tril(T, k=-1), np.zeros((n, n))))
        yield T





def animate(i, X):
    return next(iterator),



X = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 1]], dtype=np.float64)
np.linalg.eig(X)
eigen.qrm3(X)

fig, ax = plt.subplots()
im = ax.imshow(X)
plt.colorbar()
plt.show()


def heatmap(data,  ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


fig, ax = plt.subplots()
im, cbar = heatmap(X, ax=ax, cmap="YlGn", cbarlabel="")
plt.show()
im.set_data(X)

plt.show()
