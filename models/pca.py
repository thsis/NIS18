
import numpy as np
from matplotlib import pyplot as plt
from algorithms import eigen


class PCA(object):
    """
    Define class for Principal Component Analysis of a rank 2 array of data.

    PCA:
    Attributes:
        - data:
        - eigenvectors:
        - eigenvalues:
        - rotated data:
        - inertia:
    Methods:
        - fit(X):
        - plot(dim):
        - scree():
    """
    def __init__(self):
        self.data = None
        self.rotated_data = None
        self.cov = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.inertia = None

    def fit(self, X, norm=True):
        """
        Fit PCA to data.

        Parameters:
            - X:    numpy 2D-array. Contains data to be fit.
            - norm: boolean. If `True`, the default, use correlation matrix.
                    Else use the covariance matrix.

        Returns Attributes:
            - data:         Centered data.
            - rotated_data: Rotated data. Premultiplied with the matrix of
                            Eigenvectors
            - cov:          Covariance/correlation matrix of the data.
            - eigenvectors: Eigenvectors of `cov`.
            - eigenvalues:  Eigenvalues of `cov`.
            - inertia:      Proportion of explained variance by each component.
        """
        assert type(X) == np.ndarray
        # Prepare data.
        self.data = X - X.mean(axis=0)
        self.m = min(X.shape)

        # Decide if Covariance matrix shall be normalized.
        if norm:
            self.cov = np.corrcoef(self.data.T)
        else:
            self.cov = np.cov(self.data.T)

        # Solve Eigenvalue Problem.
        self.eigenvalues, self.eigenvectors = eigen.jacobi(self.cov)

        # Calculate inertia.
        self.inertia = self.eigenvalues / self.eigenvalues.sum()

        # Rotate the data.
        self.rotated_data = X.dot(self.eigenvectors)

    def scree(self):
        """
        Return scree plot for the supplied data.
        """
        fig, ax = plt.subplots()
        ax = plt.plot(self.inertia)
        ax = plt.scatter(x=range(self.m),
                         y=self.inertia)
        plt.axhline(y=0.05, c="gray", ls="--")
        plt.xticks(range(self.m),
                   range(1, self.m+1))
        plt.grid(True)
        plt.title("PCA: Scree plot")
        plt.xlabel("Eigenvalues")

        return fig, ax

    def plot(self, x, y):
        """
        Return 2-dimensional plot for 2 Pricipal Components.

        Parameters:
            - x: Integer indicating which Pricipal Component to plot on x-axis.
            - y: Integer indicating which Pricipal Component to plot on y-axis.

        """
        assert type(x) == type(y) == int
        assert x in range(self.m) and y in range(self.m)
        fig, ax = plt.subplots()
        ax = plt.scatter(x=self.rotated_data[:, x],
                         y=self.rotated_data[:, y])
        plt.grid(True)
        plt.xlabel(str(x+1) + ". Pricipal Component")
        plt.ylabel(str(y+1) + ". Pricipal Component")
        plt.title("PCA: Rotation")

        return fig, ax
