import numpy as np
from algorithms import eigen
from matplotlib import pyplot as plt


class LDA(object):
    """
    Define class for Linear Discriminant Analysis of a rank 2 array of data.

    LDA:
    Attributes:
        - data:
        - generalized eigenvectors:
        - generalized eigenvalues:
        - rotated data:
        - inertia:
    Methods:
        - fit(X):
        - plot(dim):
        - scree():
    """
    def __init__(self):
        self.data = None
        self.groups = None
        self.Sb = None
        self.Sw = None
        self.rotated_data = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.inertia = None

    def fit(self, X, g):
        """
        Fit Linear Discriminant model to data.

        Parameters:
            - X: numpy ndarray of shape (n, m). Contains data and labels of
                 observations
            - g: integer indicating the column of labels.

        Defines Attributes:
            - data: copy of X without label column.
            - groups: 1D numpy array of labels.
            - Sb: 2D numpy array of Between-Class-Scatter-Matrix.
            - Sw: 2D numpy array of Within-Class-Scatter-Matrix.
            - eigenvalues: Fitted eigenvalues for generalized eigenvalue
              problem.
            - eigenvectors: 2D numpy array (m, m). Contains eigenvectors.
            - rotated_data: 2D numpy array (n, m). Contains projected points.
            - inertia: 1D numpy array. Contains calculated inertia.

        """
        _, m = X.shape
        data_cols = [i for i in range(m) if i != g]
        self.data = X[:, data_cols]
        self.groups = X[:, g]
        self.m = m - 1

        groups = np.unique(self.groups)
        group_idx = [self.groups == c for c in groups]
        self.grouped = [self.data[c, :] for c in group_idx]

        self.overall_mean = self.data.mean(axis=0)
        self.grouped_mean = [c.mean(axis=0) for c in self.grouped]

        # Calculate between scatter matrix.
        centered_class_mean = [c-self.overall_mean for c in self.grouped_mean]
        self.Sb = sum([np.outer(c, c) for c in centered_class_mean])

        # Calculate within scatter matrix.
        inner_sums = []
        for x_ci, mean_c in zip(self.grouped, self.grouped_mean):
            # Compute inner sum
            S = np.dot((x_ci - mean_c).T,
                       (x_ci - mean_c))

            inner_sums.append(S)

        # Compute outer sum
        self.Sw = sum(inner_sums)

        # Solve Eigenvalue Problem.
        # Decompose Sb to Sb05.
        evalSb, evecSb = eigen.jacobi(self.Sb)
        Lambda05 = np.diag(-np.sqrt(np.abs(evalSb)))
        Sb05 = np.dot(evecSb, Lambda05).dot(evecSb.T)

        # Calculate Inverse of Sw.
        Sw_inv = np.linalg.inv(self.Sw)

        self.eigenvalues, self.eigenvectors = eigen.jacobi(
            np.dot(Sb05, Sw_inv).dot(Sb05))
        # Retransform vector of weights.
        W = np.dot(np.linalg.inv(Sb05), self.eigenvectors)

        # Calculate inertia.
        self.inertia = self.eigenvalues / self.eigenvalues.sum()

        # Rotate the data.
        self.rotated_data = self.data.dot(W)

    def scree(self):
        """
        Return scree plot for the supplied data.
        """
        ax = plt.plot(self.inertia)
        ax = plt.scatter(x=range(self.m),
                         y=self.inertia)
        plt.axhline(y=0.05, c="gray", ls="--")
        plt.xticks(range(self.m),
                   range(1, self.m+1))
        plt.grid(True)
        plt.title("LDA: Scree plot")
        plt.xlabel("Eigenvalues")

        return ax

    def plot(self, x, y, **kwargs):
        """
        Return 2-dimensional plot for 2 Linear Discriminants.

        Parameters:
            - x: Integer indicating which Linear Discriminant to plot on
                 x-axis.
            - y: Integer indicating which Linear Discriminant to plot on
                 y-axis.
            - kwargs: keyword arguments to pyplot's scatter function.
        """
        assert type(x) == type(y) == int
        assert x in range(self.m) and y in range(self.m)
        ax = plt.scatter(x=self.rotated_data[:, x],
                         y=self.rotated_data[:, y],
                         **kwargs)
        plt.grid(True)
        plt.xlabel(str(x+1) + ". Linear Discriminant")
        plt.ylabel(str(y+1) + ". Linear Discriminant")
        plt.title("LDA: Rotation")

        return ax
