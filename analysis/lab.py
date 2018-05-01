"""
Demonstrate Various Algorithms or prerequisites.

    - Householder reflections
    - Givens Rotations used in Jacobi-Transform and QR-Method.
"""

import numpy as np
import copy
from matplotlib import pyplot as plt


def hreflect1D(x):
    """
    Calculate Householder reflection: Q = I - 2*uu'.

    Parameters:
        X: numpy array.

    Returns:
        q: reflected vector.
        Q: Reflector (matrix).
    """
    # Construct v:
    v = copy.deepcopy(x)
    v[0] += np.linalg.norm(x)

    # Construct u: normalize v.
    vnorm = np.linalg.norm(v)
    if vnorm:
        u = v / np.linalg.norm(v)
    else:
        u = v

    # Construct Q:
    Q = np.eye(len(x)) - 2 * np.outer(u, u)

    return np.dot(Q, x), Q


def qr_factorize(X):
    """
    Compute QR factorization of X s.t. QR = X.

    Parameters:
        - X: square numpy ndarray.

    Returns:
        Q: square numpy ndarray, same shape as X. Rotation matrix.
        R: square numpy ndarray, same shape as X. upper triangular matrix.
    """
    assert type(X) == np.ndarray
    assert X.shape[0] == X.shape[1]

    A = copy.deepcopy(X)
    Q = np.eye(X.shape[0])

    for i in range(X.shape[0]):
        Pi = np.eye(A.shape[0])
        _, Qi = hreflect1D(A[i:, i])
        Pi[i:, i:] = Qi

        Q = Pi.dot(Q)
        A = Pi.dot(A)

    return Q, A


A = np.array([[3, -98/28, 1, 1, 1],
              [1, 122/28, 1, 3, 1],
              [2, -8/28, 89, 1, 1],
              [1, 66/28, 1, 3, 1],
              [1, 10/28, 1, 1, 2]])

Q, R = qr_factorize(A)

Q
R
np.round(R)

np.isclose(A, np.linalg.inv(Q).dot(R))

hreflect1D(np.array([5, 1, 3, 1]))

1/66 * np.array([[-55, -11, -33, -11],
                 [-11, 65, -3, -1],
                 [-33, -3, 57, -3],
                 [-11, -1, -3, 65]])
