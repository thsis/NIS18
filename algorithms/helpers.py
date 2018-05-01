import numpy as np
import copy


def hreflect1D(x):
    """
    Calculate Householder reflection: Q = I - 2*uu'.

    Parameters:
        X: numpy array.

    Returns:
        Qx: reflected vector.
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
    Qx = np.dot(Q, x)

    return Qx, Q


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

    R = copy.deepcopy(X)
    Q = np.eye(X.shape[0])

    for i in range(X.shape[0]):
        Pi = np.eye(R.shape[0])
        _, Qi = hreflect1D(R[i:, i])
        Pi[i:, i:] = Qi

        Q = Pi.dot(Q)
        R = Pi.dot(R)

    return np.linalg.inv(Q), R
