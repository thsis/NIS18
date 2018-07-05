"""
Algorithms for solving eigenvalue problems.

1. Compute diagonalization of 2x2 matrices via jacobi iteration.
2. Generalize Jacobi iteration for symmetric matrices.
"""
import numpy as np
import copy
import warnings
from scipy import linalg as lin
from algorithms import helpers


def jacobi2x2(A):
    """
    Diagonalize a 2x2 matrix through jacobi step.

    Solve: U' A U = E s.t. E is a diagonal matrix.

    Parameters:
        A - 2x2 numpy array.
    Returns:
        A - 2x2 diagonal numpy array
    """
    assert type(A) == np.ndarray
    assert A.shape == (2, 2)
    assert A[1, 0] == A[0, 1]

    alpha = 0.5 * np.arctan(2*A[0, 1]/(A[1, 1] - A[0, 0]))
    U = np.array([[np.cos(alpha), np.sin(alpha)],
                  [-np.sin(alpha), np.cos(alpha)]])
    E = np.matmul(U.T, np.matmul(A, U))
    return E


def jacobi(X, precision=1e-6, debug=False):
    """
    Compute Eigenvalues and Eigenvectors for symmetric matrices.

    Parameters:
        X - 2D numpy ndarray which represents a symmetric matrix
        precision - float in (0, 1). Convergence criterion.

    Returns:
        A - 1D numpy array with eigenvalues sorted by absolute value
        U - 2D numpy array with associated eigenvectors (column).
    """
    assert 0 < precision < 1.
    assert type(X) == np.ndarray
    n, m = X.shape
    assert n == m
    assert all(np.isclose(X - X.T, np.zeros(n)).flatten())
    A = copy.deepcopy(X)
    U = np.eye(A.shape[0])
    L = np.array([1])
    iterations = 0

    while L.max() > precision:
        L = np.abs(np.tril(A, k=0) - np.diag(A.diagonal()))
        i, j = np.unravel_index(L.argmax(), L.shape)
        alpha = 0.5 * np.arctan(2*A[i, j] / (A[i, i]-A[j, j]))

        V = np.eye(A.shape[0])
        V[i, i], V[j, j] = np.cos(alpha), np.cos(alpha)
        V[i, j], V[j, i] = -np.sin(alpha), np.sin(alpha)

        A = np.dot(V.T, A.dot(V))
        U = U.dot(V)
        iterations += 1

    # Sort by eigenvalue (descending order) and flatten A
    A = np.diag(A)
    order = np.abs(A).argsort()[::-1]
    if debug:
        return iterations

    return A[order], U[:, order]


def qrm(X, maxiter=15000, debug=False):
    """
    Compute Eigenvalues and Eigenvectors using the QR-Method.

    Parameters:
        - X: square numpy ndarray.
    Returns:
        - Eigenvalues of A.
        - Eigenvectors of A.
    """
    n, m = X.shape
    assert n == m

    # First stage: transform to upper Hessenberg-matrix.
    A = copy.deepcopy(X)
    conv = False
    k = 0

    # Second stage: perform QR-transformations.
    while (not conv) and (k < maxiter):
        k += 1
        Q, R = helpers.qr_factorize(A)
        A = R.dot(Q)

        conv = np.alltrue(np.isclose(np.tril(A, k=-1), np.zeros((n, n))))

    if not conv:
        warnings.warn("Convergence was not reached. Consider raising maxiter.")
    if debug:
        return k
    Evals = A.diagonal()
    order = np.abs(Evals).argsort()[::-1]
    return Evals[order], Q[order, :]


def qrm2(X, maxiter=15000, debug=False):
    """
    First compute similar matrix in Hessenberg form, then compute the
    Eigenvalues and Eigenvectors using the QR-Method.

    Parameters:
        - X: square numpy ndarray.
    Returns:
        - Eigenvalues of A.
        - Eigenvectors of A.
    """
    n, m = X.shape
    assert n == m

    # First stage: transform to upper Hessenberg-matrix.
    A = lin.hessenberg(X)
    conv = False
    k = 0

    # Second stage: perform QR-transformations.
    while (not conv) and (k < maxiter):
        k += 1
        Q, R = helpers.qr_factorize(A)
        A = R.dot(Q)

        conv = np.alltrue(np.isclose(np.tril(A, k=-1), np.zeros((n, n))))

    if not conv:
        warnings.warn("Convergence was not reached. Consider raising maxiter.")
    if debug:
        return k
    Evals = A.diagonal()
    order = np.abs(Evals).argsort()[::-1]
    return Evals[order], Q[order, :]


def qrm3(X, maxiter=15000, debug=False):
    """
    First compute similar matrix in Hessenberg form, then compute the
    Eigenvalues and Eigenvectors using the QR-Method.

    Parameters:
        - X: square numpy ndarray.
    Returns:
        - Eigenvalues of A.
        - Eigenvectors of A.
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

    if not conv:
        warnings.warn("Convergence was not reached. Consider raising maxiter.")
    if debug:
        return k
    Evals = T.diagonal()
    order = np.abs(Evals).argsort()[::-1]
    return Evals[order], Q[order, :]


def eigen(X):
    """
    Compute eigenvalues and eigenvectors of X.

    Parameters:
        - X: square numpy ndarray.
    Returns:
        - Eigenvalues of A.
        - Eigenvectors of A.

    """

    symmetric = np.alltrue(np.isclose(X - X.T, np.zeros(n)))
    small = max(X.shape) <= 11

    if symmetric:
        return jacobi(X)
    elif small:
        maxiter = 10 ** max(*X.shape, 4)
        return qrm3(X, maxiter=maxiter)
    else:
        maxiter = 10 ** max(*X.shape, 4)
        return qrm2(X, maxiter=maxiter)
