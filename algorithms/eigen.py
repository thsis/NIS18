"""
Algorithms for solving eigenvalue problems.

1. Compute diagonalization of 2x2 matrices via jacobi iteration.
2. Generalize Jacobi iteration for symmetric matrices.
"""
import numpy as np
import copy


def isDiagonal(X, precision=1e-15):
    nonZeros = np.abs(X - np.diag(X.diagonal())).sum()
    return nonZeros < precision


def jacobi2x2(A):
    """
    Diagonalize a 2x2 matrix through jacobi step.

    Solve: U' A U = E s.t. E is a diagonal matrix.

    Parameters:
        A - 2x2 numpy array.
    Returns:
        A 2x2 diagonal numpy array
    """
    assert type(A) == np.ndarray
    assert A.shape == (2, 2)
    assert A[1, 0] == A[0, 1]

    alpha = 0.5 * np.arctan(2*A[0, 1]/(A[1, 1] - A[0, 0]))
    U = np.array([[np.cos(alpha), np.sin(alpha)],
                  [-np.sin(alpha), np.cos(alpha)]])
    E = np.matmul(U.T, np.matmul(A, U))
    return E


def jacobi(X, precision=1e-6):
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
    assert all((X - X.T == 0).flatten())
    A = copy.deepcopy(X)
    U = np.eye(A.shape[0])
    L = np.array([1])

    while L.max() > precision:
        L = np.abs(np.tril(A, k=0) - np.diag(A.diagonal()))
        i, j = np.unravel_index(L.argmax(), L.shape)
        alpha = 0.5 * np.arctan(2*A[i, j] / (A[i, i]-A[j, j]))

        V = np.eye(A.shape[0])
        V[i, i], V[j, j] = np.cos(alpha), np.cos(alpha)
        V[i, j], V[j, i] = -np.sin(alpha), np.sin(alpha)

        A = np.dot(V.T, A.dot(V))
        U = U.dot(V)

    # Sort by eigenvalue (descending order) and flatten A
    A = np.diag(A)
    order = np.abs(A).argsort()[::-1]

    return A[order], U[:, order]
