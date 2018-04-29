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


def jacobi(X, precision=1e-20):
    assert type(X) == np.ndarray
    assert all((X - X.T == 0).flatten())
    A = copy.deepcopy(X)

    while not isDiagonal(A, precision=precision):
        L = np.tril(A, k=0) - np.diag(A.diagonal())
        i, j = np.unravel_index(L.argmax(), L.shape)
        alpha = 0.5 * np.arctan(2*A[i, j] / (A[i, i]-A[j, j]))
        U, V = np.eye(A.shape[0]), np.eye(A.shape[0])
        V[i, i], V[j, j] = np.cos(alpha), np.cos(alpha)
        V[i, j], V[j, i] = -np.sin(alpha), np.sin(alpha)

        A = np.matmul(V.T, np.matmul(A, V))
        U = np.matmul(U, V)

    return A, U


X = np.array([[1, 0, 0], [0, 1, 2], [0, 2, 4]])
A, U = jacobi(X)
a, u = np.linalg.eig(X)
A
a
U
u
