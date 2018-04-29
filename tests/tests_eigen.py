"""
Automated tests for different algorithms.

Test diagonalization of 2D matrices.
Test diagonalization of 3D matrices.
Test diagonalization of arbitrary matrices.
"""
import numpy as np
from algorithms import eigen


def getRandomMatrix(dist=np.random.uniform, shape=(2, 2), **kwargs):
    "Create random symmetric matrix for test purposes."
    X = dist(size=shape, **kwargs)

    return X + X.T


def testDiagonal(Ntests):
    passed = 0
    for i in range(Ntests):
        X = getRandomMatrix()
        E = eigen.jacobi2x2(X)
        try:
            assert eigen.isDiagonal(E)
            passed += 1
        except AssertionError:
            print(E)
            continue
    print("{} out of {} tests passed.".format(passed, Ntests))


# Tests
# Test: Jacobi Diagonalization of 2x2 Matrices.
testDiagonal(1000)
