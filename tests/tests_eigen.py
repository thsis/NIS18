"""
Automated tests for different algorithms.

Test diagonalization of 2D matrices.
Test Computation of eigenvalues of arbitrary matrices.
"""
import numpy as np
import threading
from algorithms import eigen
from tqdm import tqdm
from scipy.stats import ortho_group


def getSymmetricMatrix(dist=np.random.uniform,
                       shape=(2, 2), **kwargs):
    "Create random symmetric matrix for test purposes."
    X = dist(size=shape, **kwargs)

    return X + X.T


def getAlmostDiagonal(dist=np.random.uniform, shape=(2, 2),
                      **kwargs):
    "Create random symmetric matrix, that is already close to a diagonal."
    X = np.eye(N=shape[0])
    i, j = np.random.randint(0, shape[0])
    replace = dist(size=1, **kwargs)

    X[i, j], X[j, i] = replace, replace
    return X


def testDiagonal(Ntests, precision=1e-12):
    passed = 0
    critically = 0
    for _ in tqdm(range(Ntests)):
        X = getSymmetricMatrix()
        E = eigen.jacobi2x2(X)
        try:
            assert eigen.isDiagonal(E, precision=precision)
            passed += 1
        except AssertionError:
            print(E)
            continue
        except ZeroDivisionError:
            print("X:\n", X)
            critically += 1
    print("{} out of {} tests passed.".format(passed, Ntests))
    print("Critically failed {} tests.".format(critically))
    if passed == Ntests:
        return True
    else:
        return False


def testEigen(fun, Ntests, *args, **kwargs):
    wrongValues, critical = 0, 0
    shenanigans = []
    for _ in tqdm(range(Ntests)):
        n = np.random.randint(3, 5)
        X = getSymmetricMatrix(shape=(n, n))
        myEigenVal, _ = fun(X, *args, **kwargs)

        trueEigenVal, _ = np.linalg.eig(X)
        order = np.abs(trueEigenVal).argsort()[::-1]
        trueEigenVal = trueEigenVal[order]

        # Test Eigenvalues:
        testedValues = all(np.isclose(myEigenVal, trueEigenVal))
        try:
            assert testedValues
        except AssertionError:
            print("Encountered Error:")
            print("Custom Eigenvalues: {}".format(myEigenVal))
            print("Numpy Eigenvalues: {}".format(trueEigenVal))
            shenanigans.append(X)
            wrongValues += 1
        except ZeroDivisionError:
            critical += 1
    passed = Ntests - wrongValues
    print("{} out of {} tests passed.".format(passed, Ntests))
    print("Wrong Eigenvalues: {}".format(wrongValues))
    print("{} tests failed critically.".format(critical))
    if passed == Ntests:
        return True, shenanigans
    else:
        return False, shenanigans


# Tests
# Test: Jacobi Computation of Eigenvalues/Eigenvectors.
test_jacobi, failed_jacobi = testEigen(eigen.jacobi, 100)
assert test_jacobi
# Test: QR-Method for Eigenvalues.
test_qr, failed_qr = testEigen(eigen.qrm2, 1000, maxiter=5000)

thread0 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread1 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread2 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread3 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread4 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread5 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread6 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread7 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread8 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()
thread9 = threading.Thread(target=testEigen, args=(eigen.jacobi, 100)).start()


def get_test_matrix(dim):
    """Return matrix with assosiated Eigenvalues."""
    eigenvalues = np.random.uniform(size=dim)
    eigenvectors = ortho_group.rvs(dim=dim)
    Lambda = np.diag(eigenvalues)

    matrix = np.dot(eigenvectors, Lambda).dot(eigenvectors.T)

    order = np.abs(eigenvalues).argsort()[::-1]
    return matrix, eigenvalues[order]


def testEigenAlgorithm(algo, Ntests=1000, dim=3, *args, **kwargs):
    failed = 0
    critical = 0
    shenanigans = []
    for _ in tqdm(range(Ntests)):
        try:
            A, true_eig = get_test_matrix(dim=dim)
            my_eig, _ = algo(A, *args, **kwargs)
            assert all(np.isclose(my_eig, true_eig).flatten())
        except AssertionError:
            failed += 1
            shenanigans.append(A)
            print("Encountered Error")
        except ZeroDivisionError:
            critical += 1

    print("{} out of {} tests failed.".format(failed, Ntests))
    print("{} tests failed critically.".format(critical))

    if failed == 0:
        return True, shenanigans
    else:
        return False, shenanigans


res, bogus = testEigenAlgorithm(eigen.qrm2, dim=5, maxiter=10000)


res2, bogus2 = testEigenAlgorithm(eigen.qrm2, maxiter=5000)
