"""
Automated tests for different algorithms.
"""

import numpy as np
import threading
from algorithms import eigen
from scipy.stats import ortho_group


def get_test_matrix(dim):
    """Return matrix with assosiated Eigenvalues."""
    eigenvalues = np.random.uniform(size=dim)
    eigenvectors = ortho_group.rvs(dim=dim)
    Lambda = np.diag(eigenvalues)

    matrix = np.dot(eigenvectors, Lambda).dot(eigenvectors.T)

    order = np.abs(eigenvalues).argsort()[::-1]
    return matrix, eigenvalues[order]


def test_algo(algo, Ntests=1000, dim=3, *args, **kwargs):
    """
    Test routine that allows for threading. Note that the variables:
    failed, critical and problematic need to be defined in the enveloping or
    global scope beforehand.

    Parameters:
        - algo: algorithm to be tested
        - Ntests: number of tests to compute
        - dim: dimensions of matrix
        - *args, **kwargs: additional arguments to be passed to algo.

    Returns:
        - None, but will update the variables failed, critical and problematic.
            + failed: number of failed tests
            + critical: number of ZeroDivisionErrors
            + problematic: list of numpy arrays which led to wrong eigenvalues.
    """
    global failed
    global critical
    global problematic

    for _ in range(Ntests):
        try:
            A, true_eig = get_test_matrix(dim=dim)
            my_eig, _ = algo(A, *args, **kwargs)
            assert np.alltrue(np.isclose(my_eig, true_eig))

        except AssertionError:
            failed += 1
            problematic.append(A)

        except ZeroDivisionError:
            critical += 1



def threaded_tests(algo, N, nWorkers=10, *args, **kwargs):
    global failed
    global critical
    global problematic

    assert N % nWorkers == 0

    n = N // nWorkers
    threadlist = [None] * nWorkers

    for i in range(nWorkers):
        threadlist[i] = threading.Thread(target=test_algo,
                                         args=(algo, n, *args))
        threadlist[i].start()

    for i in range(nWorkers):
        threadlist[i].join()

    logstr = """
    {} out of {} tests failed.
    {} tests failed critically.
    """.format(failed, N, critical)

    print(logstr)


# Tests
problematic = []
failed = 0
critical = 0
print("Test Jacobi Method:")
threaded_tests(eigen.jacobi, 1000, 20, 15)

failed = 0
critical = 0
print("Test QR-Method1:")
threaded_tests(eigen.qrm, 1000, 10, 15, np.inf)

print("Test QR-Method2:")
failed = 0
critical = 0
threaded_tests(eigen.qrm2, 1000, 20, 15, np.inf)

failed = 0
critical = 0
print("Test QR-Method3:")
threaded_tests(eigen.qrm3, 1000, 20, 15, np.inf)
