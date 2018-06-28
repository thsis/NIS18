"""
Automated tests for different algorithms.
"""
import os
import numpy as np
from threading import Thread
import pandas as pd
from algorithms import eigen
from scipy.stats import ortho_group
from tqdm import trange, tqdm

data_out = os.path.join("data", "accuracy_tests.csv")


class AlgoTest(object):
    def __init__(self, algo, dim, n_tests=1000, jobs=1, *args, **kwargs):
        assert n_tests % jobs == 0
        self.algorithm = self.__get_test_algorithm(algo, *args, **kwargs)
        self.dim = dim
        self.n = n_tests // jobs
        self.failed = []
        self.jobs = jobs
        self.result = None

    def __get_test_algorithm(self, algorithm, *args, **kwargs):
        def algo(X):
            return algorithm(X, *args, **kwargs)
        return algo

    def __get_test_matrix(self):
        """Return matrix with assosiated Eigenvalues."""
        eigenvalues = np.random.uniform(size=self.dim)
        eigenvectors = ortho_group.rvs(dim=self.dim)
        Lambda = np.diag(eigenvalues)

        matrix = np.dot(eigenvectors, Lambda).dot(eigenvectors.T)

        order = np.abs(eigenvalues).argsort()[::-1]
        return matrix, eigenvalues[order]

    def __run_test(self):
        """Run singular test."""
        mat, true_eig = self.__get_test_matrix()
        test_eig, _ = self.algorithm(mat)
        test_res = np.alltrue(np.isclose(true_eig, test_eig))
        self.failed.append(not test_res)

    def __run_tests(self):
        """Run multiple tests."""
        for _ in range(self.n):
            self.__run_test()

    def run(self):
        """Distribute tests accross threads."""

        threadlist = [None] * self.jobs

        for i in range(self.jobs):
            threadlist[i] = Thread(target=self.__run_tests)
            threadlist[i].start()

        for thread in threadlist:
            thread.join()

        self.result = sum(self.failed)


# Unit tests
algotest = AlgoTest(algo=eigen.qrm2, dim=3, n_tests=1000, jobs=10, maxiter=20)
algotest.run()
algotest.failed
algotest.result
