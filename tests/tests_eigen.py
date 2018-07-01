"""
Automated tests for different algorithms.
"""
import os
import sys
import numpy as np
from threading import Thread
import pandas as pd
from algorithms import eigen
from scipy.stats import ortho_group
from tqdm import tqdm
from functools import wraps


data_out = os.path.join("data", "accuracy_tests.csv")


class AlgoTest(object):
    tests = {"algorithm": [],
             "dimension": [],
             "maxiter": [],
             "failed": []}

    def __init__(self, algo, dim, filepath, n_tests=1000, jobs=1,
                 *args, **kwargs):
        assert n_tests % jobs == 0
        self.algorithm = self.__get_test_algorithm(algo, *args, **kwargs)
        self.dim = dim
        self.n = n_tests // jobs
        self.failed = []
        self.jobs = jobs
        self.result = None
        self.path = filepath
        self.maxiter = kwargs.get("maxiter", None)

        if not os.path.exists(self.path):
            self.save(header=["algorithm", "dimension", "maxiter", "failed"])

    def __get_test_algorithm(self, algorithm, *args, **kwargs):
        @wraps(algorithm)
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
            try:
                self.__run_test()
            except (KeyboardInterrupt, SystemExit):
                self.__save()
                sys.exit(0)

    def run(self):
        """Distribute tests accross threads."""

        threadlist = [None] * self.jobs

        for i in range(self.jobs):
            threadlist[i] = Thread(target=self.__run_tests, daemon=True)
            threadlist[i].start()

        for thread in threadlist:
            thread.join()

        self.result = sum(self.failed)
        self.tests["algorithm"].append(self.algorithm.__name__)
        self.tests["dimension"].append(self.dim)
        self.tests["failed"].append(self.result)
        if self.algorithm.__name__ == "jacobi":
            self.tests["maxiter"].append(None)
        else:
            self.tests["maxiter"].append(self.maxiter)

    def save(self, header=False):
        if os.path.exists(self.path):
                print("Saving results.")
        df = pd.DataFrame(self.tests)
        with open(self.path, 'a') as f:
            df.to_csv(f, header=header, index=False,
                      columns=["algorithm", "dimension", "maxiter", "failed"])


# Unit tests
if __name__ == "__main__":
    # Define Flags.
    JOBS = 20
    MAXITER = (10, 100, 1000, 10000, 100000)
    DIMS = range(3, 8)
    ALGOS = {'jacobi': eigen.jacobi,
             'qrm': eigen.qrm,
             'qrm2': eigen.qrm2,
             'qrm3': eigen.qrm3}

    # Define parameters of all runs.
    parameters = []
    for algo in ALGOS.values():
        for maxiter in MAXITER:
            for dim in range(3, 8):
                param = (algo,
                         maxiter if algo.__name__ != "jacobi" else None,
                         dim)
                parameters.append(param)

    # Check progress of previous runs.
    if os.path.exists(data_out):
        print("Check progress:")
        required = [(algo.__name__, m, dim) for (algo, m, dim) in parameters]
        required = set(required)

        progress = pd.read_csv(data_out)

        done = []
        for (a, d, m, _) in progress.values:
            if np.isnan(m):
                param = (a, None, d)
            else:
                param = (a, int(m), d)
            done.append(param)
        done = set(done)

        to_do = required.difference(done)
        parameters = [(ALGOS[a], m, d) for a, m, d in to_do]

    for algo, maxiter, dim in tqdm(parameters):
        if algo is eigen.jacobi:
            algo_test = AlgoTest(filepath=data_out,
                                 algo=algo,
                                 dim=dim,
                                 jobs=JOBS)
        else:
            algo_test = AlgoTest(filepath=data_out,
                                 algo=algo,
                                 dim=dim,
                                 maxiter=maxiter,
                                 jobs=JOBS)
        algo_test.run()

    algo_test.save()

    results = pd.read_csv(data_out)
    results.head(10)
    results.loc[results.algorithm == 'jacobi', 'maxiter'] = '-'

    crosstab = pd.crosstab(index=[results.algorithm, results.maxiter],
                           columns=results.dimension,
                           values=results.failed,
                           aggfunc=np.mean,
                           dropna=True)
    print(crosstab.to_latex())
