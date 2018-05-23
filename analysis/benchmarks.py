import os
import time
import threading
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from algorithms import eigen
from textwrap import dedent

data_out = os.path.join("analysis", "analysis2.csv")
trials = pd.DataFrame(columns=["algorithm", "dimension", "iterations", "time"])

# Flags:
ALGORITHMS = [eigen.jacobi, eigen.qrm, eigen.qrm2, eigen.qrm3]
DIMENSIONS = range(3, 20)
N = 100
N_WORKERS = 20


def test(func, Ntests, dim, *args, **kwargs):
    global trials
    for _ in range(Ntests):
        eigenvalues = np.random.uniform(size=dim)
        eigenvectors = ortho_group.rvs(dim=dim)
        Lambda = np.diag(eigenvalues)

        X = np.dot(eigenvectors, Lambda).dot(eigenvectors.T)

        start = time.clock()
        n = func(X, debug=True, *args, **kwargs)
        time_elapsed = time.clock() - start
        results = {"algorithm": func.__name__,
                   "dimension": dim,
                   "iterations": n,
                   "time": time_elapsed}
        trials = trials.append(results, ignore_index=True)


def threaded_tests(func, Ntotal, Nworkers, dim):
    assert Ntotal % Nworkers == 0
    nJobs = Ntotal // Nworkers
    threadlist = [None] * Nworkers

    for i in range(Nworkers):
        threadlist[i] = threading.Thread(target=test,
                                         args=(func, nJobs, dim))
        threadlist[i].start()
        threadlist[i].join()


for algo in ALGORITHMS:
    for dim in DIMENSIONS:
        threaded_tests(algo, Ntotal=N, Nworkers=N_WORKERS, dim=dim)
        logstr = dedent("""
        Performed {} Tests for {} with {}-dimensional matrices.""".format(
            N, algo.__name__, dim))
        print(logstr)

trials.to_csv(data_out)
