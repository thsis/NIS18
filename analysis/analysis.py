import os
import time
import threading
import numpy as np
import pandas as pd
from algorithms import eigen
from textwrap import dedent

data_out = os.path.join("analysis", "analysis.csv")
trials = pd.DataFrame(columns=["algorithm", "dimension", "iterations", "time"])

# Flags:
ALGORITHMS = [eigen.jacobi, eigen.qrm, eigen.qrm2, eigen.qrm3]
DIMENSIONS = range(3, 20)
N = 100
N_WORKERS = N // 10

trials = pd.DataFrame(columns=["algorithm", "dimension", "iterations", "time"])


def test(func, Ntests, dim, *args, **kwargs):
    global trials
    for _ in range(Ntests):
        X = np.random.uniform(low=-100, high=100, size=(dim, dim))
        X += X.T
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
