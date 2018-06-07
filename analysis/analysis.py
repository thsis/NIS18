import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

datadir = os.path.join("analysis", "benchmarks.csv")
trials = pd.read_csv(datadir, index_col=0)

trials.groupby(["algorithm", "dimension"]).iterations.describe()

# Boxplot iteration:
fig = plt.figure(figsize=(10, 5))
sns.boxplot(x="dimension", y="iterations", hue="algorithm", data=trials)
plt.yscale("log")
plt.title("Iterations needed before Convergence")
plt.savefig(os.path.join("media", "plots", "iterations_boxplot.png"))
plt.show()
plt.close()

# Boxplot elapsed time:
fig = plt.figure(figsize=(10, 5))
sns.boxplot(x="dimension", y="time", hue="algorithm", data=trials)
plt.title("Time needed before Convergence")
plt.ylabel("time (sec)")
plt.yscale('log')
plt.savefig(os.path.join("media", "plots", "time_boxplot.png"))
plt.show()
plt.close()
