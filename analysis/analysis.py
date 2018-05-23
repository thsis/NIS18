import os
import pandas as pd
from matplotlib import pyplot as plt

datadir = os.path.join("analysis", "analysis.csv")
trials = pd.read_csv(datadir, index_col=0)
trials.head(10)

# Boxplot iteration:
trials[trials.dimension == 3].groupby("algorithm").iterations.describe()
plt.show()
