import os
import numpy as np
import pandas as pd
from seaborn import pairplot, xkcd_palette
from models import lda, pca
from matplotlib import pyplot as plt


dataout = os.path.join("media", "plots")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

class_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2}

data = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length',
                               'petal width', 'target'],
                   converters={'target': lambda x: class_mapping[x]})

# Raw data:
pairplot(data, hue='target', vars=['sepal length', 'sepal width',
                                   'petal length', 'petal width'],
         palette=xkcd_palette(['dark purple', 'teal', 'yellow']))

plt.savefig(os.path.join(dataout, "iris_raw.png"))
plt.close()

features = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
features = features.values

targets = data['target'].values

# Prepare plots for presentation:
# PCA
iris_pca = pca.PCA()
iris_pca.fit(features, norm=False)

plt.figure(1, figsize=(5, 8))
plt.subplot(211)
iris_pca.scree()
plt.subplot(212)
iris_pca.plot(0, 1, c=targets)
plt.tight_layout()

plt.savefig(os.path.join(dataout, "iris_pca.png"))
plt.close()

# LDA
iris_lda = lda.LDA()
iris_lda.fit(data.values, 4)

plt.figure(1, figsize=(5, 8))
plt.subplot(211)
iris_lda.scree()
plt.subplot(212)
iris_lda.plot(1, 0, c=targets)
plt.tight_layout()

plt.savefig(os.path.join(dataout, "iris_lda.png"))
plt.close()
