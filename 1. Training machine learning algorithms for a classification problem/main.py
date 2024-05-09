import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron
from plot_utils import plot_decision_regions

data_path = "iris.data"
df = pd.read_csv(data_path)

print(df.tail())

Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show(block=False)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, Y)

plt.figure()
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('number of misclassification cases')
plt.show(block=False)

plt.figure()
plot_decision_regions(X, Y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')

plt.show()