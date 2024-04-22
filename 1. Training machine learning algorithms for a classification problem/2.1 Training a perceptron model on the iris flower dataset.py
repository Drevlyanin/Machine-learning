import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Use a file with data about iris flowers that I downloaded
from the Machine Learning Repository at the University
of California at Irvine.
"""
data_path = "E:\\Py\\virtual\\Машинное обучение\\1. Training machine learning algorithms for a classification problem\\iris.data"
df = pd.read_csv(data_path)

"""
Display the last few lines on the screen using the
tail method to make sure that the file is intact.
"""
print(df.tail())

"""
Extract the first 100 class labels, which correspond to 50 Iris bristles flowers
and 50 Iris versicolor flowers, and convert the class labels into two integer class labels
(Versicolor variegated) and -1 (Versicolor setosa) - which we assign to the vector y,
where method values of the pandas DataFrame object generates a representation corresponding
to the NumPy library. Similarly, from these 100 training samples,
extract the first feature column (sepal length) and the third feature column (petal length)
and assign them to a feature matrix X, which can be visualized using a two-dimensional
scatter plot (also called a scatterplot).
"""
Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)  # Corrected usage of Y
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa)')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')

plt.legend(loc='upper left')

plt.show()
