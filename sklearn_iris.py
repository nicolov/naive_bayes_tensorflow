#!/usr/bin/env python3

"""
Flower classification on the Iris dataset using a Naive Bayes
classifier and TensorFlow.

For more info: http://nicolovaligi.com/naive-bayes-tensorflow.html
"""

from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

if __name__ == '__main__':
    iris = datasets.load_iris()
    # Only take the first two features
    X = iris.data[:, :2]
    y = iris.target

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Fit the Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X, y)

    # Classify a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z[:, 1].reshape(xx.shape)
    Z2 = Z[:, 2].reshape(xx.shape)

    # Plot
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    # Swap signs to make the contour dashed (MPL default)
    ax.contour(xx, yy, -Z1, [-0.5], colors='k')
    ax.contour(xx, yy, -Z2, [-0.5], colors='k')

    # Plot formatting
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('sklearn decision boundary')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    plt.tight_layout()
    fig.savefig('iris_sklearn.png', bbox_inches='tight')
