import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import utils

"""
Logistic Regression Experiments

This script explores Logistic Regression and compares it with LDA using
synthetic datasets.  It generates non-Gaussian data and data with outliers
to analyze the performance of Logistic Regression in different scenarios.
"""

if __name__ == '__main__':
    # Generate non-Gaussian data
    X1 = utils.gaussian_sample(n=50)
    binom = np.random.binomial(1, 0.5, size=50)
    X2 = np.array([x * utils.gaussian_sample(mu=[5, 3], n=1) + (1 - x) * utils.gaussian_sample(mu=[8, 9], n=1) for x in binom])

    myX = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((np.full(50, 0), np.full(50, 1)))

    # Apply Logistic Regression
    lreg = LogisticRegression()
    lreg.fit(myX, y)

    # Plot the decision boundary
    plt.figure()
    utils.plotXY(myX, y)
    utils.plot_frontiere(lreg, myX, label=y)
    plt.title("Logistic Regression with Non-Gaussian Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Dataset with an outlier
    X1 = utils.gaussian_sample(mu=[0, 0])
    X2 = utils.gaussian_sample(mu=[5, 3], n=49)
    X3 = utils.gaussian_sample(mu=[20, 20], n=1).reshape(1, -1)

    X = np.r_[X1, X2, X3]
    Y = np.r_[np.ones(X1.shape[0]), -np.ones(X2.shape[0]), -np.ones(X3.shape[0])]

    # Fit the Logistic Regression model
    lreg.fit(X, Y)

    # Plot the decision boundary with outlier
    plt.figure()
    utils.plotXY(X, Y)
    utils.plot_frontiere(lreg, X, label=Y)
    plt.title("Logistic Regression with Outlier")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
