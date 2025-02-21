import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import utils

"""
AdaBoost Experiments

This script explores the AdaBoost algorithm with different datasets and
configurations. It includes experiments with synthetic Gaussian data and the
digits dataset.
"""

if __name__ == '__main__':
    # Dataset for AdaBoost
    X1 = utils.gaussian_sample(mu=[0, 0], sigma1=10, theta=np.pi/6)
    X2 = utils.gaussian_sample(mu=[5, 3], sigma1=3, sigma2=10, theta=np.pi/6, n=50)
    X3 = utils.gaussian_sample(mu=[-5, -2], sigma1=3, sigma2=10, theta=np.pi/10, n=50)

    X = np.r_[X1, X2, X3]
    Y = np.r_[np.ones(X1.shape[0]), -np.ones(X2.shape[0]), -np.ones(X3.shape[0])]

    # Fit an AdaBoost classifier with SAMME algorithm
    adaclf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME', random_state=42)
    adaclf.fit(X, Y)

    # Plot the regions
    plt.figure()
    utils.plotXY(X, Y)
    utils.map_regions(adaclf, data=X)
    plt.title("AdaBoost with SAMME")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Plot the estimator errors
    plt.figure()
    plt.plot(adaclf.estimator_errors_, label='Estimator Errors')
    plt.xlabel("Estimator Number")
    plt.ylabel("Error")
    plt.title("AdaBoost Estimator Errors")
    plt.legend()
    plt.show()

    # Load the digits dataset
    digits = load_digits()

    print("# of observations:", digits.data.shape[0])
    print("# of covariates:", digits.data.shape[1])
    print("# of classes:", digits.target_names.shape[0])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)

    # Train and evaluate AdaBoost with SAMME
    dtclf = DecisionTreeClassifier(max_depth=5, random_state=42)
    ada_clf_samme = AdaBoostClassifier(dtclf, n_estimators=200, algorithm='SAMME', random_state=42)
    ada_clf_samme.fit(X_train, y_train)

    # Plot train and test scores
    plt.figure()
    plt.plot(list(ada_clf_samme.staged_score(X_train, y_train)), label="SAMME train")
    plt.plot(list(ada_clf_samme.staged_score(X_test, y_test)), label="SAMME test")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Score")
    plt.title("AdaBoost Performance on Digits Dataset (SAMME)")
    plt.legend()
    plt.show()
