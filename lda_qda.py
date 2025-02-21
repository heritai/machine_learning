import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, LinearClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.optimize import minimize
import utils

"""
Linear and Quadratic Discriminant Analysis (LDA and QDA)

This script implements LDA and FisherDA from scratch and compares them with
scikit-learn's LDA and QDA. It generates synthetic data to visualize the
decision boundaries of these classifiers.
"""

class LDA(BaseEstimator, LinearClassifierMixin):
    """
    LDA classifier for two classes (implementation from scratch).

    This class implements a linear discriminant analysis classifier from scratch,
    following the scikit-learn estimator API.
    """
    def __init__(self, prior=None):
        """
        Initializes the LDA classifier.

        Args:
            prior (bool, optional): Whether to use prior in the intercept.
                                     Default is None (not used).
        """
        self.prior = prior

    def fit(self, X, y):
        """
        Fits the LDA classifier to the training data.

        Args:
            X (np.ndarray): The training data (features), shape (n_samples, n_features).
            y (np.ndarray): The target variable (labels), shape (n_samples,).
        """
        X1 = X[y == 1]
        X0 = X[y == 0]
        mu1 = np.mean(X0, axis=0)
        mu2 = np.mean(X1, axis=0)
        cov1 = np.cov(X0.transpose())
        cov2 = np.cov(X1.transpose())
        myCov = (cov1 + cov2) / 2  # Pooled covariance matrix
        myCovInv = np.linalg.inv(myCov)
        self.hCoef = np.dot((mu1 - mu2).transpose(), myCovInv)  # Weight vector
        term1 = np.dot(np.dot(mu1.transpose(), myCovInv), mu1)
        term2 = np.dot(np.dot(mu2.transpose(), myCovInv), mu2)
        self.b = (1 / 2) * (term2 - term1)  # Bias term
        print("LDA has been fitted to Data")

    def decision_function(self, X):
        """
        Computes the decision function for the input data.

        Args:
            X (np.ndarray): The input data (features), shape (n_samples, n_features).

        Returns:
            np.ndarray: The decision function values for each sample.
        """
        return np.array([np.dot(self.hCoef, x) + self.b for x in X])

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Args:
            X (np.ndarray): The input data (features), shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted class labels for each sample (0 or 1).
        """
        truFalsePred = self.decision_function(X) >= 0
        return np.array([int(x) for x in truFalsePred])


class FisherDA(BaseEstimator, LinearClassifierMixin):
    """
    Fisher discriminant analysis for two classes (implementation from scratch).
    """

    def fit(self, X, y):
        """
        Fits the FisherDA classifier to the training data.

        Args:
            X (np.ndarray): The training data (features).
            y (np.ndarray): The target variable (labels).
        """
        self.label = y
        X1 = X[y == 1]
        X0 = X[y == 0]
        mu1 = np.mean(X0, axis=0)
        mu2 = np.mean(X1, axis=0)
        cov1 = np.cov(X0.transpose())
        cov2 = np.cov(X1.transpose())
        myCov = (cov1 + cov2) / 2  # With prior 1/2,1/2 (pooled covariance)
        myCovInv = np.linalg.inv(myCov)
        self.hCoef = np.dot(myCovInv, (mu1 - mu2).transpose())  # Fisher's linear discriminant
        self.hx = np.array([np.dot(self.hCoef, x) for x in X])

        # Optimize intercept using minimize function
        self.b = minimize(self.intercept, x0=0, method='BFGS')['fun'] # Using BFGS method, will minimize the function self.intercept

        print("FisherDA has been fitted to Data")
        return self

    def intercept(self, a):
        """
        Calculates the intercept (threshold) for classification.

        Args:
            a (float): A candidate intercept value.

        Returns:
            float: Mean misclassification rate of FisherDA + the potential candidate intercept, by minimizing this score
                   we will find the proper intercept which best separated our labeled samples.
        """
        return np.mean(self.label == (self.hx + a < 0))

    def decision_function(self, X):
        """
        Computes the decision function for the input data.

        Args:
            X (np.ndarray): The input data (features).

        Returns:
            np.ndarray: The decision function values for each sample.
        """
        self.hx = np.array([np.dot(self.hCoef, x) for x in X]) # For each value, the score between the 'hCoef' and the potential sample "X".

        # Returns the final score
        return self.hx + self.b

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Args:
            X (np.ndarray): The input data (features).

        Returns:
            np.ndarray: The predicted class labels for each sample (0 or 1).
        """
        truFalsePred = self.decision_function(X) < 0 # If under 0 will yield true , else will yield false.
        return np.array([int(x) for x in truFalsePred])

if __name__ == '__main__':
    # Generate two multivariate Gaussian samples
    samp1 = utils.gaussianGenerator([0, 5], [2, 2, 1.5], 50)
    samp2 = utils.gaussianGenerator([5, 0], [2, 2, 1.5], 50)

    # Plot the samples
    plt.figure()
    utils.plotXY(samp1, samp2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Gaussian Samples")
    plt.show()

    # Create NumPy arrays and fit the LDA classifier
    X = np.concatenate([samp1, samp2])
    y = np.concatenate([np.full(samp1.shape[0], 0), np.full(samp2.shape[0], 1)])

    lda = LDA()
    lda.fit(X, y)
    preds = lda.predict(X)

    # Plot the decision boundary
    plt.figure()
    utils.plotXY(X, y)
    utils.plot_frontiere(lda, X, label=y)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("LDA Classification and Decision Boundary")
    plt.show()

    # Compare with scikit-learn LDA
    skLda = LinearDiscriminantAnalysis()
    skLda.fit(X, y)

    print("Scikit-learn LDA Decision Function:", skLda.decision_function(X))
    print("Custom LDA Decision Function:", lda.decision_function(X))

    print("Scikit-learn LDA Coefficients:", skLda.coef_, "Custom LDA Coefficients:", lda.hCoef)
    print("Scikit-learn LDA Intercept:", skLda.intercept_, "Custom LDA Intercept:", [lda.b])

    # Generate anisotropic Gaussian samples
    size = 50
    covMat1 = [[2, 0], [0, 3]]
    mu1 = [5, 10]
    samp1 = np.random.multivariate_normal(mu1, covMat1, size)
    covMat2 = [[3, 0], [0, 2]]
    mu2 = [10, 5]
    samp2 = np.random.multivariate_normal(mu2, covMat2, size)

    # Plot the anisotropic Gaussian samples
    plt.figure()
    utils.plotXY(samp1, samp2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Anisotropic Gaussian Samples")
    plt.show()

    # Apply LDA and QDA
    X = np.concatenate([samp1, samp2])
    y = np.concatenate([np.full(samp1.shape[0], 0), np.full(samp2.shape[0], 1)])

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    predsLda = lda.predict(X)

    # Plot LDA decision boundary
    plt.figure()
    utils.plotXY(X, y)
    utils.plot_frontiere(lda, X, label=y)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Anisotropic Data and LDA")
    plt.show()

    # Apply and plot QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    # Plot QDA decision boundary
    plt.figure()
    utils.plotXY(X,y)
    utils.plot_frontiere(qda, data=X, label=y)
    plt.title("Anisotropic Data and QDA")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Example usage of FisherDA
    fda = FisherDA()
    fda.fit(X, y)
