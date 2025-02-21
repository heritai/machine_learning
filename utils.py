import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

"""
Utility Functions for Machine Learning Experiments

This module contains reusable functions for generating data and visualizing results,
used throughout the machine learning scripts.

Functions:
    - gaussianGenerator: Generates multivariate Gaussian samples.
    - gaussian_sample: Generates Gaussian samples with specified parameters.
    - plotXY: Plots data points from two classes on a scatter plot.
    - plot_frontiere: Plots the decision boundary of a binary classifier.
    - map_regions: Maps the prediction regions of a classifier.
"""

def gaussianGenerator(mu, varCovar, size):
    """
    Generates a multivariate Gaussian sample.

    Args:
        mu (list): Mean of the Gaussian distribution (list of two floats).
        varCovar (list): Variances and covariance (sigma1, sigma2, theta) as a list of three floats.
        size (int): Sample size (number of data points to generate).

    Returns:
        np.ndarray: A NumPy array of shape (size, 2) containing the generated sample.
    """
    covMat = [[varCovar[0], varCovar[2]], [varCovar[2], varCovar[1]]]
    return np.random.multivariate_normal(mu, covMat, size)

def gaussian_sample(mu=[0,0], sigma1=1, sigma2=1, theta=0, n=50):
    """
    Generates a gaussian sample based on it's parameters

    Args:
        mu (list, optional): Mean of the Gaussian distribution (list of two floats). Defaults to [0,0].
        sigma1 (float, optional): Variance in the x-direction. Defaults to 1.
        sigma2 (float, optional): Variance in the y-direction. Defaults to 1.
        theta (float, optional): Rotation angle in radians. Defaults to 0.
        n (int, optional): Sample size. Defaults to 50.

    Returns:
        np.ndarray: A NumPy array of shape (n, 2) containing the generated sample.
    """
    s = np.sqrt
    cov = [[sigma1, 0], [0, sigma2]]
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

    return np.random.multivariate_normal(mu, rotation @ cov @ np.transpose(rotation), size=n)

def plotXY(X, Y, legend=True):
    """
    Scatter points from two classes.

    Input:
        X and Y may be:
        - two numpy arrays with two columns; each array is the data matrix for a class
          (works only for two classes).
        - a numpy array with two columns (the data matrix) and the vector of labels
          (works for many classes).

    Args:
        X (np.ndarray): Input data (features).
        Y (np.ndarray): Target variable (labels).
        legend (bool, optional): Whether to display the legend. Defaults to True.
    """
    if Y.ndim > 1:
        X1 = X
        X2 = Y
        XX = np.concatenate((X, Y), axis=0)
        YY = np.concatenate((np.ones(X.shape[0]), -np.ones(Y.shape[0])))
    else:
        XX = X
        YY = Y
    for icl, cl in enumerate(np.unique(YY)):
        plt.scatter(XX[YY==cl, 0], XX[YY==cl, 1], label='Class {0:d}'.format(icl+1))
    plt.axis('equal')
    if legend:
        plt.legend()


def plot_frontiere(clf, data=None, num=500, label=None):
    """
    Plot the frontiere f(x)=0 of the classifier clf within the same range as the one
    of the data.

    Input:
        clf: binary classifier with a method decision_function
        data: input data (X)
        num: discretization parameter
    """
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num))
    z = clf.decision_function(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    cs = plt.contour(x, y, z, [0], colors='r')
    if label is not None:
        cs.levels = [label]
        plt.gca().clabel(cs)
    return cs


def map_regions(clf, data=None, num=500):
    """
    Map the regions f(x)=1…K of the classifier clf within the same range as the one
    of the data.

    Input:
        clf: classifier with a method predict
        data: input data (X)
        num: discretization parameter
    """
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    x, y = np.meshgrid(np.linspace(xmin, xmax, num), np.linspace(ymin, ymax, num))
    z = clf.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
    plt.imshow(z, origin='lower', interpolation="nearest",
               extent=[xmin, xmax, ymin, ymax], cmap=cm.coolwarm,
              alpha=0.3)
