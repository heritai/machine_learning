# -*- coding: utf-8 -*-
"""
svm.py

Kernel methods for classification and regression

This script explores Support Vector Machines (SVMs) for classification and
regression, performance analysis with different kernels, and investigation of the duality gap.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, load_digits
from sklearn.kernel_ridge import KernelRidge
import utils


"""
SVM Classification Experiments

This script explores the performance of linear SVM classifiers with different
values of the regularization parameter C. It generates a synthetic dataset
and plots the classification accuracy as a function of C.
"""



# Classification dataset
X, y = make_classification(n_samples=500, n_classes=2,
                           n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42) #Implemented a random seed to improve the stability

# Plot the data
plt.figure()
utils.plotXY(X, y)
plt.title("Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Define a function to calculate and return the function, and iterate over it.
def svcFit(c, X, y, classification=True):
    """Fits an SVC model and returns the score."""
    if classification:
        model = SVC(C=c, kernel='linear', gamma='auto', random_state=42) #Added the linear kernel , and the random state.
    else:
        model = SVR(C=c, gamma='scale')
    model.fit(X, y)
    return model.score(X, y)

# Fit a linear SVM classifier for different values of C
Clist = 10**np.arange(-3, 3, 1, dtype=float) #Create range for each score

# Get each score by going over the range of CLIST previously created.
scores = [svcFit(cc, X, y) for cc in Clist]

# Plot the score vs the values of C
plt.figure()
plt.plot(np.log10(Clist), scores)
plt.xlabel("log10(C)")
plt.ylabel("Classification Accuracy")
plt.title("SVM Classification Accuracy vs C")
plt.show()

# Determine the value of C leading to the best score
best_C = Clist[np.argmax(scores)] # Get the best score
print("Value of C leading to the best score:", best_C)

"""
SVM Regression Experiments

This script explores the performance of Support Vector Machines (SVMs) for
regression. It generates a synthetic dataset and plots the training points
and the prediction for the test set.
"""

# Regression dataset
n = 100
X_train = np.sort(5 * np.random.rand(n)) # The train is created with a normal distribution.
y_train = np.sin(X_train) # the correct data for the train created.
y_train[::5] += 1 * (0.5 - np.random.rand(n//5)) # Some noise to improve it's value

X_test = np.arange(0, 5, step=1e-2) # The test dataset
y_test = np.sin(X_test) # the correct data for the test created.

# Make 2d-arrays
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Plot the training data
plt.figure()
plt.scatter(X_train, y_train)
plt.title("Regression dataset")
plt.show()

# Fit a SVR model for different values of C
from sklearn.svm import SVR
scores2=[svcFit(cc,X_train,y_train,False) for cc in Clist]

# Plot the test results
plt.figure()
plt.plot(np.log10(Clist),scores2)
plt.show()


"""
SVM Kernel Regression Experiments

This script focuses on molecule activity prediction using Support Vector Regression
(SVR) and Kernel Ridge Regression (KRR) with a precomputed kernel. It loads
the kernel matrices and targets from files and evaluates the performance of
the models for different values of C and epsilon.

Make sure you have the following files in the /data directory for this code to work:
ncicancer_kernel_hf_ex0.txt
ncicancer_targets_ex0.txt
"""

# Load the data
try:
    K = np.loadtxt("data/ncicancer_kernel_hf_ex0.txt")  # Load the kernel
    y = np.loadtxt("data/ncicancer_targets_ex0.txt")[:, 0]  # Load the targets
    y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Scale the targets
except FileNotFoundError:
    print("Make sure that the data is in the proper folder")

# Split train/test sets
indices = np.random.permutation(K.shape[0]) # Getting a random permutation for the data.
train_idx, test_idx = indices[:K.shape[0]//4], indices[K.shape[0]//4:] #The train and test are given by the function
K_train = K[train_idx][:, train_idx] # K values for train
y_train = y[train_idx] # Y values for train
K_test = K[test_idx][:, train_idx] # K values for test
y_test = y[test_idx] # y values for test

print("Number of training examples:", K_train.shape[0])
print("Number of test examples:", K_test.shape[0])

#We would like to apply support vector regression.
#Plot the training and test accuracies for $C=10^{-1}$ and different values of $\epsilon$ in $[10^{-3}, 10^{-1}]$.

def svFitNpredict(c):
    model = SVR(C=c, gamma='scale',kernel="precomputed") #the SVR object
    model.fit(K_train, y_train) # we fit this object
    return model.score(K_train,y_train),model.score(K_test,y_test) # and return the corresponding values for test, and train

# We use a new set of paramaters.
Cparams=np.sort(np.concatenate((np.random.rand(3)*(10**-1-10**-3)+10**-3,[.1])))

scoresTrNTes=[svFitNpredict(cc) for cc in Cparams]

plt.figure() #Creates the plotting object
plt.plot(Cparams,scoresTrNTes) #creates a plotting between Cparams and the scores that the code is given
plt.legend(Cparams)
plt.show()

#Do the same with kernel regularized regression.
from sklearn.kernel_ridge import KernelRidge # We use the KernelRidge for the task

def kernFitNpredict(c):
    model = KernelRidge(alpha=c,kernel="precomputed")# The kernFitNpredict is fit with an alpha , and a precomputed kernel.
    model.fit(K_train, y_train) # and the method is then applied to to K_train and to y_train.
    return model.score(K_train,y_train),model.score(K_test,y_test) # and has to be within the data.

scoresTrNTes=[svFitNpredict(cc) for cc in Cparams] #Create a scoresTransTes list
plt.figure()
plt.plot(Cparams,scoresTrNTes) #blue is for train set and orange is for test set
plt.legend(Cparams) #Use for the cParams data
plt.show()

#Given the kernel matrices for training and testing, give the best possible accuracy on the test sample with a kernel machine.
#The testing data should not intervene in fitting the model.

#Compare to regularized regression with kernels.

"""
Duality Gap Analysis in SVM

This script explores the duality gap in Support Vector Machines (SVMs) and its
relationship with the optimization tolerance (tol) parameter. It loads the digits
dataset and calculates the primal and dual objective values for different tolerances.
"""

# Dataset

#Load the digits data, for proper usage, implement a function to check the double result, and get all the double results.

X, Y = load_digits(return_X_y=True) # Load the Digits to X and Y
Y[Y<5] = 1  # Class 1: digits 1, …, 4. Every value under is put as 1
Y[Y>4] = -1  # Class -1: digits 5, …, 9. Every value over is put as -1

ind = np.random.permutation(X.shape[0])[:X.shape[0]//4] # Create a random variable, of the following value of shape.
X = X[ind] #Create a shape of the dataset X
Y = Y[ind] #Create a shape of the dataset Y

print("Digits dataset:")
print("X shape:", X.shape)
print("Labels:", Y.min(), Y.max()) # the min and max, should be negative 1 and 1


#After being fitted, the object [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) has many interesting attributes:
#- `coef_` (1 x #features): is the vector defining the Riesz representation (primal coefficients);
#- `intercept_` (1): is the model intercept;
#- `support_` (#support vectors): is the array of indexes of the support vectors;
#- `dual_coef_` (1 x #support vectors): is the array of non-zero signed dual variables (that is $y_i \alpha_i$).

#Write a function, called `primal_dual(clf, X_train, y_train)`, that given a classifier object, a data matrix, and a label array, fits the classifier and returns the tuple `(primal, dual)` of primal and dual objective values.
#Check, on the dataset previously loaded, that the primal and the dual objectives are close to each other.
C=0.01

def primal_dual(clf, X_train, y_train):
    clf.fit(X_train, y_train) #The value is fit with the following properties, and we can make it more efficient.
    y_pred = clf.decision_function(X_train) #Get the decission function from the data
    loss = 1 - y_train*y_pred #The loss is 1 - the multiplication of the trained data and the prediction.
    primal = 0.5*np.sum(clf.coef_ * clf.coef_) + C*np.sum(loss[loss>0]) # The primal objective function is calculated by getting 0.5 of the weight vector, using the C, and calculating the loss in all the samples

    K = np.dot(X_train[clf.support_], X_train[clf.support_].T) # # The calculation of K is the dot product between the train and Transpose of the train.
    dual = np.sum(y_train[clf.support_] * clf.dual_coef_[0]) # The double objective function is calculated by,
    dual -= 0.5*np.dot(clf.dual_coef_[0], np.dot(K, clf.dual_coef_[0])) # The double objective function is calculated by,

    return primal, dual


clf=SVC(kernel="linear",C=0.01) #Create the SVC with the given parameters.
primal, dual=primal_dual(clf,X,Y) # Do a primal dual to a previously created function.

#How does the duality gap (difference between primal and dual objectives) behave with respect to the optimization tolerance (parameter `tol` of [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))?
#To anwser, plot the gap with respect to the tolerance in x-log-scale.

def svctol(tol, X, Y):
  """
  Fits an SVM and returns the absolute difference between primal and dual objectives.

    Args:
        tol (float): Optimization tolerance.
        X (np.ndarray): The training data (features).
        Y (np.ndarray): The target variable (labels).

    Returns:
        float: The absolute difference between primal and dual objective values.
  """
  sv=SVC(tol=tol,kernel="linear", random_state=42)
  sv.fit(X,Y)
  primal, dual=primal_dual(sv,X,Y)
  return np.abs(primal-dual)


tols=10**np.arange(-10,0,dtype=float) # Create a range between the -10 and 0 , as a float type.
diff_primal_dual=np.array([svctol(x,X,Y) for x in tols]) # for each iteration, test with X and Y, and append a numpy array

plt.figure() # creates the plot function
plt.plot(np.log10(tols),diff_primal_dual) # create a plotting with the logarithm of the tols , and the different values on the primal and dual function.

plt.xlabel("log10(Tolerance)") # X data is "Tolorence"
plt.ylabel("Primal Dual diff") # Y data is "primal double differential"
plt.title("Tolorence to Primal Double Differential") # Adding a title to improve readability
plt.show()
#plt.legend("tol")
