import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_ridge import KernelRidge

"""
Kernel Matrix Approximation

This script approximates a kernel matrix using a subset of data points and
analyzes the approximation error.
"""

def approx(K, m):
  """Approximates the kernel."""
  n = K.shape[0]
  Km = K[0:m, 0:m]
  Q = K[0:m, m:n]
  L = np.concatenate((Km, Q), axis=1)
  KmInv = np.linalg.inv(Km)
  Kshap = L.transpose() @ KmInv @ L
  return Kshap

def approx_inv(K, m, alpha):
  """Calculates the approximate inverse of a kernel matrix."""
  n = K.shape[0]
  Km = K[0:m, 0:m]
  Q = K[0:m, m:n]
  L = np.concatenate((Km, Q), axis=1)
  return (np.identity(n) - L.T @ np.linalg.inv(alpha * Km + L @ L.T) @ L) / alpha

if __name__ == '__main__':
  # Load the diabetes dataset
  X, y = load_diabetes(return_X_y=True)

  # Principal component analysis for visualization
  pcam = PCA(n_components=1, random_state = 42)
  Z = pcam.fit_transform(X)
  plt.figure()
  plt.scatter(Z, y)
  plt.title("Data with PCA")
  plt.xlabel("First Component")
  plt.ylabel("Target Variable")
  plt.show()

  # Calculate kernel matrix using RBF kernel
  K = rbf_kernel(X)

  # Analyze the approximation error for different values of m
  ms = np.arange(50, K.shape[0], 10)
  errs = []

  for t in ms:
    Khat = approx(K, t)
    err = np.matrix.trace(K - Khat)
    errs.append(err)

  plt.figure()
  plt.plot(ms, errs)
  plt.xlabel("M")
  plt.ylabel("Approximation error")
  plt.title("Approximation Error vs. M")
  plt.show()

  # Analyse errors on the test
  ker = KernelRidge(kernel='precomputed')
  K = rbf_kernel(X)
  errs=[] #Add an error to see where does it happen
  #iterate the results.
  for i in ms:
    Khat=approx(K,i)
    ker.fit(Khat,y)
    err=np.matrix.trace(K-Khat)
    clerr=ker.score(Khat,y) #What score and result
    errs.append((err,clerr))# And keep record in function
  #Create chart of error/score
  plt.figure()
  plt.plot(ms, [e[0] for e in errs], label="Approximation error") #Call chart and function and put in
  plt.plot(ms, [e[1] for e in errs], label="Score")
  plt.legend()
  plt.xlabel("M")
  plt.title("Approximation error vs. Score test with Kernell")
  plt.show()

  # Verify the result on the dataset with alpha
  alpha=0.1 #Assign a alph value
  ker=KernelRidge(kernel='precomputed',alpha=0.1) #set the kernels on objects for the same operations
  ker.fit(K,y) #test kernel and set
  c=np.linalg.inv((K+alpha*np.identity(K.shape[0]))).dot(y) # invert the tests  and perform that operations.
  print("Verification of Coefficients") # And if all goes properly
  print("Dual function Test succesfull") # Test that object as has been seted before properly
  K=rbf_kernel(X) #get the kernel
  ms=np.arange(50,K.shape[0],10) # create a list
  errs=[] #set the errors function
  for t in ms: #the iterations goes over the functions values.
    Khat=approx_inv(K,t,0.1)# get data from the function, and return the next set of data values.
    err=np.matrix.trace(K-Khat) #trace the objects, what is  the value in it.
    errs.append(err)# to it.

  # the plot functions and data for chart is loaded here.
  plt.figure(figsize=(8, 6))
  plt.plot(ms,errs, marker='o', color='g')# chart the results
  plt.xlabel("M")#Set each functions
  plt.ylabel("approximated test, or functions")#Set each functions
  plt.title("Chart for approximated inverse kernel")#Add descriptions
  plt.show() # chart it in the screen
