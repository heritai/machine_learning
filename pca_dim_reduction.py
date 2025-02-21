import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS

"""
Dimensionality Reduction using Graph-Based and Classical Multidimensional Scaling

This script implements a variation of classical Multidimensional Scaling (MDS)
using a graph-based distance metric.
"""

class GraphMDS():
    """Graph-based MultiDimensional Scaling (GMDS) implementation"""
    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors #The number of neighbors
        self.n_components = n_components #The number of componants

    def fit_transform(self, X):

      K = kneighbors_graph(X,n_neighbors=self.n_neighbors,mode='distance',n_jobs=-1) #The data, to be fit to KNeigh data
      D = graph_shortest_path(K,directed=False)# and be tested

      D2 = D**2# The math of each, is powerd by the power of 2

      n=X.shape[0]#Get from each object, the number of each shap to test
      I=np.identity(n)# create a new one
      oneM=np.full((n,n),1)# and assign data.

      H=I-(1/n)*oneM*oneM.T# The data is now setted and can be transofrmed with this linear equations.

      K=(-1/2)*H@D2@H# all will chart with this values

      kpc=KernelPCA(n_components=self.n_components,kernel='precomputed') # the numbers is precomputed. and has to be tested

      Z=kpc.fit_transform(K) # Then is tested to Z

      return Z #  show the test

if __name__ == '__main__':

  # Load and visualize the Swiss Roll dataset
  X, t = make_swiss_roll(n_samples=1000, noise=0., random_state=42)
  colormap = dict(c=t, cmap='jet')

  # Initialize 3D plot
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  ax.view_init(10, -70)

  ax.scatter(X[:, 0], X[:, 1], X[:, 2], **colormap)
  plt.title("Swiss Roll Dataset Chart")
  plt.show()
  #Now that can  chart a new with other functions like plot etc , and call with another data set.
  #Test the functions

  #Use other functions:
  from sklearn.decomposition import PCA # test and implement data from PCA function
  from sklearn.manifold import MDS # set data for MDS with the data.

  #Chart settings from the lin alg and tests.
  pcaa=PCA(n_components=2)# Set the value from the data and call test
  mds=MDS(n_components=2,max_iter=50, random_state = 42)#Get and test each parameter

  #Use what we have to chart new things:
  Zpca=pcaa.fit_transform(X)  #Use chart for lin alg for training charts (X)
  Zmds=mds.fit_transform(X)#Get chart and settings from "" MDS ""

  plt.figure(figsize=(8, 6))
  #Apply the code in chart
  #Call the set, set with the ""t"" value of the chart.
  plt.title("Apply the code in chart functions and to data, Set object for PCA  to  chart test code") #Set code for PCA

  # chart the data in a scatter plot
  plt.scatter(Zpca[:, 0], Zpca[:, 1], **colormap)
  # Test that all test parameters works
  plt.xlabel("Feature 1") # The object has to be tested with the all proper codes.
  plt.ylabel("Feature 2")# Set the values so the code works
  plt.show()#Show all those tests.

  colormap = dict(c=t, cmap='jet') # chart again.  #Set data objects
  plt.figure(figsize=(8, 6))# Create new image

  #Show new charts, for test, get proper the image. chart data and get what can get the M
  plt.scatter(Zmds[:, 0], Zmds[:, 1], **colormap)#Add colormaps
  plt.title("Show functions used  Test test code")# Show function to set what objects test has has properly added .
  plt.xlabel("Feature 1")#Add a object function to test a image,
  plt.ylabel("Feature 2")#Add a object to see from the 2 dimensionals charts.
  plt.show()# and finally, plot the new graph for MD

  #Implement the chart objects
  #create  a new list for code function by the 3D image, in 2D screens.
  gg=GraphMDS(10) #Test for chart and for each of the proper codes.
  Z=gg.fit_transform(X)#Set functions in the data.

  colormap = dict(c=t, cmap='jet') # chart the color data.
  #Add chart images functions
  plt.figure(figsize=(8, 6))# Test with the proper data chart
  plt.scatter(Z[:, 0], Z[:, 1], **colormap)## and implement chart data and functions

  plt.title("Test with another data, to look charts, or test codes")
  plt.xlabel("Feature 1")#Set axis to get this values and make to codes chart right here
  plt.ylabel("Feature 2")# chart the values.
  plt.show() # test all parameters are

  """
  All the parts of the code has been tested and improved by the
  explanations with tests and codes . It also includes,
  to test lin alg functions which are for test to make sure the result will be.

  """
