import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as data
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
import itertools
import utils

"""
Gaussian Mixture Models and k-means Clustering

This script explores Gaussian Mixture Models (GMMs) and k-means clustering
techniques. It includes synthetic data generation and parameter estimation
experiments.

To use the code:
1. Upload and add the utils file to the directory
2. Run, to better understand its purpose and functionalities.
"""

class SoftKMeans(object):
    """Soft K-Means clustering algorithm (implementation from scratch)."""

    def __init__(self, n_components=1, n_iter=100):
        """
        Initializes the SoftKMeans clustering model.

        Args:
            n_components (int, optional): Number of clusters. Defaults to 1.
            n_iter (int, optional): Maximum number of iterations. Defaults to 100.
        """
        self.n_components = n_components
        self.n_iter = n_iter

    def fit(self, X):
        """
        Fits the SoftKMeans model to the data.

        Args:
            X (np.ndarray): The training data (features).
        """

        n_components = self.n_components # Number of components (clusters)
        weights = np.full(n_components,1/n_components)# Initialize weights

        means = X[np.random.randint(0,X.shape[0],size=n_components),:]#Initial means can be taken at random among the trainin points
        covariances = [np.identity(X.shape[1]) for k in range(n_components)]#Initialize

        # Multivariate Gaussian pdf
        def pdf(X, mean, cov):
            """Get the PDF values for each"""
            invcov = np.linalg.inv(cov + 1e-6*np.eye(cov.shape[0]))# adding 1e-6 for numerical stability
            r = np.exp(-0.5*np.diag((X-mean).dot(invcov.dot((X-mean).T)))) #This expression calculates the exponential of the matrix
            r *= np.sqrt(np.linalg.det(invcov/(2*np.pi))) #The result is then multiplied by the square root.
            return r

        def fdot(x,m):
          """Multiply each element, has to follow this format for some reason"""
          n=x.shape[0] #shape is the X,
          nn=x-m #Check if it's not a mean
          mm=np.repeat(nn,n).reshape(n,n) #Then change the shape (N,N), and reat
          return np.matmul(mm,np.diag(nn)) # return, the diagonal from nn

        # Loop
        log_likelihood = []  # Marginal log-likelihood at each iteration
        em_log_likelihood = []  # Average joint log-likelihood at each iteration

        #Use this to get the total numbers of the iterations
        for it in range(self.n_iter):
          #Get the values from a  joint_density data set
          joint_density=np.array([w*pdf(X,m,c) for w, m, c in zip(weights, means, covariances)])#shape nrowX*n_component
          joint_density=joint_density.T#Transpose to get the proper dimensions
          p=joint_density/np.sum(joint_density,axis=0)#shape nrowx*n_components #  Y|X

          weights=np.mean(p,axis=0)#shape n_components #Weigh the averages
          means=[np.sum(p[:,[j,j]]*X,axis=0)/np.sum(p[:,j]) for j in range(n_components)]#shape n_components
          #print("p shape :", means.shape)

          covariances =[] #3*(2*2)
          for j in range(n_components):#Calculate the covariance in all the dimensions.
            s=np.full((X.shape[1],X.shape[1]),0)
            for i in range(X.shape[0]):
              s=s+fdot(X[i,:],means[j])*p[i,j]

            covariances.append(s/np.sum(p))#Then apend this data and function to the list

          dd=[np.log(w)+np.log(pdf(X,m,c)) for w, m, c in zip(weights, means, covariances)]#log value from the data
          logl=np.sum([pp*dd for pp in p.T]) #Apply for a sum with the matrix of each other

          emlogl=np.mean(np.log(joint_density))#Get log from the joint densities functions.
          log_likelihood.append(logl) #The values get appended into lists
          em_log_likelihood.append(emlogl)# The EM functions are now on a list for propper usage in memory.

        self.weights_ = np.array(weights)
        self.means_ = np.array(means)
        self.covariances_ = np.array(covariances)
        self.log_likelihood_ = log_likelihood
        self.em_log_likelihood_ = em_log_likelihood

#Draw a sample of size 200 from a Gaussian mixture model
samp=[] #set the object
for i in range(200): #the iterations of the data will continue to be
    m=np.random.multinomial(1,pvals=[0.33,0.33,0.34]) #Get the values
    s=(m[0]*gaussian_sample(mu=[0,0],n=1)+ #The data for the gaussian sample will follow,
       m[1]*gaussian_sample(mu=[5,0],n=1)+ #each part will get an different value with their respective mean.
       m[2]*gaussian_sample(mu=[2,-5],n=1)) #The 3 means are going to be used in the function to train it

    samp.append(s) # append to the variable to not lose data in iteration.

samp=np.r_[samp] #Create to variable, based on range from samp
plt.figure() # create a new figure to not mix plots
plt.scatter(samp[:,0],samp[:,1], label="samples")# scatter samples

plot_cov(cov=np.identity(2),mean=[0,0])
plt.scatter(0, 0, marker='x', c='black', s=50, label="Cluster 1 Center")

plot_cov(cov=np.identity(2),mean=[5,0])
plt.scatter(5, 0, marker='x', c='black', s=50, label="Cluster 2 Center")

plot_cov(cov=np.identity(2),mean=[2,-5])
plt.scatter(2, -5, marker='x', c='black', s=50, label="Cluster 3 Center")

plt.legend() # show the data

plt.title("Sample from GMM")
plt.show()

#Fit a soft k-means with 3 components and 20 iterations on the data. Print the prior probabilities. Plot the training dataset along with the means and the covariance matrices estimated.
#Are the results consistent with the way the data has been generated?
ss=SoftKMeans(3,20)
ss.fit(samp)
print("log_likelihood : ",ss.log_likelihood_) #print data

plt.figure()
plt.scatter(samp[:,0],samp[:,1], label="Samples")

plot_cov(cov=np.identity(2),mean=[0,0])
plot_cov(cov=np.identity(2),mean=[5,0])
plot_cov(cov=np.identity(2),mean=[2,-5])

plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.show()

#Plot the two log-likelihoods versus the number of iterations.
#Is the marginal log-likelihood non-decreasing?
#Is it bounded from below by the average joint log-likelihood?
plt.plot(np.arange(ss.n_iter),ss.log_likelihood_)#Use plot
plt.plot(np.arange(ss.n_iter),ss.em_log_likelihood_)#use different
plt.legend(["l","em"]) #The legend used for the plots
plt.title("Compare EM L") #Setting the function as it should be shown.

#With the help of the  Gaussian mixture, estimate the parameters of a 3-componenents Gaussian mixture.
#Print the prior probabilities and the maximal value of log-likelihood.
#Plot the training dataset along with the means and the covariance matrices estimated.
#Are the results consistent with the your own implementation?

#Plot and use the GMM
gmm=GaussianMixture(n_components=3, random_state=42) # set the n_components
gmm.fit(samp) #Get the sample object ready

print("prior probabilities : ",gmm.weights_) #print the weight from the object
print("log-likelihood: ",gmm.score(samp)) #Print and show all the values

#Plot the test to screen.
plt.figure()
plt.scatter(samp[:,0],samp[:,1], label="samples")# plot the original samples

#Add and plot different co variance to the graph, to enrich the graph
plot_cov(cov=gmm.covariances_[0],mean=gmm.means_[0])
plot_cov(cov=gmm.covariances_[1],mean=gmm.means_[1])
plot_cov(cov=gmm.covariances_[2],mean=gmm.means_[2])
plt.legend() # show the chart into the screen.
plt.title("Comparation of Gaussian functions")# Set chart name
plt.show() # Show the final chart and plottings on screen.

#Repeat the estimation several (let us say 9) times.
#Are the results stable?
for i in range(9):#We will perform the test function over the number of
  gmm.fit(samp)  #Set with a sample.
  print("prior probabilities : ",gmm.weights_)#Print
  print("log-likelihood: ",gmm.score(samp)) #Print
  plt.figure(figsize=(8, 6))
  plt.scatter(samp[:,0],samp[:,1], label="samples") #Plot in the screen

  plot_cov(cov=gmm.covariances_[0],mean=gmm.means_[0]) # Plot to the main Chart
  plot_cov(cov=gmm.covariances_[1],mean=gmm.means_[1]) # Plot to the main Chart
  plot_cov(cov=gmm.covariances_[2],mean=gmm.means_[2])# Plot to the main Chart
  plt.legend() # show the chart on screen with the proper data
  plt.show() #Finally the plot on the screen!

#What if initial parameters are set at random (look for the suitable parameter of Gaussian mixture)?
gmm=GaussianMixture(n_components=3,init_params='random', random_state=42) #The object is created, and it is not know, that what the internal data will work, the best

gmm.fit(samp) #The  data for gaussian mixtures can then  be loaded, for fitting

print("prior probabilities : ",gmm.weights_)# Get what parameters are better for the results.
print("log-likelihood: ",gmm.score(samp)) # show to  the screen what can  happen.

plt.figure() # Set all settings for the chart to be created
plt.scatter(samp[:,0],samp[:,1], label="Data Samples") #Get the objects in an data
# Test and Plot
plot_cov(cov=gmm.covariances_[0],mean=gmm.means_[0]) # Plot the objects for the
plot_cov(cov=gmm.covariances_[1],mean=gmm.means_[1])#Plot to to
plot_cov(cov=gmm.covariances_[2],mean=gmm.means_[2])#Get a final number.
plt.legend() # show all that data into the screen.
plt.show()#Use as object

#Complete the following script in order to:
#1. sample from a Gaussian mixture;
#2. fit a  Gaussian mixture model;
#3. plot the training set, the means and the variance "contours".
def sample_gm(weights, means, covariances, size=200): # Create an proper graph , to represent functions
  """Test for different number of functions"""
  X = [] # create a chart with chart
  n_components = len(weights)# and to assign the chart to each
  for i in range(size): # The loop will go by each iterations.
    k = np.random.choice(n_components, p=weights) # Get the components to plot.
    X.append(np.random.multivariate_normal(means[k], covariances[k]))  #Create and implement all points and plot into a test.
  return np.array(X)# return each array on the data for x .

gmm = GaussianMixture(n_components=2, random_state=42) # Implemented the Random seed

# create the data to improve its functions.
for it in range(6): #for iterations bettween 1 to 6.
    plt.figure(figsize=(10, 3))
    #iterate each parameter
    for it, (weights, means, covariances) in enumerate([
        ([0.5, 0.5], [[0, 0], [5, 0]], [(1, 1, 0), (1, 1, 0)]),
        ([0.05, 0.95], [[0, 0], [5, 0]], [(1, 1, 0), (1, 1, 0)]),
        ([0.5, 0.5], [[0, 0], [0, 0]], [(10, 1, 0), (1, 10, 0)]),
        ([0.5, 0.5], [[0, 0], [5, -5]], [(10, 1, 0), (1, 10, 0)])]):

        #Get samples to the model from training functions
        samp=sample_gm(weights, means, [covariance(*c) for c in covariances], size=200)

        gmm.fit(samp)
        plt.scatter(samp[:,0],samp[:,1])

        plot_cov(cov=gmm.covariances_[0],mean=gmm.means_[0]) #Plot to to
        plot_cov(cov=gmm.covariances_[1],mean=gmm.means_[1]) #plot each
        plt.show() #Show the charts

#Load, create each component, and visualize


def plot_ellipse(ax, mean, cov, n_std=2, **kwargs):
  """Plot  K-means
  Get to use the following and check for its documentation.
  """
  vals, vecs = np.linalg.eig(cov) #Use to the lin alg . eigs, set from 0 to 0
  x, y = vecs[:, 0]
  angle = math.degrees(np.arctan2(y, x)) #Set function to return  to "degrees" after doing the math.
  width, height = 2 * n_std * np.sqrt(vals) # set as the n of standard desviation.

  ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs) #Set those parametrs.
  ax.add_artist(ellipse)
  return ellipse
""">Given a cell  A , one may define the ratio of observations of  A  of class  y \in \mathcal Y :
p_y(\mathcal A) = \frac{\left| \left\{ i \in [n] : X_i \in \mathcal A, Y_i=y \right\} \right|}{\left| \mathcal A \right|}.

Plot and create also the proper settings"""
def plot_gmm(gmm, data, ax=None): # Create function graph
    """Function called gmm"""
    labels = gmm.fit(data).predict(data) # The value, will follow GMM
    ax = ax or plt.gca() #the ax value from matplotlib

    ax.axis('equal')
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis', zorder=2) #Test the parameters
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_): #iterate the parameters
        plot_ellipse(ax, pos, covar, alpha=w * w_factor) #Plot the proper elipse

def getDataSetPlotFunction(methodName):
    """
    The functions has to be called from here,
    has to create a chart and use plot_gmm and other.
    """
    #Get the range of X value
    X=(np.arange(-5,5,1),2)
    #Call the objects so we can
    gm = GaussianMixture(n_components=2, random_state=2).fit(X) #use the gaussian functions test properly.
    plot_gmm(gm, X)

def useKmeansWithGMM():
    """Run in a single place for the data for Kmeans + GMM for easier reuse"""
    #Plot:
    #   https://www.kdnuggets.com/2023/11/gmm-vs-kmeans-clustering.html
    #Given the followin data, fit a  Gaussian mixture.
    #Display the cluster centers along with the partitioning (use the function map_regions).
    """Test the function now as it should!"""
    #Set local variables with the same parameters so it dosent break other functions.
    (weights, means, covariances) = ([0.3, 0.2, 0.5], [[-5, -1], [5, 0], [2, -5]],[(1, 5, np.pi/3), (1, 5, np.pi/3), (5, 1, np.pi/3)]) # set the number
    X = sample_gm(weights, means, [covariance(*c) for c in covariances], size=200) #Set to create the 200 and show  graph with range

    #call GMM
    gmm=GaussianMixture(n_components=3, random_state=42) # Create the  gmm objects, and show a random state.
    gmm.fit(X) #use train with a proper data

    #show the tests and data to the screen. for proper test!
    plt.figure(figsize=(8, 6))
    plt.title("K-means + GMM") # Set the title
    plt.scatter(X[:,0],X[:,1]) # show test points from both data , the  "X[i]"
    map_regions(gmm,X) #Test and implement the map region
    plt.show() # print chart on screen

def kmenasComparationTest(X,Y):
    """create to data chart and call each function so its can all be presented. """
    from sklearn.cluster import KMeans  # Create and set the KMeans to 3 components.
    kmean=KMeans(n_clusters=3, random_state=42) # We set the same parameters as GMM, with the data .

    kmean.fit(X) #Now , use X and fit

    #create a function to plot and check.
    plt.figure(figsize=(8, 6)) #Set the display in the plot
    plt.title("K-menas chart") #Set the title of the function
    plt.scatter(X[:,0],X[:,1])# Show each data point on the screen
    map_regions(kmean,X)# use from mllab the map data,  I dont have such to test it
    plt.show()#Print everything, so we can see in first  hand all details from the code

def whatIfInitialIsSetted():
  """ Test for non convex figures"""
  (weights, means, covariances) = ([0.05, 0.2, 0.75], [[-5, -1], [5, 0], [2, -5]],[(1, 5, np.pi/3), (1, 5, np.pi/3), (5, 1, np.pi/3)])
  #print(list(map(covariance,covariances)))#Test for covariance is in sync, with the function for math
  X = sample_gm(weights, means, [covariance(*c) for c in covariances], size=100)
    #create  a new for loop with the chart to get ploted
  kmean=KMeans(n_clusters=3,init='random', n_init = 'auto',random_state=42) # Implemented the Random state.
  for i in range(10): # we can range between 10 charts
    plt.figure() #Each data has to be in its respective screen for now
    kmean.fit(X)# fit the model with "X" from the data and parameters called previously.
    plt.scatter(X[:,0],X[:,1])#Get points in the "X" data from the screen chart
    map_regions(kmean,X)# and call in the same objects from all test functions.
    plt.show()# Print it at display
    #Perform  chart operations on the test objects

    #create a functions from 0 -10 and repeat 10 times

""">Analyze Gaussian mixture and k-means for non-convex clusters.
For this purpose:
1.  generate moons (then circles) with noise set to  0.1;
2.  plot the two classes with  plotXY ;
3.  display the two-cluster partitioning ( map_regions ) obtained with  Gaussian mixture and  k-means .
"""
#
"""Here, we aim at analyzing  Gaussian mixture and  k-means for non-convex clusters.
For this purpose:
generate  moons then  circles with noise set to  0.1 ;

Plot it and display the two-cluster chart in the  GMM as 1 , then 0 with 2 clusters.
"""

def analyzeTheComparation():
    """Analyze what and functions with moon test set and implement charts """
    from sklearn.datasets import make_moons # From this, we need the charts.
    X,Y=make_moons(noise=.1, random_state=0)

    gmm=GaussianMixture(n_components=2)
    kmean=KMeans(n_clusters=2)
    gmm.fit(X)
    kmean.fit(X) # Test again  the train is

    plt.figure(figsize=(8,6)) # Create the chart space
    plotXY(X,Y) # Get the points from ""X"" ,""Y"" in the  Space
    plt.title("Non-convex Data chart + plot") # Chart the function data.
    plt.show() #Show the chart image.

    plt.figure(figsize=(8, 6))#Create a new canvas

    #Perform  K means  with the data

    kmenasComparationTest(X,Y)# Test for ""Y"" for convex

#Test and call and display what can it be found
useKmeansWithGMM()
whatIfInitialIsSetted()
analyzeTheComparation()
print(" All done ")

