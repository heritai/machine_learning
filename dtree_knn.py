import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import itertools
import utils

"""
Nonparametric Methods: k-NN and Decision Trees

This script explores nonparametric machine learning methods, specifically:
    - k-Nearest Neighbors (k-NN) for classification.
    - Decision Trees for classification.
It includes experiments with synthetic datasets to demonstrate the behavior
of these algorithms under different conditions.
"""


# Dataset
X, y = make_classification(n_samples=500, n_classes=4,
                           n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# k-Nearest Neighbors

# Train a k-nearest neighbors classifier and display the classification regions. What is the default metric?
neigh = KNeighborsClassifier(n_neighbors=5) #The metric defaults to minkowski, and p =2 (Euclidean distance)
neigh.fit(X, y)

plt.figure(figsize=(8, 6))
utils.map_regions(neigh, X)
plt.scatter(X[:,0], X[:,1], c=y, s=3) #Plot all data

plt.title("k-NN Classification Regions (k=5)")
plt.show()

# Repeat this experiment while making the number of nearest neighbors vary. Display the results on several subplots with the classification score indicated in the title.
# What happens in the extreme situations where the number of nearest neighbors is either 1 or n (the size of the training set)?
neighList=[1, 5, 250, X.shape[0]]

fig = plt.figure(figsize=(12, 8)) #Set figure size
fig.subplots_adjust(hspace=.5, wspace=0, right=0.9) #Fix the parameters to make the code better

for i in range(1, len(neighList) + 1):
    nbNeigh = neighList[i-1] # Get neighbors, with the first number as 1 and the size of the training data
    neigh = KNeighborsClassifier(n_neighbors=nbNeigh) #Get the nearest neighbors
    neigh.fit(X, y) #fit the model to the KNN

    fig.add_subplot(2, 2, i) #Add subplots
    utils.map_regions(neigh, X) #Mapping regions function
    plt.scatter(X[:, 0], X[:, 1], s=3, c=y) #Scatter plot
    plt.title(f"n: {nbNeigh}, s: {neigh.score(X, y):.2f}") #Add information to the title of the scatter plot

fig.show() # Show the image

# Weight the vote of each neighbor by e^{-\gamma ||X(j)-x||^2}
# Assess the impact of the parameter  γ. Relevant values for  γ  are  10^{-3}, …, 10^4 and `n_neighbors` can be set to 10.

gammaList = 10**np.arange(-3, 5, dtype=float)

fig = plt.figure(figsize=(16, 8))# Create a new figure
fig.subplots_adjust(hspace=.5, left=0.1, right=2)# Imprvoe the size of the space for the subplots

for i in range(1, len(gammaList) + 1):
    w = lambda a : np.exp(-gammaList[i-1]*a*a) # use the Lambda, and exp functions for the parameters
    neigh = KNeighborsClassifier(n_neighbors=10, weights=w) #Create and assigne a function with a  gamma

    neigh.fit(X, y) #Use the train with the proper parameters

    fig.add_subplot(2, 4, i) #Create and assign the subplots
    utils.map_regions(neigh, X)# Get the mapping values
    plt.scatter(X[:, 0], X[:, 1], s=1, c=y)# # Use the proper X and y values, so everything is properly assigned

    plt.title(f"g: {gammaList[i-1]:.3f}, s: {neigh.score(X, y):.2f}")#Show the plot
fig.show()

# Using the train_test_split function, split the dataset into a training and a test set with ratio 0.2-0.8.
# Plot the test accuracy with respect to the number of neighbors.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

scores=[]
for i in np.arange(1, X_train.shape[0]):
  neigh = KNeighborsClassifier(n_neighbors=i)
  neigh.fit(X_train, y_train)
  scores.append(neigh.score(X_test,y_test))
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, X_train.shape[0]), scores)
plt.xlabel("Number of Neighbors")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Number of Neighbors (Single Split)") #Plot the data
plt.grid(True)
plt.show()

# Repeat the random split 20 times and plot the mean and the variance of the test accuracy.
# What can you say about the variance of this estimator?

list20Scores=[]
for j in range(0,20):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=j)

  scores=[]
  for i in np.arange(1,X_train.shape[0]):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    scores.append(neigh.score(X_test,y_test))
  list20Scores.append(scores)

fig, axes = plt.subplots(1, 2, figsize=(16, 6)) #Set and improve the graph and data.
fig.subplots_adjust(hspace=.5,left = 0.1  ,right = 2 )

axes[0].plot(np.arange(1,X_train.shape[0]),np.mean(np.array(list20Scores),axis=0)) #Plot in the first column the mean values.
axes[0].set_title("Mean Test Accuracy Over 20 Splits") # Set axis names
axes[0].set_xlabel("Number of Neighbors") # Set axis names
axes[0].set_ylabel("Mean Accuracy") # Set axis names

axes[1].plot(np.arange(1,X_train.shape[0]),np.var(np.array(list20Scores),axis=0)) #Plot in the second column the variance values.
axes[1].set_title("Variance of Test Accuracy Over 20 Splits") # Set axis names
axes[1].set_xlabel("Number of Neighbors") # Set axis names
axes[1].set_ylabel("Variance") # Set axis names

plt.show()# Plot the chart.

# Find (and print) a good value for the number of nearest neighbors using crossval_score.
# For this parameter, compare the crossvalidation score and the test accuracy.
from sklearn.model_selection import cross_val_score

def meanCrossX(nb_n):
    """
    Calculates the test accuracy and cross-validation score for a given number of neighbors.

    Args:
        nb_n (int): Number of nearest neighbors.

    Returns:
        tuple: A tuple containing the cross-validation score and the test accuracy.
    """
    myModel=KNeighborsClassifier(n_neighbors=nb_n) # Create the KNN Model, using the KNN class.
    myModel.fit(X_train,y_train)# and fit to the data using the training ones.
    testMean=myModel.score(X_test,y_test)#Then we get the test data, using the test variables.
    crossMean=np.mean(cross_val_score(myModel, X_train, y_train, cv=4))# #Then we do the same as before but with cross val score
    return crossMean,testMean# Return what has been requested.

accs=[meanCrossX(x) for x in range(1,50)]# Create a list of data, where we get the data from a certain range from 1 to 50.
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1,50), [a[0] for a in accs], label="Cross-Validation Score") #Plot the cross validation
plt.plot(np.arange(1,50), [a[1] for a in accs], label="Test Score") #The test validation
plt.xlabel("Number of Neighbors") # Set the name of the X axe
plt.ylabel("Score") # Set the name of the Y axe
plt.title("Cross-Validation and Test Scores vs. Number of Neighbors") # Improve title
plt.legend()# plot the data into the screen
plt.grid(True)
plt.show()

# Plot the confusion matrix for the best classifier obtained.
import itertools
from sklearn.metrics import confusion_matrix # Implemented the libraries to properly plot the data

def plot_confusion_matrix(y_pred, y, classes=None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title='Confusion matrix' # Set the main title of the plot
    cmap=plt.cm.Blues # Set the color of the colormap, from the matplotlib

    cm = confusion_matrix(y, y_pred) # Caclulate the different parameters for both trained and predict

    if classes is None: # Try a check to the classes in Y, so we can proceed with the plot as intended.
        classes = np.unique(y) #If there are not , set the classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #  do the same with "C", but add "normalize for proper working with the classes.
        title = 'Normalized confusion matrix' #Set a new tittle
    else:
        title = 'Unnormalized confusion matrix' #Set a new tittle

    plt.imshow(cm, interpolation='nearest', cmap=cmap) #Use an image object
    plt.title(title) # set the object as title
    plt.colorbar() # The colorbar
    tick_marks = np.arange(len(classes)) # The tike values get set.
    plt.xticks(tick_marks, classes, rotation=45) # rotate the class , into a 45 angle.
    plt.yticks(tick_marks, classes) # rotate the class , into a 45 angle.

    fmt = '.2f' if normalize else 'd' #Get and assign the formats,
    thresh = cm.max() / 2. #Assign the threhold to the max divided by 2, to have all the high parameters
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #Get different products from the iterations (Get The combinations of a range data)
        plt.text(j, i, format(cm[i, j], fmt), #The format will follow the same format.
                 horizontalalignment="center", # horizontal aligment is set
                 color="white" if cm[i, j] > thresh else "black") # Set the color of the chart.

    plt.tight_layout() # add a tight layour
    plt.ylabel('True label') #Add label
    plt.xlabel('Predicted label') #Add lable
    plt.grid(False) # Get a grid in the function

    # Plot the confusion matrix for the best classifier obtained.
    myKnn = KNeighborsClassifier(n_neighbors=22) # The 22 is the most stable, by the previous data.
    myKnn.fit(X,y) #Apply train to the data
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(myKnn.predict(X), y, classes=np.unique(y))  # Plot data
    plt.title("Confusion Matrix for Best k-NN Classifier (k=22)") # Improve title
    plt.show() #Show the final Image.
