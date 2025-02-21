import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

"""
Ensemble Methods: Bagging and Random Forests

This script explores ensemble methods for regression, including:
    - Bagging (Bootstrap Aggregating) with Decision Tree regressors.
    - Random Forests.

It uses the diabetes dataset to compare the performance of these methods.
"""

# the distribution is binomial
#Import the libraries
from scipy.special import comb # import the comb function

xdam=np.arange(0,10) # create the list of values.
p=.7 # The  p(x)
binomDens=comb(9,xdam)*(p**xdam)*((1-p)**(9-xdam)) # create the  binomDens function

#Plot the binomial distribution
plt.figure(figsize=(8, 6))# create a chart that has x and Y data
plt.vlines(xdam, 0, binomDens, colors='red') # Get the range of x in a form of lines, set them as red.
plt.xticks(np.arange(0, 10, step=1)) # Set the values to be presented, every number.
plt.xlabel("Number of Successes") #Set the axis
plt.ylabel("Probability") #Set the axis
plt.title("Binomial Distribution (T=9, p=0.7)")#Set the title of the graphic, to make it easier to read and understand
plt.grid() #Improve readibilty
plt.show() #Show

#Test the function from Acccal and generate a graph

# code for AccCal function


y=np.random.choice([-1,1],100) # set the data to 100 samples with choices between -1 and 1
def accCal(tt,p):
    """calculates the accuraccy and perform and tests the lambda on each data point. """
    f= lambda x: [x,-x][np.random.binomial(1,1-p)] # set the lamdba function to be tested and proper return
    v=[] # create an empty array to handle values
    for i in range(0,tt): # for each T range
        yHatG=np.array([f(x) for x in y]) #apply the testing into an Y array
        v.append(yHatG) # and save its value into an array.
    GThat=np.sign(np.mean(np.array(v),axis=0)) #then , sign function has to be run,
    return np.mean(y==GThat) # the return is what to be shown.

#Iterate each parameter p, to test and plot.
plt.figure(figsize=(8, 6)) # Set canvas size
for p in np.arange(0.55,1,0.1 ): # set a range from 0.55, to 1 with step 0.1

    #Set the accuraccy values for each y
    accs=[accCal(x,p) for x in np.arange(1,50,2 ) ]#Test a with 20 interations, the result for the function
    plt.plot(np.arange(1,50,2 ),accs)#Test all functions from 1 to 49, with a stpe size of 2.

plt.xlabel("Number of Estimators") #Add name
plt.ylabel("Accuracy")# Add name
plt.grid(True) # Imrpove visualization
plt.title("Baggin Regressor bootstrap vs .25 sampling") # The object will use the data from the function
plt.legend(np.arange(0.55,1.05,0.1 ))# Add function
plt.show()# Plot to the screen

"""
Apply Bagging and Random Forest Algorithms to regression problems
"""
#Load data to perform ML for regressor functions

#Load the data
dbt = load_diabetes()#The object dataset get linked

#The train,test functions and the X and y datasets are assigned.
X_train , X_test , y_train , y_test = train_test_split(dbt.data , dbt.target , test_size = 0.2, random_state=42)
#We set the scaler as standard scaler function to use with sklearn, to transform for good training data.
stdScaler = StandardScaler() #Standardize
#We use both data and transforms in the Xtrain and Xtest objects
X_train_scaled = stdScaler.fit_transform(X_train)
X_test_scaled = stdScaler.transform(X_test)

#The function, bagging_scores is set to get and plot to screen
def bagging_scores(xtrain, xtest, ytrain, ytest, model="Bagging", iters=100,
                    bootstrap=True, randstate = 42, min_samples_leaf=5):# The function has the test and train objects, and parameters for bagging.
    '''Function to calculate the list of accuracy scores for any boosting algorithm'''
    #Set the score variable as a list and to empty, set the iters with its test iterations, then set the base learners
    #This is part of bagging algorithm creation
    scores, est = [] , iters #Initialize the scores, test and iteration values
    n = xtrain.shape[0] # Set object that carries the info
    model_name = str(model) #Get the name of the model
    #Now we can use either the bagging regressor of a random forest regressor

    if model_name=="RandomForest": #We check if the model type is "Random Forrest"
        model = RandomForestRegressor(n_estimators = iters, max_features = 0.3, # if we are not random forrest, what this means? that all random forest are good
                                    random_state = randstate, oob_score = True, min_samples_leaf=min_samples_leaf) #Add the random_state to improve the stability
    else:# we set as  "Bagging" , a default case
        model = BaggingRegressor(DecisionTreeRegressor(), # So we can also use in the following way
                                n_estimators = iters, max_samples = 1.0 if bootstrap else 0.2, # The boostrap also has its data
                                random_state = randstate, oob_score = True) # Set the random_state for stability and reproductivity
    #The data are fit with traint and test parameters
    model.fit(xtrain, ytrain) #The traint data has to be tested
    for i in range(0, len(model.estimators_)): #for each data point, calculate the score so we dont set new variables again and again
        model = BaggingRegressor(DecisionTreeRegressor(),
                        n_estimators = i+1, max_samples = 1.0 if bootstrap else 0.2) # The boosting is created the number of iterations for the loop.
        model.fit(xtrain, ytrain) # all that objects must be fit. # Do not use the loop.
        scores.append(model.score(xtest, ytest)) #all objects are scored, and appended to score results.

    return scores # show

#Code for the functions used:
bootsTrue=bagging_scores(X_train_scaled, X_test_scaled, y_train , y_test,"Bagging") #Bootsrap
bootsFals=bagging_scores(X_train_scaled, X_test_scaled, y_train , y_test,"Bagging",bootstrap=False)#Sampling

Randforr=bagging_scores(X_train_scaled, X_test_scaled, y_train , y_test,iters=12,model="RandomForest") #Now using Radnom forrest.
#Create plottings so we can have better information from the functions, add plot points, descriptions, etc.
x = range(1,len(bootsTrue)+1) #Create the range, it has to go from 1 to the length.

plt.plot(x,bootsTrue, label="boostrap") #Add label
plt.plot(x,bootsFals,label = "Sampling")# add label
plt.plot(x,Randforr,label = "RandForrest") # Add the randforr
plt.xlabel("n_estimators")#Add the name to the x
plt.ylabel("accuracy score")# Add the name to the Y
plt.legend()# Add legend and show, for transparent plots.
plt.title("Comparision of bagging techniques and models by its score.")# Test is now tested proper, by its type and score.
plt.show() #SHOW chart! The following is a sample for the Dailty Gap

#Test the code
def BagTress(X,y,X_tr,Y_tr):
  #create the data
  bgt = BaggingTree() # We are creating a baggin object, so we can use it
  bgt.fit(X,y) #The object will get fit
  predict=bgt.predict(X_tr) # all train data can be predict
  errs=bgt.error(X,y) # and we can compare against errors
  return print(errs) #Print the data
#BagTress(X,y,X_train,y_train) ##Error that variable X doesn exists, is not linked to the class
