import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

"""
Model Selection Experiments

This script performs model selection using GridSearchCV to determine optimal
parameters for different machine learning models. It compares the
performance of Random Forest, Gradient Boosting, and Logistic Regression
on the digits dataset using cross-validation and statistical testing.
"""

#Load the digits data, for proper usage, implement a function to check the double result, and get all the double results.

#Split data
#Load the functions or data objects to run and fit the data, you need the followings

#load digits so it performs over the best data test to it.
digits = load_digits() # Load all digit related parameters to to X and Y value

X, Y = digits["data"], digits["target"] #Split X,Y variables.

#Create all test , test and scalers to the dataset for proper function working.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Split the function X,Y into traint,test and set the rest test to 0.2

scaler = StandardScaler() # StandardScaler to improve the data handling
X_train = scaler.fit_transform(X_train) # the fitting and the transform parameters are called, and the data is saved for all X
X_test = scaler.transform(X_test) # The test data , is fitted and transformed with  the standard scaler.

#Define the list of all parameters so it can be easy to call and test.
tree_params = {"max_depth": [5, 10, 15],# The treepramaters are to be checked
              "n_estimators": [10, 100]} # and the n_estimators to this numbers.
gb_params = tree_params.copy() # the other values has to take the copy from each others

gb_params.update({"learning_rate": np.logspace(-2, 0, num=3), #Get the learnign rate
                 "max_features": [0.1]})# and the max value to one

lr_params = {"C": np.logspace(-3, 3, num=7)}  # Parameters for logistic regression #Set from -3 to 3 the values.

#Perform the test and implement on the function

#set the functions so all this is perform in the grid test set.
rfgs=GridSearchCV(RandomForestClassifier(random_state=42), tree_params,cv=3, n_jobs= -1, verbose= 0) #The parameters of the tree are, to improve the score.
gbgs=GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params,cv=3, n_jobs= -1, verbose= 0) #The parameters of the other one, follow a number.
lrgs=GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=3, n_jobs= -1, verbose= 0) # and a logisitic object to test to the following test functions.

#Test, and fit the data for this functions.

rfgs.fit(X_train,y_train) # The randoms functions gets the data.
gbgs.fit(X_train,y_train)# The gradients is also has to be run
lrgs.fit(X_train,y_train)# the Logisitic function has to perform too.
 #Print the values of the best function that has been found

#Print to screen functions and results to see them.
print("best Random Forest score: ",rfgs.best_estimator_.score(X_test,y_test)) #The text, is tested if is a random forest score.
print("best GradiBoosting score: ",gbgs.best_estimator_.score(X_test,y_test)) #The Gradient values.
print("best LogisticRegrs score: ",lrgs.best_estimator_.score(X_test,y_test)) #Test if the best number to run Logistictregresion.

#Then, for a more scientific insight we want also to test if the results is of proper use, and use what is called with function from "wilocoxon"
scoreGB,scoreLr,scocreRandFr=[],[],[] # Create empty dataset before looping.

#The functions will run N =20 times, so they can test each individual tests and results
for k in range(20):
    #The process of the test, from testing the grid, to fitting, and improving the best way
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=k) #Add data, test

    scaler = StandardScaler() # call the scaler for this to work.
    X_train = scaler.fit_transform(X_train) #the objects, X train has to be with .
    X_test = scaler.transform(X_test) # Same as Before, but not transform only.

    #Again, the data has to re fit
    rfgs=GridSearchCV(RandomForestClassifier(random_state=k), tree_params,cv=3, n_jobs= -1, verbose= 0)
    gbgs=GridSearchCV(GradientBoostingClassifier(random_state=k), gb_params,cv=3, n_jobs= -1, verbose= 0)
    lrgs=GridSearchCV(LogisticRegression(random_state=k), lr_params, cv=3, n_jobs= -1, verbose= 0) #Again objects has to be assigned
    # again objects is fitted
    rfgs.fit(X_train,y_train) #After being fitted with tranings, and test and seed
    gbgs.fit(X_train,y_train)#After being fitted with tranings, and test and seed
    lrgs.fit(X_train,y_train)#After being fitted with tranings, and test and seed
    #The results from those operations are appended, this is more organized to have data.
    scocreRandFr.append(rfgs.best_estimator_.score(X_test,y_test)) # The tests are all appened, so its organized
    scoreGB.append(gbgs.best_estimator_.score(X_test,y_test))# The other functions are created, so the data is appended to them, to check, test each result later
    scoreLr.append(lrgs.best_estimator_.score(X_test,y_test))# The loop goes again 20 times
#
    #print to screen the functions with average data, or the standard deviation,
print("Random Forest mean: ",np.mean(scocreRandFr),"sd",np.std(scocreRandFr))# The test result is printed to the user for a better experience.
print("GradientBoosting mean: ",np.mean(scoreGB),"sd",np.std(scoreGB)) #Get the gradiatn value
print("LogisticRegrs mean: ",np.mean(scoreLr),"sd",np.std(scoreLr)) #Get the scores for the logistics


w, p_value = wilcoxon(scocreRandFr,scoreGB)# run to  gradient compare betweens what has the random numbers and the result
print("Wilcoxon tested random forrest, and graddient boost: ",p_value) #Show also on the screen the P value.
