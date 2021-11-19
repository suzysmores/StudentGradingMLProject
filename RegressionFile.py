from os import X_OK
import pandas as pd
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style #change the style of our grid 

#Read in the data , data separated by semi colon so separate by that 
data = pd.read_csv("student-mat.csv", sep = ';')

#Grab first 5 elements and print them 
# print(data.head())

#Trim data down to what we need using the following attributes
#Pick data with integer values to them otherwise you'll have to change in databframe
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

#Array to define attributes , other is label 
#data drop remnoves the column of data that you want your model to guess 
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])


#Taking all attributes and splitting them into 4 different arrays 
#X train is section of X array
#Y train is section of Y array 
#You shouldn't train the model off the test data, otherwise it's going to see that information already
#Use best loop in order to better the model 

best = 0 
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    ####Linear Regression####
    # Creates a best fit line and tries to plot data as close as possible to that line 
    # Data directly correlating to each other means dots on x-y access can have an easy best fit line 
    # Strong correlation from data to a best fit line, use linear regression 
    # If data is dispersed, not really correlated then do not use linear regresison use other models
    # y = mx + b is used for best fit line  
    linear = linear_model.LinearRegression()
    #Find best fit line
    linear.fit(x_train,y_train)
    #Fidn how accurate the model is by seeing how close best fit line is to the data 
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy>best: 
        best = accuracy
        #store in a pickle file so we can open it and use it ; saving of model 
        with open("studentModel.pickle", "wb") as f: 
            pickle.dump(linear, f)
            

#Show that we can use pickle file to execute this model 
pickle_in = open("studentModel.pickle", "rb")
linear = pickle.load(pickle_in)




#What is slope and y intercept of this best fit line? 
print("Co: " , linear.coef_)
print("Intercept: ", linear.intercept_)

#use model to predict 
predictions = linear.predict(x_test)

#print out all predictions and show input data for that prediction 
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    #x_test[x] is what the input was 
    # y_test[x] is what the value actually was
    #predictions[x] is the predicted value 

p= 'G1'
style.use("ggplot")
#Set up a scatter plot 
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


