#import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

#load dataset
X, y = datasets.load_iris(return_X_y=True)

#print dataset
print("dataset:\n",X,"\n---\n",y)

#split the iris data into a train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#print training and testing datasets
print("training dataset:\n",X_train,"\n---\n",y_train)
print("\ntesting dataset:\n",X_test,"\n---\n",y_test)

#define logistic regression classifier
clf =  LogisticRegression(C = 1, penalty='l2', solver='newton-cg', multi_class='multinomial')

#fit the model to training data
clf.fit(X_train, y_train)

#make predictions on testing data
predictions = clf.predict(X_test)

#print predictions against actual labels
print("predictions: ",predictions)
print("actual labels: ",y_test)

#calculate accuracy
corr = predictions == y_test
acc = (sum(corr)/len(predictions)*100)

#print accuracy
print("accuracy = "+str(acc)+"%")

