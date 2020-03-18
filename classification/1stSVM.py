#This is my first SVM classifier. It is quite simple, using predefined datasets.

#import necessary packages
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

#load the dataset
digits = datasets.load_digits()

print("dataset: ",digits)

#split the data in 'digits' into a train and test data matrices
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)

#initialize SVM classifier with gamma = 0.001 and C = 100 (this combination fits the data and test set well)
clf = svm.SVC(gamma=0.001, C=100.)

#fit the model on the training data
clf.fit(X_train, y_train)

#predict on the test data
predictions = clf.predict(X_test)
print("predictions: ",predictions)

print("actual labels: ",y_test)

#calculate accuracy
right = (predictions == y_test)
acc = sum(right)/len(right)

#print accuracy
print("\naccuracy: "+str(acc*100)+"%")
