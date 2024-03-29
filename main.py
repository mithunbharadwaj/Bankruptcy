import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
#read the CSV
#

#read the CSV

data = pd.read_csv("data.csv")
X = np.array(data.drop(["Class"],axis=1))
y = np.array(data["Class"])

def rep(x):
    if x == 'NB':
        return 0
    else:
        return 1

def rep_column(x):
    if x == 'P':
        return float(1)
    elif x == 'A':
        return float(2)
    else:
        return float(3)


data["Class"] = data["Class"].map(rep)
data["Industrial Risk"] = data["Industrial Risk"].map(rep_column)
data["Industrial Risk.1"] = data["Industrial Risk.1"].map(rep_column)
data[" Financial Flexibility"] = data[" Financial Flexibility"].map(rep_column)
data["Credibility"] = data["Credibility"].map(rep_column)
data["Competitiveness"] = data["Competitiveness"].map(rep_column)
data["Operating Risk"] = data["Operating Risk"].map(rep_column)

data.to_csv("data_new.csv",index=None,header=True)



#Name columns
#data.columns = ["Industrial Risk","Industrial Risk"," Financial Flexibility","Credibility","Competitiveness","Operating Risk","Class"]

#store file as data.csv

data.to_csv("data_new.csv",index=None,header=True)

data = pd.read_csv("data_new.csv")

X = np.array(data.drop(["Class"],axis=1))

y = np.array(data["Class"])

#Perform data split 90% tranin and 10% test data
[X_train,X_test,y_train , y_test] = train_test_split(X, y, test_size=0.1, random_state=0)

X,y = datasets.load_iris(return_X_y=True)


# Build Logistic Regression
Classifier = LogisticRegression(solver='liblinear')
model = Classifier.fit(X_train,y_train)
accu = model.score(X_train,y_train)


#Perform data split 70% data split and 30% test
[X_train,X_test,y_train , y_test] = train_test_split(X, y, test_size=0.2, random_state=0)

# Build SVC Classifier
Classifier = SVC(kernel='linear')
model = Classifier.fit(X_train,y_train)
accu = model.score(X_train,y_train)

print()
print("Accuracy of SVC: " , accu)
score_cv = cross_val_score(model,X,y , cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (score_cv.mean(), score_cv.std()))
print("Scores of CV: " ,score_cv)
print("Traning accuracy")
print(accuracy_score(y_train, model.predict(X_train)))
print("testing accuracy")
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print()

# Build Logistic Regression
Classifier = LogisticRegression(solver='liblinear')
model = Classifier.fit(X_train,y_train)
accu = model.score(X_train,y_train)

print("Accuracy of Logistic Regression: " , accu)
score_cv = cross_val_score(model,X,y , cv=5)
print("Scores of CV: " ,score_cv)
print("Traning accuracy")
print(accuracy_score(y_train, model.predict(X_train)))
print("testing accuracy")
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print()


# Build Neural Network
Classifier = MLPClassifier(max_iter=1000)
model = Classifier.fit(X_train,y_train)
print("%0.2f accuracy with a standard deviation of %0.2f" % (score_cv.mean(), score_cv.std()))
accu = model.score(X_train,y_train)

print()
print("Accuracy of Neural Network: ",accu)
score_cv = cross_val_score(model,X,y , cv=5)
print("Scores of CV: " ,score_cv)
print("Traning accuracy")
print(accuracy_score(y_train, model.predict(X_train)))
print("testing accuracy")
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print("%0.2f accuracy with a standard deviation of %0.2f" % (score_cv.mean(), score_cv.std()))
print()
print()
# Build Decision tree
Classifier = tree.DecisionTreeClassifier()
model = Classifier.fit(X_train,y_train)
accu = model.score(X_train,y_train)

print()
print("Accuracy of Decision tree: ",accu)
score_cv = cross_val_score(model,X,y , cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (score_cv.mean(), score_cv.std()))
print("Scores of CV: " ,score_cv)
print("Traning accuracy")
print(accuracy_score(y_train, model.predict(X_train)))
print("testing accuracy")
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print()
