import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
#read the CSV
#
data = pd.read_csv("data.csv")
X = np.array(data.drop(["Class"],axis=1))
y = np.array(data["Class"])

#Name columns
#data.columns = ["Industrial Risk","Industrial Risk"," Financial Flexibility","Credibility","Competitiveness","Operating Risk","Class"]

#store file as data.csv

data.to_csv("data.csv",index=None,header=True)
#

[X_train,X_test,y_train , y_test] = train_test_split(X, y, test_size=0.1, random_state=0)

#SVC Classifier
Classifier = SVC(kernel='_linear_')
model = Classifier.fit(X_train,y_train)
accu = model.score(X_train,y_train)

print("Accuracy of SVC: " , accu)

#Logistic Regression
Classifier = LogisticRegression()
model = Classifier.fit(X_train,y_train)
accu = model.score(X_train,y_train)

print("Accuracy of Logistic Regression: " , accu)