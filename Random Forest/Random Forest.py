# Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Load the data
dataSet = pd.read_csv("heart.csv")

# Normalize the data
dataSet['age'] = np.log(dataSet['age'])
dataSet['trtbps'] = np.log(dataSet['trtbps'])
dataSet['chol'] = np.log(dataSet['chol'])
dataSet['thalachh'] = np.log(dataSet['thalachh'])

# Encode categorical variable
labelEncoder = LabelEncoder()
dataSet['sex'] = labelEncoder.fit_transform(dataSet['sex'])
dataSet['cp'] = labelEncoder.fit_transform(dataSet['cp'])
dataSet['fbs'] = labelEncoder.fit_transform(dataSet['fbs'])
dataSet['restecg'] = labelEncoder.fit_transform(dataSet['restecg'])
dataSet['exng'] = labelEncoder.fit_transform(dataSet['exng'])
dataSet['slp'] = labelEncoder.fit_transform(dataSet['slp'])
dataSet['caa'] = labelEncoder.fit_transform(dataSet['caa'])
dataSet['thall'] = labelEncoder.fit_transform(dataSet['thall'])

X = dataSet.drop(['output'], axis = 1)
y = dataSet['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

RF = RandomForestClassifier()
RF.fit(X, y)

# predictions
print("Accucary Train: " + str(accuracy_score(y_train, RF.predict(X_train))))
print("Accucary Test: " + str(accuracy_score(y_test, RF.predict(X_test))))

# Using the confusion
confusion_matrix = metrics.confusion_matrix(y_test, RF.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
