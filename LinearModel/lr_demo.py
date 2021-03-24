# Demo code for homework 2
# You may need to install the following libraries in advance
# In order to read excel, you may add the package "xlrd" by "pip install xlrd"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

# read the data from bankloan.xls
filename = 'bankloan.xls'
df = pd.read_excel(filename)
X = df.iloc[:, :-1]  # the features
y = df.iloc[:, -1]  # the labels

# split the dataset into test and training sets
# make sure that you vary the "random_state" when you need to randomly split the dataset repeatly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# learn the logistic regression model
LR = lr()  # create the logistic regression model
LR.fit(X_train, y_train)  # learn the parameters

# prediction and evaluation
test_accuracy = LR.score(X_test, y_test)
print('The accuracy on test set is %0.2f' %test_accuracy)

# predict_proba gives you the predicted probability of default
probs_y = LR.predict_proba(X_test)

# precision_recall_curve gives you the prevision, recall with different thresholds
# you need to import precision_recall_curve from sklearn.metrics before calling this function
precision, recall, thresholds = precision_recall_curve(y_test, probs_y[:, 1])
# plot the precision curve with thresholds
plt.plot(thresholds, recall[: -1], "b", label="Precision")
plt.xlabel("Thresholds")
plt.title("Recall")
plt.legend()
plt.show()
