import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

#read the dataset from csv
filename = './spambase.csv'
df = pd.read_csv(filename)

X = df.iloc[:, :-1]  # the features
y = df.iloc[:, -1]  # the labels

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#learning svm
clf = svm.SVC(C=1, kernel='linear')
clf.fit(X_train, y_train)

#evaluation on training and testing sets
print("The accuracy on training data is %s" % clf.score(X_train, y_train))
print("The accuracy on test data is %s" % clf.score(X_test, y_test))
