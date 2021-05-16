import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics, ensemble
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(11)

filename = './spambase.csv'

data = pd.read_csv(filename)
data = shuffle(data)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# set the hyperparameter for the base models
tree_depth = 3
min_sample = 5
num_tree = 50

# train and test the decision tree model
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth, min_samples_leaf=min_sample)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)
print('The prediction accuracy of the decision tree: ', np.round(metrics.accuracy_score(y_test, y_pred_dt), 3))


# Plot the decision tree
plt.figure(figsize=(8, 8))
tree.plot_tree(dtree, filled=True, class_names=['ham', 'spam'], feature_names=X.columns, fontsize=7)
plt.show()


# train and test the bagging decision tree model
bt = ensemble.BaggingClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,
                                                                                                min_samples_leaf=min_sample))
bt.fit(X_train, y_train)
y_pred_bt = bt.predict(X_test)
print('The prediction accuracy of the bagged decision tree: ', np.round(metrics.accuracy_score(y_test, y_pred_bt), 3))


# train and test the random forest
rf = ensemble.RandomForestClassifier(n_estimators=num_tree, max_depth=tree_depth, min_samples_leaf=min_sample)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('The prediction accuracy of the random forests: ', np.round(metrics.accuracy_score(y_test, y_pred_rf), 3))

# train and test the AdaBoost decisiontree
AdaBoost = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=tree_depth,
                                                                                                  min_samples_leaf=min_sample))
AdaBoost.fit(X_train, y_train)
y_pred_ada = AdaBoost.predict(X_test)
print('The prediction accuracy of the AdaBoost Decision Tree: ', np.round(metrics.accuracy_score(y_test, y_pred_ada), 3))


# plot the learn curve for the adaboost decision tree with different tree depth
staged_score = AdaBoost.staged_score(X_test, y_test)
staged_score_train = AdaBoost.staged_score(X_train, y_train)

# the score of the decision tree with depth 3
dt_score = metrics.accuracy_score(y_pred_dt, y_test)

# the score of the decision stump
dstump = tree.DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_leaf=min_sample)
dstump.fit(X_train, y_train)
y_pred_ds = dstump.predict(X_test)
dstump_score = np.round(metrics.accuracy_score(y_test, y_pred_ds), 3)

# the score of the adaboost decision tree with depth 1
AdaBoost_1 = ensemble.AdaBoostClassifier(n_estimators=num_tree, base_estimator=tree.DecisionTreeClassifier(max_depth=1,
                                                                                                 min_samples_leaf=min_sample))
AdaBoost_1.fit(X_train, y_train)
y_pred_ada_1 = AdaBoost_1.predict(X_test)
staged_score_1 = AdaBoost_1.staged_score(X_test, y_test)
staged_score_train_1 = AdaBoost_1.staged_score(X_train, y_train)

#plot the accuracy with different number of trees
plt.figure()
plt.plot(np.arange(1, num_tree+1), list(staged_score), 'r', label='AdaBoost_test(D=3)')
plt.plot(np.arange(1, num_tree+1), list(staged_score_train), 'r--', label='AdaBoost_train(D=3)')

plt.plot(np.arange(1, num_tree+1), list(staged_score_1), 'b', label='AdaBoost_test(D=1)')
plt.plot(np.arange(1, num_tree+1), list(staged_score_train_1), 'b--', label='AdaBoost_train(D=1)')

plt.plot(np.arange(1, num_tree+1), [dt_score]*num_tree, label='Decision tree')
plt.plot(np.arange(1, num_tree+1), [dstump_score]*num_tree, label='Decision stump')

plt.xlabel('num of trees')
plt.ylabel('accuracy')
plt.legend()

plt.show()