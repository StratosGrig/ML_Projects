from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

# Load breastCancer data
breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# create Leave-One-Out Cross Validation procedure
cv = LeaveOneOut()
# enumerate splits
y_true, y_pred = list(), list()

for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    # fit model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    # evaluate model
    y_predicted = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(y_predicted[0])
# calculate accuracy
acc = metrics.accuracy_score(y_true, y_pred)
print('Accuracy: %.3f' % acc)

TP = 0  # True positive
FP = 0  # False positive
TN = 0  # True negative
FN = 0  # False negative

for i in range(len(y_pred)):
    if y_true[i] == y_pred[i] == 1:
        TP += 1
    if y_pred[i] == 1 and y_true[i] != y_pred[i]:
        FP += 1
    if y_true[i] == y_pred[i] == 0:
        TN += 1
    if y_pred[i] == 0 and y_true[i] != y_pred[i]:
        FN += 1

print('True positive: ', TP)
print('False positive: ', FP)
print('True negative: ', TN)
print('False negative: ', FN)
