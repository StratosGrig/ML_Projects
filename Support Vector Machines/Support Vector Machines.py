import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

data = pd.read_csv(r'D:\Msc\ML\Projects\Εργασία 8η\creditcard.csv')

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

X = np.array(data.iloc[:, data.columns != 'Class'])
y = np.array(data.iloc[:, data.columns == 'Class'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train_res)
X_train = scaler.transform(X_train_res)
X_test = scaler.transform(X_test)

classifiers = [
    SVC(C=0.1, kernel="poly", gamma=0.2, degree=2, random_state=42, verbose=1),
    SVC(C=10, kernel="poly", gamma=6, degree=5, random_state=42, verbose=1),
    SVC(C=0.1, kernel="rbf", gamma=0.3, random_state=42, verbose=1),
    SVC(C=10, kernel="rbf", gamma=5, random_state=42, verbose=1),
    SVC(C=0.1, kernel="sigmoid", gamma=0.5, random_state=42, verbose=1),
    SVC(C=10, kernel="sigmoid", gamma=2, random_state=42, verbose=1),
    SVC(C=100, kernel="sigmoid", gamma=5, random_state=42, verbose=1)
]

SVM_results = defaultdict(dict)

for clf in tqdm(classifiers):
    clf.fit(X_train, y_train_res)
    y_predicted = clf.predict(X_test)
    params = clf.get_params()
    name = "For C:" + str(params['C']) + ", kernel:" + params['kernel'] + ", gamma:" + str(params['gamma'])
    if params['kernel'] == 'poly':
        name += ", degree:" + str(params['degree'])

    SVM_results[name]['Accuracy'] = accuracy_score(y_test, y_predicted)
    SVM_results[name]['Precision'] = precision_score(y_test, y_predicted, average='macro')
    SVM_results[name]['Recall'] = recall_score(y_test, y_predicted, average='macro')
    SVM_results[name]['F1 Score'] = f1_score(y_test, y_predicted, average='macro')

pd.DataFrame.from_dict(SVM_results).transpose().to_csv("SVM_Results.csv")
