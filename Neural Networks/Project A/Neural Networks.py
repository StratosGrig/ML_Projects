from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = [
    MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='sgd', tol=0.0001, max_iter=100, verbose=1),
    MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='sgd', tol=0.0001, max_iter=100, verbose=1),
    MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='adam', tol=0.00001, max_iter=100, verbose=1),
    MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='relu', solver='adam', tol=0.00001, max_iter=100,
                  verbose=1),
    MLPClassifier(hidden_layer_sizes=(50), activation='tanh', solver='lbfgs', tol=0.00001, max_iter=100, verbose=1),
    MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='lbfgs', tol=0.00001, max_iter=100,
                  verbose=1),
]

MLP_results = defaultdict(dict)
i = 0

for clf in tqdm(classifiers):
    clf.fit(X_train_scaled, y_train)
    y_predicted = clf.predict(X_test_scaled)
    params = clf.get_params()
    i = i + 1
    name = i

    MLP_results[name]['Accuracy'] = accuracy_score(y_test, y_predicted)
    MLP_results[name]['Precision'] = precision_score(y_test, y_predicted, average='macro')
    MLP_results[name]['Recall'] = recall_score(y_test, y_predicted, average='macro')
    MLP_results[name]['F1 Score'] = f1_score(y_test, y_predicted, average='macro')


pd.DataFrame.from_dict(MLP_results).transpose().to_csv("MLP_Results.csv")