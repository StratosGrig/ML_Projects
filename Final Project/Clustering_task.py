import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

parser = argparse.ArgumentParser(description='Specify the clustering algorithm')
parser.add_argument("-algo", help="clustering_algorithm")

args = parser.parse_args()

clustering_algorithm = args.algo

# Import train and test datasets
KDD_Train = pd.read_csv(r'D:\Msc\ML\Projects\Final Project\NSL-KDDTrain.csv')
KDD_Test = pd.read_csv(r'D:\Msc\ML\Projects\Final Project\NSL-KDDTest.csv')

# We will use one-hot-encoding to convert categorical data to numerical data
categorical_data_cols = ['protocol_type', 'service', 'flag']

onehot_encoder = OneHotEncoder()

onehot_encoder_df = pd.DataFrame(onehot_encoder.fit_transform(KDD_Train[categorical_data_cols]).toarray())
KDD_Train = KDD_Train.join(onehot_encoder_df).drop(columns=categorical_data_cols)

onehot_encoder_df = pd.DataFrame(onehot_encoder.transform(KDD_Test[categorical_data_cols]).toarray())
KDD_Test = KDD_Test.join(onehot_encoder_df).drop(columns=categorical_data_cols)

if clustering_algorithm == "K-means" or clustering_algorithm == "gaussian" or clustering_algorithm == "birch":
    X_train = KDD_Train.values

X_cols = [column for column in KDD_Test.columns if column != 'target']

X_test = KDD_Test[X_cols].values
y_test = KDD_Test['target'].values

# We use scaler to normalize our data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if clustering_algorithm == "K-means":
    model = KMeans(n_clusters=2, random_state=44)
elif clustering_algorithm == "gaussian":
    model = GaussianMixture(n_components=2, random_state=44,covariance_type="tied")
elif clustering_algorithm == "birch":
    model = Birch(n_clusters=2, threshold=0.8)
else:
    print("Please input a valid algorithm. We support K-means, birch and gaussian at the moment! ")

# fit the model

model.fit(X_train)

if clustering_algorithm != "gaussian":
    First_cluster_length = len([label for label in model.labels_ if label == 0])
    Second_cluster_length = len([label for label in model.labels_ if label == 1])
else:
    y_labels = model.predict(X_train)
    First_cluster_length = len([label for label in y_labels if label == 0])
    Second_cluster_length = len([label for label in y_labels if label == 1])

print("First cluster's length:", First_cluster_length)
print("Second cluster's length:", Second_cluster_length)

Attack = 0
Normal = 1
if First_cluster_length > Second_cluster_length:
    Attack = 1
    Normal = 0

y_test = np.array([Attack if target == "attack" else Normal
                   for target in y_test])

y_predicted = model.predict(X_test)

# print the metrics

print("Accuracy: %.6f" % accuracy_score(y_test, y_predicted))
print("Precision: %.6f" % precision_score(y_test, y_predicted, average='macro'))
print("Recall: %.6f" % recall_score(y_test, y_predicted, average='macro'))
print("F1 score: %.6f" % f1_score(y_test, y_predicted, average='macro'))
