import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
import numpy as np

parser = argparse.ArgumentParser(description='Specify the clustering algorithm')
parser.add_argument("-algo", help="clustering_algorithm")

args = parser.parse_args()

cluster_algorithm = args.algo

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




