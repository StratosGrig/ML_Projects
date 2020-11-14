# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

random.seed = 42
np.random.seed(666)

# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
titanic = pd.read_csv(r'D:\Msc\ML\Projects\Εργασία 4η\titanic.csv')

titanic.drop(['PassengerId','Sex'],axis= 1,inplace=True)

df_train = pd.DataFrame(titanic)

print(df_train)



# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
scaler = MinMaxScaler()

#df_train = scaler.fit_transform(df_train)

#print(df_train)




# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
#imputer =



# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================

model = KNeighborsClassifier()


# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
#plt.title('k-Nearest Neighbors (Weights = '<?>', Metric = '<?>', p = <?>)')
#plt.plot(f1_impute, label='with impute')
#plt.plot(f1_no_impute, label='without impute')
#plt.legend()
#plt.xlabel('Number of neighbors')
#plt.ylabel('F1')
#plt.show()

