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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import metrics, model_selection

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

titanic_drop = titanic.drop(['PassengerId','Name','Ticket','Cabin'],axis= 1,inplace=True)

#Check the maximum frequency of Embarked
print(titanic["Embarked"].mode())
#Fill in the most common category S
titanic['Embarked'].fillna('S', inplace=True)

le = LabelEncoder()
titanic['Embarked'] = le.fit_transform(titanic['Embarked'])
titanic['Sex'] = le.fit_transform(titanic['Sex'])

df1 = titanic.filter(['Pclass','SibSp','Parch','Fare','Sex','Embarked'], axis = 1)
X = df1

df1_imputed = titanic.filter(['Pclass','SibSp','Parch','Fare','Sex','Embarked','Age'], axis = 1)
X_imp = df1_imputed

df2 = titanic.filter(['Survived'], axis = 1)
y = df2

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================



scaler = MinMaxScaler()

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 33 )

scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('-----------------------------------------------------')

# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================

imputer =  KNNImputer(n_neighbors=3)

X_imp = imputer.fit_transform(X_imp)

x_train_imp, x_test_imp, y_train_imp, y_test_imp = model_selection.train_test_split(X_imp, y, random_state = 33)

scaler_imp = MinMaxScaler()

scaler_imp = scaler_imp.fit(x_train_imp)
x_train_imp = scaler_imp.transform(x_train_imp)
x_test_imp = scaler_imp.transform(x_test_imp)


# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================

model = KNeighborsClassifier(n_neighbors= 53 , p= 3 , weights = 'distance')

model.fit(x_train, y_train.values.ravel())

y_predicted = model.predict(x_test)
print("Not Imputed")
print("Accuracy score: %.6f" % metrics.accuracy_score(y_test,y_predicted))
print("Recall score: %.6f" % metrics.recall_score(y_test, y_predicted))
print("Precision score: %.6f" % metrics.precision_score(y_test, y_predicted))
print("F1 score: %.6f" % metrics.f1_score(y_test,y_predicted))

    

print('---------------------------------------------------------------')
#Imputed model

model_imp = KNeighborsClassifier(n_neighbors= 53 , p= 3, weights = 'distance')

model_imp.fit(x_train_imp, y_train_imp.values.ravel())

y_predicted_imp = model_imp.predict(x_test_imp)

print("Imputed")
print("Accuracy score: %.6f" % metrics.accuracy_score(y_test_imp,y_predicted_imp))
print("Recall score: %.6f" % metrics.recall_score(y_test_imp, y_predicted_imp))
print("Precision score: %.6f" % metrics.precision_score(y_test_imp, y_predicted_imp))
print("F1 score: %.6f" % metrics.f1_score(y_test_imp,y_predicted_imp))



n_neighbors = np.arange(1, 200)
f1_no_impute = []
f1_impute = []

#weights --> uniform | distance
#p --> 1 (Manhattan) , 2 () , 3
#n_neighbours (1-200)

for n in n_neighbors:
  
    #Not imputed
    model.set_params(n_neighbors=n , p= 2, weights = 'uniform')
    model.fit(x_train, y_train.values.ravel())
    y_predicted = model.predict(x_test)
    f1_no_impute.append(metrics.f1_score(y_test,y_predicted)) 
   
    #imputed
    model_imp.set_params(n_neighbors=n , p= 2 , weights = 'uniform')
    model_imp.fit(x_train_imp, y_train_imp.values.ravel())
    y_predicted_imp = model_imp.predict(x_test_imp)
    f1_impute.append(metrics.f1_score(y_test_imp,y_predicted_imp)) 
    

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================
plt.title('k-Nearest Neighbors (Weights = uniform , Metric = Minkowski, p = 2)')
plt.plot(f1_impute, label='with impute')
plt.plot(f1_no_impute, label='without impute')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.show()

