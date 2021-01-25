import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

parser = argparse.ArgumentParser(description='Specify the algorithm')
parser.add_argument("-algo", help="algorithm")

args = parser.parse_args()

algorithm = args.algo

# import dataset
data = pd.read_csv(r'D:\Msc\ML\Projects\Final Project\fuel_emissions.csv')

# We will keep the data that are notNA on column fuel_cost_12000_miles
data = data[data["fuel_cost_12000_miles"].notna()]

# We will keep the columns that give us only useful info
useful_columns = [column for column in data.columns
                  if column not in [
                      "file", "description", "tax_band", "thc_nox_emissions", "particulates_emissions"
                      , "standard_12_months", "standard_6_months", "first_year_12_months", "first_year_6_months",
                      "fuel_cost_12000_miles"]]
X = data[useful_columns]
y = data['fuel_cost_12000_miles'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=44)

# We will use one-hot-encoding to convert categorical data to numerical data
categorical_data_cols = ['manufacturer', 'model', 'transmission', 'transmission_type', 'fuel_type']

onehot_encoder = OneHotEncoder(handle_unknown='ignore')

onehot_encoder_df = pd.DataFrame(onehot_encoder.fit_transform(x_train[categorical_data_cols]).toarray())
x_train = x_train.join(onehot_encoder_df).drop(columns=categorical_data_cols)

onehot_encoder_df = pd.DataFrame(onehot_encoder.transform(x_test[categorical_data_cols]).toarray())
x_test = x_test.join(onehot_encoder_df).drop(columns=categorical_data_cols)

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# We will use SimpleImputer to replace missing values with mean values
# As always, fit on train, transform on train and test.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x_train)
x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

# Create our different models based on algorithm used
# Train the model and predict. Print the performance metrics.
if algorithm == "knn":
    model = KNeighborsRegressor(n_neighbors=3, weights='distance', p=3)
elif algorithm == "linear_reg":
    model = LinearRegression(n_jobs=10)
elif algorithm == "linear_svr":
    model = LinearSVR(C=10, random_state=42, verbose=1, max_iter=10000)
elif algorithm == "random_forest":
    model = RandomForestRegressor(criterion="mae", n_estimators=10)
elif algorithm == "decision_tree":
    model = DecisionTreeRegressor(random_state=42, criterion="mae")
else:
    print("Please input a valid algorithm. We support K-means, birch and gaussian at the moment! ")

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

print("Mean absolute percentage error score: ", mean_absolute_percentage_error(y_test, y_predicted))
print("Mean absolute error score: ", mean_absolute_error(y_test, y_predicted))
print("Mean squared error score: ", mean_squared_error(y_test, y_predicted))
