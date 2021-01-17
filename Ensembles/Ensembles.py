from sklearn import datasets, metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

#
# Load the breast cancer dataset
#
breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
#
# Pipeline Estimator
#
pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
#
# Fit the model
#
pipeline.fit(X_train, y_train)

y_predicted = pipeline.predict(X_test)
#
# Pipeline Estimator
#
pipeline = make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=1))
#
# Instantiate the bagging classifier
#
bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                                 max_features=10,
                                 max_samples=100,
                                 random_state=1, n_jobs=5)
#
# Fit the bagging classifier
#
bgclassifier.fit(X_train, y_train)

y_predicted_bg = bgclassifier.predict(X_test)
#
# Print metrics

Accuracy_score_bg = metrics.accuracy_score(y_test, y_predicted_bg)
Precision_score_bg = metrics.precision_score(y_test, y_predicted_bg)
Recall_score_bg = metrics.recall_score(y_test, y_predicted_bg)
F1_score_bg = metrics.f1_score(y_test, y_predicted_bg)

print("For Bagging Classifier: ")
print("Accuracy score: %.6f" % Accuracy_score_bg)
print("Precision score: %.6f" % Precision_score_bg)
print("Recall score: %.6f" % Recall_score_bg)
print("F1 score: %.6f" % F1_score_bg)

print('----------------------------------------------------------------------------------------------')
print("For Random Forest: ")

model = DecisionTreeClassifier(criterion="gini", max_depth=3)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=44)

model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

Accuracy_score_RF = metrics.accuracy_score(y_test, y_predicted)
Precision_score_RF = metrics.precision_score(y_test, y_predicted)
Recall_score_RF = metrics.recall_score(y_test, y_predicted)
F1_score_RF = metrics.f1_score(y_test, y_predicted)

print("Accuracy score: %.6f" % Accuracy_score_RF)
print("Precision score: %.6f" % Precision_score_RF)
print("Recall score: %.6f" % Recall_score_RF)
print("F1 score: %.6f" % F1_score_RF)

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
BaggingClassifierResults = [Accuracy_score_bg, Precision_score_bg, Recall_score_bg, F1_score_bg]
RandomForestResults = [Accuracy_score_RF, Precision_score_RF, Recall_score_RF, F1_score_RF]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, BaggingClassifierResults, width, label='BaggingClassifier')
rects2 = ax.bar(x + width / 2, RandomForestResults, width, label='RandomForest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by Classifier')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
