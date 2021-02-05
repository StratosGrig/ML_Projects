# =============================================================================
# MACHINE LEARNING
# EXAMS - FEBRUARY 2021
# PROGRAMMING PROJECT
# Complete the missing code by implementing the necessary commands.
# =============================================================================

# Libraries to use
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

random_seed = 74

breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=74)

classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski')
classifier.fit(X_train, y_train)

y_predicted = classifier.predict(X_test)

y_predicted_train = classifier.predict(X_train)

Accuracy_score = accuracy_score(y_test, y_predicted)
Precision_score = precision_score(y_test, y_predicted)
Recall_score = recall_score(y_test, y_predicted)
F1_score = f1_score(y_test, y_predicted)

print("Accuracy score: %.6f" % Accuracy_score)
print("Precision score: %.6f" % Precision_score)
print("Recall score: %.6f" % Recall_score)
print("F1 score: %.6f" % F1_score)

Accuracy_score_train = accuracy_score(y_train, y_predicted_train)
Precision_score_train = precision_score(y_train, y_predicted_train)
Recall_score_train = recall_score(y_train, y_predicted_train)
F1_score_train = f1_score(y_train, y_predicted_train)

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
Test_res = [Accuracy_score, Precision_score, Recall_score, F1_score]
Train_res = [Accuracy_score_train, Precision_score_train, Recall_score_train, F1_score_train]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, Test_res, width, label='Test')
rects2 = ax.bar(x + width / 2, Train_res, width, label='Train')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by Test/Train')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
