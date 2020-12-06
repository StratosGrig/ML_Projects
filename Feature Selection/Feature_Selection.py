# Data analysis
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# enter data
data = pd.read_csv(r'D:\Msc\ML\Projects\Εργασία 7η\HTRU_2.csv')

cols = [
    "Mean of the integrated profile",
    "Standard Deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard Deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve",
    "Class",
]

data.columns = cols

# Remove the target variable
y_data = data["Class"]
X_data = data.drop("Class", axis=1)

# Split the dataset into a train test and a test set
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

# Fit the StandardScaler to the train set
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_data.columns)

# Transform the test set
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_data.columns)

# Fitting the model and predicting
model = LogisticRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

# Time to measure scores. We will compare predicted output with the real output
Accuracy_score = metrics.accuracy_score(y_test, y_predicted)
Precision_score = metrics.precision_score(y_test, y_predicted, average="macro")
Recall_score = metrics.recall_score(y_test, y_predicted, average="macro")
F1_score = metrics.f1_score(y_test, y_predicted, average="macro")

print("Accuracy score: ", Accuracy_score)
print("Precision score: ", Precision_score)
print("Recall score: ", Recall_score)
print("F1 score: ", F1_score)

probs = model.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('ROC_plot')
