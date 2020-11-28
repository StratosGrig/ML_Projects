from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Here we load our train and test dataset
data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

#Here we use TfiddVectorizer to fit in our data

Tfid_vectorizer = TfidfVectorizer()

X_train = Tfid_vectorizer.fit_transform(data_train.data)
X_test = Tfid_vectorizer.transform(data_test.data)

# Here we load our model and look for the best alpha parameter
model = MultinomialNB()
parameters = {'alpha': [.0001, .001, .1, .5, 1, 5]}
clf = GridSearchCV(model, parameters)

# Let's train our model.
clf.fit(X_train, data_train.target)

# Ok, now let's predict the output for the test set
y_predicted = clf.predict(X_test)
Accuracy_score = metrics.accuracy_score(data_test.target, y_predicted)
Precision_score = metrics.precision_score(data_test.target, y_predicted, average="macro")
Recall_score = metrics.recall_score(data_test.target, y_predicted, average="macro")
F1_score = metrics.f1_score(data_test.target, y_predicted, average="macro")

# Time to measure scores. We will compare predicted output with the real output
print("Accuracy score: ", Accuracy_score)
print("Recall score: ", Recall_score)
print("Precision score: ", Precision_score)
print("F1 score: ", F1_score)

# Create our heatmap

confusion_matrix = metrics.confusion_matrix(data_test.target, y_predicted)
df = pd.DataFrame(confusion_matrix, columns=data_test.target_names, index=data_test.target_names)

figure = sns.heatmap(df, annot=True, cmap='Oranges', fmt='g').get_figure()

plt.figure(figsize=(20, 12))

plt.title("Multinomial NB - Confusion matrix (a = {:.3f})[Prec = {:.5f}, Rec = {:.5f}, F1 = {:.5f}]"
          .format(clf.best_params_['alpha'], Precision_score, Recall_score, F1_score))
figure.savefig("Heatmap.png") 
