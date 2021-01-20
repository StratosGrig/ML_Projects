from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation

#
# Load the breast cancer dataset
#
breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# For KMeans clustering algorithm
for n in range(2, 8):
    model = KMeans(n_clusters=n, random_state=10)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    labels = model.predict(X)

    print("Silhouette score with K-Means algorithm and %d_clusters is: " % n,
          silhouette_score(X, model.labels_, metric='euclidean'))


# For Affinity Propagation clustering algorithm
for n in range(-60, -50):
    af = AffinityPropagation(preference=n, random_state=0).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)
    print("Silhouette score with Affinity Propagation algorithm and %d preference is: " % n,
          silhouette_score(X, labels, metric='sqeuclidean'))
