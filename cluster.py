from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt("result1.txt", dtype=float)
dataset = dataset/1000
X = dataset.reshape(5,1000).T
kmeans = KMeans(2, random_state=0)
sse = {}

# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
#     #X["clusters"] = kmeans.labels_
#     #print(data["clusters"])
#     sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
# plt.figure()
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel("Number of cluster")
# plt.ylabel("SSE")
# plt.show()


model = kmeans.fit(X)
labels = model.predict(X)

acc = model.score(X)
print(acc)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()

# gmm = GaussianMixture(n_components=5,covariance_type='diag').fit(X)
# labels1 = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels1, s=40, cmap='viridis');
# plt.show()