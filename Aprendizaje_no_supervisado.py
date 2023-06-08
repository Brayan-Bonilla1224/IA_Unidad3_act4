import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


np.random.seed(42)  

X = np.random.randint(10, 100, size=30)
Y = np.random.randint(10, 100, size=30)

data = pd.DataFrame({'X': X, 'Y': Y})
print(data)


plt.scatter(data['X'], data['Y'])
plt.show()


kmeans = KMeans(n_clusters=5).fit(data)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(data['X'], data['Y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,1], c='red')
plt.show()