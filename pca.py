import numpy as np
import os
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import *

#cluster = 10
#load feature vector
seven = np.loadtxt('results.csv', delimiter=',')
label = seven[:,0]
seven = np.delete(seven, 0, axis=1)

pca = PCA(n_components='mle')
#pca.fit(seven)

seven = pca.fit_transform(seven)

kmeans_model = KMeans(random_state=1).fit(seven)
labels = kmeans_model.labels_
silhouette_score(seven, labels, metric='euclidean')

for i in range(len(set(labels))):
    os.mkdir('./7/'+str(i)+'/')
for i in range(len(labels)):
    shutil.copyfile('./data/testing/7/' + str(int(label[i]))+'.png', './7/' + str(labels[i]) + '/' + str(int(label[i]))+'.png')

