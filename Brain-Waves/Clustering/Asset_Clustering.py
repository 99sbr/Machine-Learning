import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
from scipy import stats
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt

def outlier_detection(x):
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    l=q25-iqr
    u=q75+iqr
    return l,u

def Plot_Dendogram(matrix):
	%matplotlib inline
	Z = hac.linkage(np.asarray(matrix),'ward')
	# Plot the dendogram
	plt.figure(figsize=(25, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	hac.dendrogram(
	    Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=8.,  # font size for the x axis labels
	)
	plt.show()

def K_Means(matrix):
	
	K = range(5,20)
	KM = [KMeans(n_clusters=k).fit(matrix) for k in K]
	centroids = [k.cluster_centers_ for k in KM]
	D_k = [cdist(matrix, cent, 'euclidean') for cent in centroids]
	cIdx = [np.argmin(D,axis=1) for D in D_k]
	dist = [np.min(D,axis=1) for D in D_k]
	avgWithinSS = [sum(d)/matrix.shape[0] for d in dist]
	# Total with-in sum of square
	wcss = [sum(d**2) for d in dist]
	tss = sum(pdist(matrix)**2)/matrix.shape[0]
	bss = tss-wcss
	kIdx = 7
	# elbow curve
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(K, avgWithinSS, 'b*-')
	ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
	markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
	plt.grid(True)
	plt.xlabel('Number of clusters')
	plt.ylabel('Average within-cluster sum of squares')
	plt.title('Elbow for KMeans clustering')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(K, bss/tss*100, 'b*-')
	plt.grid(True)
	plt.xlabel('Number of clusters')
	plt.ylabel('Percentage of variance explained')
	plt.title('Elbow for KMeans clustering')

def Save_output(data,clusters):
	file = open("asset_cluster.txt", "a")
	file.write("Asset,Cluster\n")
	for i in range(0,len(data)):
    file.write(str(list(data.index)[i]))
    file.write(",")
    file.write(str(clusters[i]))
    file.write("\n")
    file.close()
    
if __name__ == '__main__':
	train=pd.read_csv("train.csv")
	test=pd.read_csv("test.csv")
	y=train.Y
	train=train.drop(['Y'],1)
	train_matrix = train.pivot_table(columns=['Time'])
	test_matrix=test.pivot_table(columns=['Time'])
	data=pd.concat([train_matrix,test_matrix],1)

	print("Data preprocessing")
    # Outlier detection
	for i in range(0,len(data.iloc[0,:])):
		l,u=outlier_detection(data.iloc[:,i])
		for j in range(0,len(data.iloc[:,i])):
			if (data.iloc[j:j+1,i]).values[0]>u:
				data.iloc[j:j+1,i].values[0]=u
			elif data.iloc[j:j+1,i].values[0]<l:
				data.iloc[j:j+1,i].values[0]=l
	# Normalization
	for i in range(0,len(data.columns)):
		data.iloc[:,i]=preprocessing.normalize(data.iloc[:,i], norm='l2').reshape(100,1)
	matrix=np.corrcoef(data)
	# Data Visualization and cluster determination
	Plot_Dendogram(matrix)
	K_Means(matrix)
	Z= hac.linkage(matrix,'ward')
	clusters=fcluster(Z_best,12, criterion='maxclust')
	Save_output(data,clusters)