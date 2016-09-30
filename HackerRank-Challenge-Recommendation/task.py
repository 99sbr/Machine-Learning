import pandas as pd
import csv
import distance
from operator import itemgetter
'''
this function writes the output challenge recommended to each
'''

def Csv_writer(challenge_set):
	  
	csv.register_dialect(
    'mydialect',
    delimiter = ' ',
    quotechar = '"',
    doublequote = False,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)
	with open('/home/subir_sbr/Desktop/hackerrank-challenge-recommendation-dataset/hackerrank-challenge-recommendation-dataset/recommend.csv', 'a') as mycsvfile:
		thedatawriter = csv.writer(mycsvfile, dialect='mydialect')
		for row in challenge_set:
			thedatawriter.writerow(row)

'''

Get_challenge_recommendation function takes each cluster of customers as an
argument and for each hacker in the cluster it calculates the hamming distance wrt each hacker in that
particular cluster.

After calculating the hamming distance i have sorted the hacker_ids by hamming distance
and selected the first 10 non-repeted challenge attempted by the hackers and also not solved by the hacker to whom i am 
recommending the challenge.

'''
def get_distance(data1, data2):
	return distance.hamming(data1,data2)

def _get_tuple_distance(training_instance, test_instance):
	return (training_instance, get_distance(test_instance, training_instance))

def Get_Challenge_Recommendation(cluster):
	for index in range(0,len(cluster)):
		print(" %d out of %d of this cluster done" %(index,len(cluster)))
		distances=[_get_tuple_distance(pd.Series(cluster.iloc[i]).values,
		pd.Series(cluster.iloc[index]).values) 
		for i in range(0,len(cluster))]
		sorted_distances = sorted(distances, key=itemgetter(1))
		near_hackers=[]
		for i in range(0,100):
				near_hackers.append(sorted_distances[i][0][1])

		#for hacker in near_hackers:
		 #   challenge.append(df[df['hacker_id']==hacker].challenge_id.tolist())
		recommend=[]
		challenge=[]
		challenge_set=[]
		challenge.append(near_hackers[0])
		for i in range(0,len(near_hackers)):
			if(i==0):
				newset=pd.DataFrame(df.loc[df['hacker_id'] == near_hackers[i]])		
				for i in range(0,len(newset)):
					if newset.solved.iloc[i]!=1 and len(recommend)<10 and newset.challenge_id.iloc[i] not in recommend:
			    	# newset.solved.iloc[i]!=1 ::: makes sure no attempted challenge by hacker is recommended again
			    	# len(recomend) < 10 ::: to ensure number of recommendatons is 10
			    	# newset.challenge_id.iloc[i] ::: not in recommend ensures no challenge is suggested again which already has been suggested
						recommend.append(newset.challenge_id.iloc[i])
						challenge.append(newset.challenge_id.iloc[i])
				
			else:
				newset=pd.DataFrame(df.loc[df['hacker_id'] == near_hackers[i]])
				for i in range(0,len(newset)):
					if len(recommend)<10 and newset.challenge_id.iloc[i] not in recommend:
			    	# len(recomend) < 10 ::: to ensure number of recommendatons is 10
			    	# newset.challenge_id.iloc[i] ::: not in recommend ensures no challenge is suggested again which already has been suggested
						recommend.append(newset.challenge_id.iloc[i])
						challenge.append(newset.challenge_id.iloc[i])
		challenge_set.append(challenge)
		Csv_writer(challenge_set)
				



if __name__ == '__main__':
	import pandas as pd
	challenge=pd.read_csv('challenges.csv')
	submissions=pd.read_csv('submissions.csv')
	n_challenge = challenge.challenge_id.unique().shape[0]
	n_contest =challenge.contest_id.unique().shape[0]
	print ('Number of challenge = ' + str(n_challenge) + ' | Number of contest = ' + str(n_contest)  )
	n_users = submissions.hacker_id.unique().shape[0]
	n_items =submissions.challenge_id.unique().shape[0]
	print ('Number of hackers = ' + str(n_users) + ' | Number of challenges = ' + str(n_items)  )
	print("shape of challenge:\n") 
	print(challenge.shape)
	print("challenge domain null count is %d \n" %challenge.domain.isnull().sum())
	print("challenge subdomain null count is %d \n" %challenge.subdomain.isnull().sum())
	submissions=submissions.drop('created_at',1)
	# I have deleted these columns because near about 50% of data was missing which
	# makes filling the missing value irrelevant and better to drop these variables
	del challenge['domain']
	del challenge['subdomain']
	df=pd.merge(submissions,challenge)
	matrix = df.pivot_table(index=['hacker_id'], columns=['challenge_id'], values='solved')
	# a little tidying up. fill NA values with 0 and make the index into a column
	matrix = matrix.fillna(0).reset_index()
	# save a list of the 0/1 columns. we'll use these a bit later
	x_cols = matrix.columns[1:]
	from sklearn.cluster import KMeans
	cluster = KMeans(n_clusters=5)
	# slice matrix so we only include the 0/1 indicator columns in the clustering
	matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[2:]])
	matrix.cluster.value_counts()
	from sklearn.decomposition import PCA
	# two components which makes data visualization and plotting easier
	pca = PCA(n_components=2)
	matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
	matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
	matrix = matrix.reset_index()
	hacker_clusters = matrix[['hacker_id', 'cluster', 'x', 'y']]
	print("Hacker cluster head:\n")
	print(hacker_clusters.head())
	df = pd.merge(submissions, hacker_clusters)
	df = pd.merge(challenge, df)
	'''
	un comment this  to get graphical overiew of clustering
	from ggplot import *

	ggplot(df, aes(x='x', y='y', color='cluster')) + \
    	geom_point(size=75) + \
    	ggtitle("Hackers Grouped by Cluster")'''
    # Grouping down the Hackers in different clusters
	cluster_0 = matrix[matrix['cluster'] == 0]
	cluster_1 = matrix[matrix['cluster'] == 1]
	cluster_2 = matrix[matrix['cluster'] == 2]
	cluster_3 = matrix[matrix['cluster'] == 3]
	cluster_4 = matrix[matrix['cluster'] == 4]
	clusters=[cluster_0,cluster_1,cluster_2,cluster_3,cluster_4]
	for cluster in clusters:	
		Get_Challenge_Recommendation(cluster)