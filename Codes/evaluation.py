from util import *
import math
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs: list, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		top_k = query_doc_IDs_ordered[:k]
		precision = 0
	
		for value in true_doc_IDs:
			if int(value) in top_k:
				precision += 1
		precision /= k
		return precision

	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1
		Precesion = 0
		#Fill in code here
		query_true_doc_ids = []
		true_doc_ids = []
		for i in range(len(qrels) - 1):
			if qrels[i]['query_num'] != qrels[i+1]['query_num']:
				true_doc_ids.append(int(qrels[i]['id']))
				query_true_doc_ids.append(true_doc_ids)
				true_doc_ids = []
			else:
				true_doc_ids.append(int(qrels[i]['id']))
		true_doc_ids.append(qrels[i+1]['id'])
		query_true_doc_ids.append(true_doc_ids)

		for i in range(len(query_ids)):
			Precesion += self.queryPrecision(doc_IDs_ordered[i], query_ids[i], query_true_doc_ids[i], k)
		meanPrecision = Precesion/len(query_ids) if len(query_ids) > 0 else 0.0
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		top_k = query_doc_IDs_ordered[:k]
		recall = 0
		for value in true_doc_IDs:
			if int(value) in top_k:
				recall += 1
		recall  /= len(true_doc_IDs)
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		Recall = 0
		#Fill in code here
		query_true_doc_ids = []
		true_doc_ids = []

		# construction of the query and their relevant documents list
		for i in range(len(qrels) - 1):
			if qrels[i]['query_num'] != qrels[i+1]['query_num']:
				true_doc_ids.append(qrels[i]['id'])
				query_true_doc_ids.append(true_doc_ids)
				true_doc_ids = []
			else:
				true_doc_ids.append(qrels[i]['id'])
		true_doc_ids.append(qrels[i+1]['id'])
		query_true_doc_ids.append(true_doc_ids)

		# iterating through the queries for further calculation
		for i in range(len(query_ids)):
			Recall += self.queryRecall(doc_IDs_ordered[i], query_ids[i], query_true_doc_ids[i], k)
		meanRecall = Recall/len(query_ids) if len(query_ids) > 0 else 0.0

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		Precesion = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs,k)
		Recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs,k)
		fscore = 1/((1/Precesion)+1/(Recall)) if Precesion != 0.0 and Recall != 0.0 else 0.0
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1
		Fscore = 0
		#Fill in code here
		query_true_doc_ids = []
		true_doc_ids = []

		for i in range(len(qrels) - 1):
			if qrels[i]['query_num'] != qrels[i+1]['query_num']:
				true_doc_ids.append(qrels[i]['id'])
				query_true_doc_ids.append(true_doc_ids)
				true_doc_ids = []
			else:
				true_doc_ids.append(qrels[i]['id'])
		# adding the last element and adding to the true doc ids of queries
		true_doc_ids.append(qrels[i+1]['id'])
		query_true_doc_ids.append(true_doc_ids)

		for i in range(len(query_ids)):
			Fscore += self.queryFscore(doc_IDs_ordered[i], query_ids[i], query_true_doc_ids[i], k)
		meanFscore = Fscore/len(query_ids) if len(query_ids) > 0 else 0.0
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		relevances = [1 if doc_id in true_doc_IDs else 0 for doc_id in query_doc_IDs_ordered[:k]]
		dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
		idcg = sum(1 / np.log2(idx + 2) for idx in range(min(len(true_doc_IDs), k)))
		nDCG = dcg / idcg if idcg > 0 else 0
		return nDCG



	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		nDCGs = []
		for idx, query_id in enumerate(query_ids):
			true_doc_IDs = []
			for doc in qrels:
				# print(type(doc['query_num']))
				if doc['query_num'] == str(query_id):
					true_doc_IDs.append(int(doc['id']))
			nDCGs.append(self.queryNDCG(doc_IDs_ordered[idx], query_id, true_doc_IDs, k))
		meanNDCG = sum(nDCGs) / len(nDCGs) if nDCGs else 0
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		# num_relevant_docs_retrieved = 0
		# total_precision = 0.0

		relevant_indices = [i for i, doc_id in enumerate(query_doc_IDs_ordered[:k]) if doc_id in true_doc_IDs]
		precisions = [self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i + 1) for i in relevant_indices]
		avgPrecision = sum(precisions) / len(precisions) if precisions else 0
		return avgPrecision



	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		averagePrecisions = []
		for idx, query_id in enumerate(query_ids):
			true_doc_IDs = []
			for doc in q_rels:
				# print(type(doc['query_num']))
				if doc['query_num'] == str(query_id):
					true_doc_IDs.append(int(doc['id']))
			averagePrecisions.append(self.queryAveragePrecision(doc_IDs_ordered[idx], query_id, true_doc_IDs, k))
		meanAveragePrecision = sum(averagePrecisions) / len(averagePrecisions) if averagePrecisions else 0

		return meanAveragePrecision

