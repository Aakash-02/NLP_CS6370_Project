from util import *
import math
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
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
		top_k_query = set(query_doc_IDs_ordered[:k])
		precision = len(top_k_query.intersection(set(true_doc_IDs)))/k
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
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			doc_ID_ordered = doc_IDs_ordered[i]
			true_doc_Ids = qrels[i]
			Precesion += self.queryPrecision(doc_ID_ordered, query_id, true_doc_Ids, k)
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
		top_k_query = set(query_doc_IDs_ordered[:k])
		recall = len(top_k_query.intersection(set(true_doc_IDs)))/len(true_doc_IDs)
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
		for i in range(len(query_ids)):
			query_id = query_ids[i]
			doc_ID_ordered = doc_IDs_ordered[i]
			true_doc_Ids = qrels[i]
			Recall += self.queryPrecision(doc_ID_ordered, query_id, true_doc_Ids, k)
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
		for i in range(len(query_ids)):
			query_doc_id = doc_IDs_ordered[i]
			true_doc_id = qrels[i]
			query_id = query_ids[i]
			Fscore += self.queryFscore(query_doc_id, query_id, true_doc_id, k)
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

		nDCG = -1

		#Fill in code here
		DCG = 0.0
		IDCG = 0.0
		
		for i in range(min(k, len(query_doc_IDs_ordered))):
			doc_id = query_doc_IDs_ordered[i]
			relevance = 1 if doc_id in true_doc_IDs else 0
			DCG += (2**relevance - 1) / (math.log2(i + 2))
		
		for i in range(min(k, len(true_doc_IDs))):
			IDCG += (2**1 - 1) / (math.log2(i + 2))
		
		nDCG = DCG / IDCG if IDCG > 0 else 0.0

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

		meanNDCG = -1
		NDGC = 0.0
		#Fill in code here
		for i in range(len(query_ids)):
			doc_id_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_doc_id = qrels[i]
			NDGC += self.queryNDCG(doc_id_ordered, query_id, true_doc_id, k)
		meanNDCG = NDGC/len(query_ids) if len(query_ids) > 0 else 0.0
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
		num_relevant_docs_retrieved = 0
		total_precision = 0.0

		# Calculate precision at each retrieved document
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				num_relevant_docs_retrieved += 1
				precision_at_i = num_relevant_docs_retrieved / (i + 1)
				total_precision += precision_at_i

		# Calculate average precision
		avgPrecision = total_precision / len(true_doc_IDs) if len(true_doc_IDs) > 0 else 0.0

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

		meanAveragePrecision = -1
		AveragePrecesion = 0
		#Fill in code here
		for i in range(len(query_ids)):
			doc_id_ordered = doc_IDs_ordered[i]
			query_id = query_ids[i]
			true_id = q_rels[i]
			AveragePrecesion += self.queryAveragePrecision(doc_id_ordered, query_id, true_id, k)
			
		meanAveragePrecision = AveragePrecesion/len(query_ids) if len(query_ids) > 0 else 0.0
		return meanAveragePrecision

