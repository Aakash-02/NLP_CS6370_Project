from util import *

# Add your import statements here
import numpy as np

class InformationRetrieval():
    def __init__(self):
        self.index = None
        self.docs = None
    def buildIndex(self, docs, docIDs):
            
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None

        #Fill in code here
        # Inverted Index
        vocab = []
        docs_words = []

        for doc in docs:
            docs_words.append([])
            for sent in doc:
                vocab += sent
                docs_words[-1] += sent

        vocab = list(set(vocab))

        index = {word : [] for word in vocab}

        for word in vocab:
            for i in range(len(docs_words)):
                if word in docs_words[i]:
                    index[word].append(docIDs[i])

        self.index = index
        self.docs = {}
        for i in range(len(docIDs)):
            self.docs[docIDs[i]] = docs[i]
        # self.docs = {docIDs: doc for doc in docs}
        self.vocab = vocab
        # print("index",index)
    def rank_docs_query(self, query, docIDs):
        docs = [self.docs[idx] for idx in docIDs]

        # unpacking the last two dimensions to have the list of docs and each having a sub list of all words in the doc
        doc_words = []
        for doc in docs:
            doc_words.append([])
            for sent in doc:
                doc_words[-1] += sent
        # tf calculation for the docs
        tf = np.zeros((len(self.vocab), len(docs)))
        for i in range(len(doc_words)):
            # print("docs: ", docs[i])
            # print("doc_words: ",doc_words[i])
            for j in range(len(self.vocab)):
                tf[j, i] += doc_words[i].count(self.vocab[j])
            # print("vocab: ", self.vocab[j])
            tf[:, i] /= len(doc_words[i])
        # print(sum(sum(tf)))
        # print(self.vocab)
        # idf calculation for the docs
        idf = np.ones((len(self.vocab), 1))
        for i in range(len(self.vocab)):
            for j in range(len(doc_words)):
                if self.vocab[i] in doc_words[j]:
                    idf[i] += 1
        # print("TF: ", tf)
        # tf-idf calculation
        tf_idf = tf * np.log((len(doc_words)+len(self.vocab)) / idf)
        # query vector:
        query_words = []
        for sent in query:
            query_words += sent

        query_vector = np.zeros((len(self.vocab), 1))
        for i in range(len(self.vocab)):
            query_vector[i] += query_words.count(self.vocab[i])
        query_vector = query_vector * np.log(len(doc_words) / idf)

        
        # cosine similarity
        cos_sim = tf_idf.T @ query_vector / (np.linalg.norm(tf_idf) * np.linalg.norm(query_vector))
        print(cos_sim)
        # ranking
        rank = np.argsort(cos_sim, axis=0)
        # print(rank)
        # print(type(rank))
        # print(type(rank[0]))
        # print("rank_sum: ", np.sum(rank))
        rank = [docIDs[i[0]] for i in rank]

        return rank
    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []
        
        for query in queries:
            # creating all queries in terms of words
            print(query)
            words = []
            for sent in query:
                words += sent
            words = list(set(words))
            # retrieving relevant documents
            docs_rel = []
            for word in words:
                # assert word in self.vocab
                # print("word: ", word)
                # print("index: ",self.index[word])
                if word in self.index.keys():
                    docs_rel +=self.index[word]
            docs_rel = list(set(docs_rel))
            # print(docs_rel)
            # ranking the documents
            rank = self.rank_docs_query(query, docs_rel)
            doc_IDs_ordered.append(rank)
            
        return doc_IDs_ordered