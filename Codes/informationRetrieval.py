import gensim.downloader
from util import *

# Add your import statements here
import numpy as np
from tqdm import tqdm
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import umap
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
        self.docs = docs
        self.vocab = vocab
        self.doc_IDs = docIDs
        print(vocab)
        # v = np.array(self.vocab)
        # np.save("vocab", v)
       
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
        docs = self.docs

        # model = gensim.models.Word2Vec(self.vocab, min_count = 1, vector_size= 128, window = 5)
        # word_embed = np.zeros((len(self.vocab), 128))
        # for word in self.vocab:
        #     print(word)
        #     word_embed[i] = model.wv[word]
        # print(word_embed)

        # unpacking the last two dimensions to have the list of docs and each having a sub list of all words in the doc
        doc_words = []
        for doc in docs:
            doc_words.append([])
            for sent in doc:
                doc_words[-1] += sent
        
        def dummy_func(docs):
            return docs
        vectorizer = TfidfVectorizer(analyzer= 'word', tokenizer=dummy_func, preprocessor=dummy_func, token_pattern=None)
        tf_idf_sklearn = vectorizer.fit_transform(doc_words)
        
        # # tf calculation for the docs
        # tf = np.zeros((len(self.vocab), len(docs)))
        # for i in range(len(doc_words)):
        #     for j in range(len(self.vocab)):
        #         tf[j, i] += doc_words[i].count(self.vocab[j])
            
        #     tf[:, i] = tf[:,i] / len(doc_words[i]) if len(doc_words[i]) !=0 else 0.0
        # print("=============doc_tf done=============")

        # # idf calculation for the docs
        # idf = np.zeros((len(self.vocab), 1))
        # for i in range(len(self.vocab)):
        #     for j in range(len(doc_words)):
        #         if self.vocab[i] in doc_words[j]:
        #             idf[i] += 1
        # # print("TF: ", tf)

        # # tf-idf calculation
        # print("=============idf done=============")
        # tf_idf = tf * np.log((len(doc_words)) / (idf + 1e-9))
        
        # print("=============doc_tfidf done=============")
        query_words = []
        for query in queries:
            query_words.append([])
            for sent in query:
                query_words[-1] += sent

        # # tf calculation for the docs
        # tf_q = np.zeros((len(self.vocab), len(queries)))
        # for i in range(len(query_words)):
        #     for j in range(len(self.vocab)):
        #         tf_q[j, i] += query_words[i].count(self.vocab[j])
        #     tf_q[:, i] /= len(query_words[i])

        tf_idf_q_sklearn = vectorizer.transform(query_words)
        # print("=============query_tf done=============")
        # tf_idf_q =  tf_q * np.log((len(doc_words)) / (idf + 1e-9))
        # print("=============query_tfidf done=============")

        tf_idf = tf_idf_sklearn.todense().T
        tf_idf_q = tf_idf_q_sklearn.todense().T

        norm_d = np.linalg.norm(tf_idf, axis=0).reshape(-1,1)
        norm_q = np.linalg.norm(tf_idf_q, axis = 0).reshape(-1,1)
        print("tf_idf_q.T: ",tf_idf_q.shape)
        print("tf_idf: ", tf_idf.shape)
        print("norm_q: ", norm_q.shape)
        print("norm_d.T: ",norm_d.T.shape)
        cos_sim = tf_idf_q.T@tf_idf/(norm_q@norm_d.T + 1e-7)
        print("=============cosine done=============")

        ranks = []
        for i in tqdm(range(len(cos_sim))):
            rank = np.argsort(-cos_sim[i,:], axis=1)
            rank = [self.doc_IDs[rank[0, j]] for j in range(rank.shape[1])]
            ranks.append(rank)
       
        doc_IDs_ordered = ranks
        return doc_IDs_ordered
# class InformationRetrieval():

#     def __init__(self):
#         self.index = None
#         self.vectorizer = TfidfVectorizer()
#         self.docIDs = []

#     def buildIndex(self, docs, docIDs):
#         """
#         Builds the document index in terms of the document
#         IDs and stores it in the 'index' class variable

#         Parameters
#         ----------
#         arg1 : list
#             A list of strings where each string is a document.
#         arg2 : list
#             A list of integers denoting IDs of the documents
#         Returns
#         -------
#         None
#         """
#         # Convert list of lists of sentences into a list of strings
#         docs = [' '.join([' '.join(sentence) for sentence in doc]) for doc in docs]
#         # Building the TF-IDF matrix
#         self.index = self.vectorizer.fit_transform(docs)
#         self.docIDs = docIDs


#     def rank(self, queries):
#         """
#         Rank the documents according to relevance for each query

#         Parameters
#         ----------
#         arg1 : list
#             A list of strings where each string is a query.

#         Returns
#         -------
#         list
#             A list of lists of integers where the ith sub-list is a list of IDs
#             of documents in their predicted order of relevance to the ith query
#         """
#         doc_IDs_ordered = []

#         # Transform queries to TF-IDF
#         queries = [' '.join([' '.join(sentence) for sentence in query]) for query in queries]
#         query_tfidf = self.vectorizer.transform(queries)

#         # Calculate the cosine similarity between documents and queries
#         cosine_similarities = np.dot(query_tfidf, self.index.T).toarray()

#         # Sorting documents for each query based on similarity
#         for sims in cosine_similarities:
#             ranked_docs = [self.docIDs[i] for i in np.argsort(-sims)]
#             doc_IDs_ordered.append(ranked_docs)

#         return doc_IDs_ordered

class EmmbeddingRetrieval():
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
        # print(vocab)

        index = {word : [] for word in vocab}

        for word in vocab:
            for i in range(len(docs_words)):
                if word in docs_words[i]:
                    index[word].append(docIDs[i])

        self.index = index
        self.docs = docs
        self.vocab = vocab
        self.doc_IDs = docIDs
        vocab = np.array(vocab)
        np.save('vocab', vocab)
    
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
        docs = self.docs

        # model = gensim.downloader.load('word2vec-google-news-300')
        # word_embed = np.zeros((len(self.vocab), 128))
        # for word in tqdm(self.vocab):
        #     word_embed[i] = model[word]
        # print(word_embed)

        # unpacking the last two dimensions to have the list of docs and each having a sub list of all words in the doc
        # embed_vocab = np.load("vocab.npy")
        with open("vocab_vector.json") as f:
            embed = json.load(f)
        embed_vocab = embed.keys()
        doc_words = []
        for doc in docs:
            doc_words.append([])
            for sent in doc:
                for wo in sent:
                    if wo not in embed_vocab:
                        sent.remove(wo)
                doc_words[-1] += sent
        print(doc_words)
        def dummy_func(docs):
            return docs
        vectorizer = TfidfVectorizer(analyzer= 'word', tokenizer=dummy_func, preprocessor=dummy_func, token_pattern=None, vocabulary=embed_vocab)
        tf_idf_sklearn = vectorizer.fit_transform(doc_words)
        # print(tf_idf_sklearn_umap.shape)
        query_words = []
        for query in queries:
            query_words.append([])
            for sent in query:
                for wo in sent:
                    if wo not in embed_vocab:
                        sent.remove(wo)
                query_words[-1] += sent
        tf_idf_q_sklearn = vectorizer.transform(query_words)
        # print(tf_idf_sklearn_umap.shape)
        embed_vector = []
        for vocab_ in embed_vocab:
            embed_vector.append(embed[vocab_])
        embed_vector = np.array(embed_vector)
        print(embed_vector.shape)
        tf_idf = tf_idf_sklearn.todense().T
        tf_idf_q = tf_idf_q_sklearn.todense().T
        print(tf_idf.shape)
        print(tf_idf_q.shape)
        tf_idf = tf_idf.T@embed_vector
        tf_idf_q = tf_idf_q.T@embed_vector
        print(tf_idf_q.shape, tf_idf.shape)
        norm_d = np.linalg.norm(tf_idf.T, axis=0).reshape(-1,1)
        norm_q = np.linalg.norm(tf_idf_q.T, axis = 0).reshape(-1,1)
        print(norm_d.shape, norm_q.shape)
        cos_sim = tf_idf_q@tf_idf.T/(norm_q@norm_d.T + 1e-7)
        # cos_sim = cosine_similarity(tf_idf, tf_idf_q)

        ranks = []
        for i in tqdm(range(len(cos_sim))):
            rank = np.argsort(-cos_sim[i,:], axis=1)
            rank = [self.doc_IDs[rank[0, j]] for j in range(rank.shape[1])]
            ranks.append(rank)
       
        doc_IDs_ordered = ranks
        return doc_IDs_ordered
class TfIDFUmap():
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
        # print(vocab)

        index = {word : [] for word in vocab}

        for word in vocab:
            for i in range(len(docs_words)):
                if word in docs_words[i]:
                    index[word].append(docIDs[i])

        self.index = index
        self.docs = docs
        self.vocab = vocab
        self.doc_IDs = docIDs
        vocab = np.array(vocab)
        np.save('vocab', vocab)
    
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
        docs = self.docs

        # model = gensim.downloader.load('word2vec-google-news-300')
        # word_embed = np.zeros((len(self.vocab), 128))
        # for word in tqdm(self.vocab):
        #     word_embed[i] = model[word]
        # print(word_embed)

        # unpacking the last two dimensions to have the list of docs and each having a sub list of all words in the doc
        # embed_vocab = np.load("vocab.npy")
        with open("vocab_vector.json") as f:
            embed = json.load(f)
        embed_vocab = embed.keys()
        doc_words = []
        for doc in docs:
            doc_words.append([])
            for sent in doc:
                for wo in sent:
                    if wo not in embed_vocab:
                        sent.remove(wo)
                doc_words[-1] += sent
        
        def dummy_func(docs):
            return docs
        vectorizer = TfidfVectorizer(analyzer= 'word', tokenizer=dummy_func, preprocessor=dummy_func, token_pattern=None, vocabulary=embed_vocab)
        tf_idf_sklearn = vectorizer.fit_transform(doc_words)
        
        query_words = []
        for query in queries:
            query_words.append([])
            for sent in query:
                for wo in sent:
                    if wo not in embed_vocab:
                        sent.remove(wo)
                query_words[-1] += sent
        tf_idf_q_sklearn = vectorizer.transform(query_words)
        
        
        tf_idf = tf_idf_sklearn.todense().T
        tf_idf_q = tf_idf_q_sklearn.todense().T
        
        reducer = umap.UMAP()
        tf_idf_sklearn_umap = reducer.fit_transform(tf_idf_sklearn)
        tf_idf_q_sklearn_umap = reducer.transform(tf_idf_q_sklearn)

        tf_idf = tf_idf_sklearn_umap
        tf_idf_q = tf_idf_q_sklearn_umap
        print(tf_idf_q.shape, tf_idf.shape)
        norm_d = np.linalg.norm(tf_idf.T, axis=0).reshape(-1,1)
        norm_q = np.linalg.norm(tf_idf_q.T, axis = 0).reshape(-1,1)
        print(norm_d.shape, norm_q.shape)
        cos_sim = tf_idf_q@tf_idf.T/(norm_q@norm_d.T + 1e-7)
        # cos_sim = cosine_similarity(tf_idf, tf_idf_q)
        print(cos_sim.shape)
        ranks = []
        for i in tqdm(range(len(cos_sim))):
            rank = np.argsort(-cos_sim[i,:], axis=0)
            # print(rank)
            rank = [self.doc_IDs[rank[j]] for j in range(rank.shape[0])]
            ranks.append(rank)
       
        doc_IDs_ordered = ranks
        return doc_IDs_ordered

class LSA():
    def __init__(self, k):
        self.index = None
        self.docs = None
        self.k = k
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
        # print(vocab)

        index = {word : [] for word in vocab}

        for word in vocab:
            for i in range(len(docs_words)):
                if word in docs_words[i]:
                    index[word].append(docIDs[i])

        self.index = index
        self.docs = docs
        self.vocab = vocab
        self.doc_IDs = docIDs
        vocab = np.array(vocab)
        np.save('vocab', vocab)
    
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
        docs = self.docs

        # unpacking the last two dimensions to have the list of docs and each having a sub list of all words in the doc
        doc_words = []
        for doc in docs:
            doc_words.append([])
            for sent in doc:
                doc_words[-1] += sent
        
        def dummy_func(docs):
            return docs
        vectorizer = TfidfVectorizer(analyzer= 'word', tokenizer=dummy_func, preprocessor=dummy_func, token_pattern=None)
        print("initalised model")
        tf_idf_sklearn = vectorizer.fit_transform(doc_words)
        print("fit done for doc")
        query_words = []
        for query in queries:
            query_words.append([])
            for sent in query:
                query_words[-1] += sent
        print("transformed query")
        tf_idf_q_sklearn = vectorizer.transform(query_words)

        tf_idf = tf_idf_sklearn.todense().T
        tf_idf_q = tf_idf_q_sklearn.todense().T

        U, S, Vt = np.linalg.svd(tf_idf, full_matrices=True)
        print("SVD done")
        print("U:",U.shape)
        print("S:",S.shape)
        print("Vt:",Vt.shape)
        U_, S_, Vt_ = U[:, :self.k], np.diag(S[:self.k]), Vt[:self.k, :]
        tf_idf_rank_k = (U_ @ S_) @ Vt_
        print("rank calculated")
        tf_idf = tf_idf_rank_k

        norm_d = np.linalg.norm(tf_idf, axis=0).reshape(-1,1)
        norm_q = np.linalg.norm(tf_idf_q, axis = 0).reshape(-1,1)
        cos_sim = tf_idf_q.T@tf_idf/(norm_q@norm_d.T + 1e-7)
        print("cosine sim done")

        ranks = []
        for i in tqdm(range(len(cos_sim))):
            rank = np.argsort(-cos_sim[i,:], axis=1)
            rank = [self.doc_IDs[rank[0, j]] for j in range(rank.shape[1])]
            ranks.append(rank)
        print("rank done")
        doc_IDs_ordered = ranks
        return doc_IDs_ordered