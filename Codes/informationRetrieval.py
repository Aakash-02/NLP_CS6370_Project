from util import *

# Add your import statements here
import numpy as np
from tqdm import tqdm
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer

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
        v = np.array(self.vocab)
        np.save("vocab", v)
       
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
        
        # tf calculation for the docs
        tf = np.zeros((len(self.vocab), len(docs)))
        for i in range(len(doc_words)):
            for j in range(len(self.vocab)):
                tf[j, i] += doc_words[i].count(self.vocab[j])
            
            tf[:, i] = tf[:,i] / len(doc_words[i]) if len(doc_words[i]) !=0 else 0.0
        print("=============doc_tf done=============")

        # idf calculation for the docs
        idf = np.zeros((len(self.vocab), 1))
        for i in range(len(self.vocab)):
            for j in range(len(doc_words)):
                if self.vocab[i] in doc_words[j]:
                    idf[i] += 1
        # print("TF: ", tf)

        # tf-idf calculation
        print("=============idf done=============")
        tf_idf = tf * np.log((len(doc_words)) / (idf + 1e-9))
        
        print("=============doc_tfidf done=============")
        query_words = []
        for query in queries:
            query_words.append([])
            for sent in query:
                query_words[-1] += sent

        # tf calculation for the docs
        tf_q = np.zeros((len(self.vocab), len(queries)))
        for i in range(len(query_words)):
            for j in range(len(self.vocab)):
                tf_q[j, i] += query_words[i].count(self.vocab[j])
            tf_q[:, i] /= len(query_words[i])

        tf_idf_q_sklearn = vectorizer.transform(query_words)

        print("=============query_tf done=============")
        tf_idf_q =  tf_q * np.log((len(doc_words)) / (idf + 1e-9))
        print("=============query_tfidf done=============")

        tf_idf = tf_idf_sklearn.todense().T
        tf_idf_q = tf_idf_q_sklearn.todense().T
        print(tf_idf.shape, tf_idf_q.shape)

        norm_d = np.linalg.norm(tf_idf, axis=0).reshape(-1,1)
        norm_q = np.linalg.norm(tf_idf_q, axis = 0).reshape(-1,1)
        cos_sim = tf_idf_q.T@tf_idf/(norm_q@norm_d.T + 1e-7)
        print("=============cosine done=============")

        ranks = []
        for i in tqdm(range(len(cos_sim))):
            rank = np.argsort(cos_sim[i,:], axis=1)
            rank = [self.doc_IDs[rank[0, j]] for j in range(rank.shape[1])]
            ranks.append(rank)
       
        doc_IDs_ordered = ranks
        return doc_IDs_ordered