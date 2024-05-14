# Add your import statements here
import json
import numpy as np




# Add any utility functions here
def stop_word_gen(docs, N):
    """ 
    This function is the bottom-up approach for generating stop words for a particular corpus
    -------
    Input: 
        docs: list -->  List of documents which is the corpus for the stop word generation
        N   :  int -->  Number of terms from the frequency table to considered as stop words
    """
    preprocess_doc = ""
    for doc in docs:
        preprocess_doc += ''.join(sentence_segmented.tokenize(doc))
    tokenized_text = tokenizer.tokenize(preprocess_doc)

    tokenized_text = sorted(tokenized_text)
    keys = set(tokenized_text)

    freq = {}
    for key in keys:
        freq[key] = tokenized_text.count(key)
        
    keys = list(freq.keys())
    values = list(freq.values())
    sorted_value_index = np.argsort(values)
    sorted_freq = {keys[i]: values[i] for i in reversed(sorted_value_index)}

    stop_words = list(sorted_freq.keys())[:N]
    
    return stop_words



