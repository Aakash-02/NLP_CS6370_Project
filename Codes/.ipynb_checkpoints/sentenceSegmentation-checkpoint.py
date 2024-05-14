from util import *

# Add your import statements here
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
        #Fill in code here
        delimiters = [".", '!', "?", "\n"] # list of delimiters
        # iterating through all the delimiters
        for delimiter in delimiters:
            # splitting and joining them using a space
            segementedText = " ".join(text.split(delimiter))
        # splitting using the space
        segmentedText = segmentedText.split()
        return segmentedText

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
        nltk.download()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        segmentedText = tokenizer.tokenize(text)
        return segmentedText