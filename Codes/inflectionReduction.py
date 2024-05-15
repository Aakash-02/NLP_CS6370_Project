from util import *

# Add your import statements here
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here
		reducedText = []
		wnl = WordNetLemmatizer()
		for sent in text:
			reducedText.append([wnl.lemmatize(word) for word in sent])
		return reducedText

	# def lemmatize(self, text):
	# 	reducedText = []
	# 	lemmatizer = WordNetLemmatizer()
	# 	for sent in text:
	# 		reducedText.append([])
	# 		for word in sent:
	# 			reducedText[-1].append(lemmatizer.lemmatize(word))
	# 	return reducedText
        


