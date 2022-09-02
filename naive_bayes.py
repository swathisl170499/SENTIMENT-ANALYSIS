import numpy as np 
import sys
import string
import re
from math import log
from math import ceil
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


class NaiveBayes:

	def get_classes(self, labels):

		sentiments = {}

		for label in labels:

			sentiments[label] = True

		return sentiments


	def get_vocabulary(self, data):

		vocabulary = {}

		for d in data:
			text = d.split()
			for word in text: 
				if word:
					vocabulary[word] = True

		return vocabulary


	def training_counts(self, data, lables, vocabulary, sentiments):

		probabilities = []
		word_class_counts = {}
		sentiment_counts = {}

		for s in sentiments:
			word_counts = {}
			for i in range(len(data)):

				if lables[i] != s:
					continue

				if s in sentiment_counts:
					sentiment_counts[s] += 1
				else:
					sentiment_counts[s] = 1

				text = data[i]

				for word in vocabulary:
					if word in text:
						if word in word_counts:
							word_counts[word] += 1
						else:
							word_counts[word] = 1

			word_class_counts[s] = word_counts

		return word_class_counts, sentiment_counts


	def predict(self, data, word_class_counts, sentiment_counts, V):

		predictions = []
		labels = []

		np.seterr(divide = 'ignore') 

		for i in range(len(data)):

			likelihoods = {}
			
			text = data[i]

			text_content = text.split()

			for s in sentiment_counts:

				probability = np.e

				class_prob = (sentiment_counts[s] / sum(sentiment_counts.values()))

				for word in text_content:
					if word:
						try:
							word_class_count = word_class_counts[s][word] + 1
						except:
							word_class_count = 1

						word_class_prob = word_class_count / (sentiment_counts[s] + V)

						probability = np.exp(np.log(probability) + np.log(word_class_prob))

				probability = np.exp(np.log(probability) + np.log(class_prob))			
				likelihoods[s] = probability

			max_sentiment_class = max(likelihoods, key=likelihoods.get)
			predictions.append(max_sentiment_class)

		return predictions


	def __init__(self, X_train, X_test, y_train, y_test):

		sentiments = self.get_classes(y_train)
		vocabulary = self.get_vocabulary(X_train)
		V = len(vocabulary)
		word_class_counts, sentiment_counts = self.training_counts(X_train, y_train, vocabulary, sentiments)

		y_pred = self.predict(X_test, word_class_counts, sentiment_counts, V)

		accuracy = accuracy_score(y_pred, y_test)
		print(f"Accuracy: {accuracy}")

		f1 = f1_score(y_pred, y_test, average='weighted')
		print(f"F1-Score: {f1}")		

		print("Confusion Matrix:")
		print(confusion_matrix(y_test, y_pred))


def extract_data(input):

	data = []
	labels = []

	with open(input, 'r') as f:
		lines = f.read()
		examples = lines.split("\n")

		n_count = 0
		p_count = 0

		for e in examples:
			if e:
				segments = e.split("\t")

				text = segments[2]
				sentiment = segments[1]
				

				# use for checking 2-class accuracy
				'''

				if sentiment == "neutral":
					continue
				'''


				#use for class balancing

				'''

				if sentiment == "neutral":
					n_count = n_count + 1

					if n_count % 3 != 0:
						continue

				if sentiment == "positive":

					p_count = p_count + 1

					if p_count % 2 == 0:
						continue

				'''

				text = text.translate(str.maketrans('', '', string.punctuation))
				text = text.lower()
				data.append(text)
				labels.append(sentiment)


		return data, labels

input_file = "2016test_processed.tsv"

X, y = extract_data(input_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

naive_bayes = NaiveBayes(X_train, X_test, y_train, y_test)
