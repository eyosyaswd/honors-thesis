"""
Script that generates the classifier and does the training.
"""

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer



def train(x_train, y_train, x_test, y_test):
	print("Initializing classifier...\n")
	vectorizer = TfidfVectorizer()
	svm_classifier = svm.LinearSVC(C=0.1)
	svm_pipeline = Pipeline([("vectorizer", vectorizer), ("linear_svc", svm_classifier)])

	print("Training...\n")
	svm_pipeline.fit(x_train, y_train)

	print("Predicting...\n")
	# svm_pipeline.predict(x_train)
	print(svm_pipeline.score(x_test, y_test))
