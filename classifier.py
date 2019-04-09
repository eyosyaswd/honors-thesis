"""
Script that generates the classifier and does the training.
"""

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer



def train(x_train, y_train, x_val, y_val, x_test, y_test):
	print("Initializing classifier...\n")
	vectorizer = TfidfVectorizer()
	svm_classifier = svm.LinearSVC(C=0.1)
	svm_pipeline = Pipeline([("vectorizer", vectorizer), ("linear_svc", svm_classifier)])

	print("Training...\n")
	svm_pipeline.fit(x_train, y_train)

	print("Predicting...\n")
	# predictions = svm_pipeline.predict(x_val)
	# for x, prediction, y in zip(x_val, predictions, y_val):
		# if prediction != y:
			# print("Incorrect:", x, 'has been classified as', prediction, 'and should be ', y)
		# else:
		# if prediction == y:
			# print(prediction)
			# print("Correct:", x, "has been classified as", prediction, "and is correct!")

	# print("Prediction:", svm_pipeline.predict(x_val))
	# svm_pipeline.predict(x_train)

	print(svm_pipeline.score(x_val, y_val))
