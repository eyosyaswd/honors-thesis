"""
Script that generates the classifier and does the training.
"""

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline


def train(x_train, y_train, x_test, y_test):
	print("\nInitializing classifier...")
	vectorizer = TfidfVectorizer(ngram_range=(1,2))
	print("vectorizer params:", vectorizer.get_params())

	svm_classifier = svm.LinearSVC()
	print("linear svc params", svm_classifier.get_params())

	svm_pipeline = Pipeline([("vectorizer", vectorizer), ("linear_svc", svm_classifier)])

	print("\nTraining...")
	print(svm_pipeline.fit(x_train, y_train))

	print("\nPerforming cross validation:")
	k_fold = KFold(n_splits=20, shuffle=True, random_state=1)
	scores = cross_val_score(svm_pipeline, x_train, y=y_train, cv=k_fold)
	# scores = cross_val_score(svm_pipeline, x_train, y_train)
	print("scores:", scores)
	print("scores.mean:", scores.mean())



	print("\nPredicting test data set...")
	# print(svm_pipeline.score(x_test, y_test))
	predictions = svm_pipeline.predict(x_test)

	print("Classification report using testing data:")
	print(metrics.classification_report(y_true=y_test, y_pred=predictions, target_names=["negative", "neutral", "positive"]))

	print("Confusion matrix using testing data:")
	print(metrics.confusion_matrix(y_test, predictions))

	print("Performing grid search to find best hyperparameters:")
	

	# for x, prediction, y in zip(x_val, predictions, y_val):
		# if prediction != y:
			# print("Incorrect:", x, 'has been classified as', prediction, 'and should be ', y)
		# else:
		# if prediction == y:
			# print(prediction)
			# print("Correct:", x, "has been classified as", prediction, "and is correct!")

	# print("Prediction:", svm_pipeline.predict(x_val))
	# svm_pipeline.predict(x_train)

	# print(svm_pipeline.score(x_val, y_val))
	# print(svm_pipeline.score(x_test, y_test))
