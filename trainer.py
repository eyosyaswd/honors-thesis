"""
Script that generates the classifier and does the training.
"""

import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline


def train(x_train, y_train, x_test, y_test):

	######################## INIT ###################################
	print("\nINITIALIZING CLASSIFIER...")

	vectorizer = TfidfVectorizer(ngram_range=(1,2))
	print("vectorizer params:", vectorizer.get_params())

	linear_svc = svm.LinearSVC()
	print("linear svc params", linear_svc.get_params())

	linear_svc_pipeline = Pipeline(steps=[("vectorizer", vectorizer), ("linear_svc", linear_svc)])


	################### CROSS VAL and GRID SEARCH ####################
	print("\nPERFORMING GRID SEARCH WITH CROSS VALIDATION...")
	k_fold = KFold(n_splits=20, shuffle=True, random_state=1)
	linear_svc_params = [
		{
			"linear_svc__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
			"linear_svc__loss": ["hinge", "squared_hinge"],
			"linear_svc__tol": [1e-4, 1e-5]
		}
	]


	scores = ["precision", "recall"]

	for score in scores:
		print("# Tuning hyper-parameters for {0}".format(score))
		print()

		grd = GridSearchCV(linear_svc_pipeline, param_grid=linear_svc_params, cv=k_fold, scoring="{0}_macro".format(score))
		grd.fit(x_train, y_train)

		print("\nBest score and parameters set found on development set:")
		print("Score:", grd.best_score_, "Params:", grd.best_params_)
		print()

		print("All grid scores on development set:")
		means = grd.cv_results_["mean_test_score"]
		stds = grd.cv_results_["std_test_score"]
		for mean, std, params in zip(means, stds, grd.cv_results_["params"]):
			print("{0:0.3f} (+/-{1:0.3f}) for {2}".format(mean, std*2, params))
		print()

		print("Detaiiled classification report:")
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		y_true, y_pred = y_test, grd.predict(x_test)
		print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
		print()

		print("Confusion matrix:")
		print(confusion_matrix(y_true, y_pred))
		print()
		print()

	create_model(x_train, y_train, x_test, y_test)


def create_model(x_train, y_train, x_test, y_test):
	""" Create a trained model using the best parameters. """

	print("\nCREATING FINAL MODEL...")

	vectorizer = TfidfVectorizer(ngram_range=(1,2))
	print("vectorizer params:", vectorizer.get_params())

	linear_svc = svm.LinearSVC(C=1.0, loss="hinge", tol=0.0001)
	print("linear svc params", linear_svc.get_params())

	linear_svc_pipeline = Pipeline(steps=[("vectorizer", vectorizer), ("linear_svc", linear_svc)])

	print("\nTRAINING FINAL MODEL...")
	linear_svc_pipeline.fit(x_train, y_train)

	print("\nPICKLING MODEL...")
	list_pickle = open("final_model/trained_linear_svc.pkl", "wb")
	pickle.dump(linear_svc_pipeline, list_pickle)
	list_pickle.close()

	print("\nUNPICKLING MODEL...")
	list_unpickle = open("final_model/trained_linear_svc.pkl", "rb")
	model = pickle.load(list_unpickle)
	list_unpickle.close()

	print("Detaiiled classification report for final model:")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	y_true, y_pred = y_test, model.predict(x_test)
	print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
	print()

	print("Confusion matrix for final model:")
	print(confusion_matrix(y_true, y_pred))
	print()
	print()





	# for x, prediction, y in zip(x_val, predictions, y_val):
		# if prediction != y:
			# print("Incorrect:", x, 'has been classified as', prediction, 'and should be ', y)
		# else:
		# if prediction == y:
			# print(prediction)
			# print("Correct:", x, "has been classified as", prediction, "and is correct!")
