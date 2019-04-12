"""
Script that generates the classifier and does the training.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline



def train(x_train, y_train, x_test, y_test):

	######################## INIT ###################################
	print("\nINITIALIZING CLASSIFIER...")

	# vectorizer = TfidfVectorizer(ngram_range=(1,2))
	vectorizer = TfidfVectorizer()
	print("vectorizer params:", vectorizer.get_params())

	linear_svc = svm.LinearSVC()
	print("linear svc params", linear_svc.get_params())

	linear_svc_pipeline = Pipeline(steps=[("vectorizer", vectorizer), ("linear_svc", linear_svc)])


	################### CROSS VAL and GRID SEARCH ####################
	print("\nPERFORMING GRID SEARCH WITH CROSS VALIDATION...")
	k_fold = KFold(n_splits=20, shuffle=True, random_state=1)
	# k_fold = KFold(n_splits=5, shuffle=True)
	linear_svc_params = [
		{	# Dual optimization
			"linear_svc__penalty": ["l2"],		# if l1, you can't use hinge
			"linear_svc__loss": ["hinge", "squared_hinge"],
			"linear_svc__dual": [True],	# if false, you can't use l2 or hinge
			# "linear_svc__tol": [1e-4, 1e-5],
			# "linear_svc__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
			"linear_svc__C":[0.1, 1],
			# "linear_svc__multi_class": ["ovr", "crammer_singer"],
			# "vectorizer__stop_words": [None, stop_words.ENGLISH_STOP_WORDS],
			"vectorizer__ngram_range": [(1,2), (1,3)],
			"vectorizer__max_df": [0.9, 1.0],
			# "vectorizer__use_idf": [True, False]
		},
		{	# Primal Optimization
			"linear_svc__penalty": ["l1", "l2"],
			"linear_svc__loss": ["squared_hinge"],
			"linear_svc__dual": [False],
			# "linear_svc__tol": [1e-4, 1e-5],
			# "linear_svc__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
			"linear_svc__C":[0.1, 1],
			# "linear_svc__multi_class": ["ovr", "crammer_singer"],
			# "vectorizer__stop_words": [None, stop_words.ENGLISH_STOP_WORDS],
			"vectorizer__ngram_range": [(1,2), (1,3)],
			"vectorizer__max_df": [0.9, 1.0],
			# "vectorizer__use_idf": [True, False]
		}
		# ,
		# {
		# 	# "linear_svc__penalty": ["l2"],		# if l1, you can't use hinge
		# 	"linear_svc__loss": ["hinge", "squared_hinge"],
		# 	# "linear_svc__dual": [True],	# if false, you can't use l2 or hinge
		# 	"linear_svc__tol": [1e-4, 1e-5],
		# 	"linear_svc__C":[0.1, 1, 10],
		# 	# "vectorizer__stop_words": [None, stop_words.ENGLISH_STOP_WORDS]
		# 	"vectorizer__ngram_range": [(1,1), (1,2), (1,3)],
		# 	"vectorizer__max_df": [0.9, 1.0]
		# 	# "vectorizer__use_idf": [True, False]
		# 	# "linear_svc__multi_class": ["ovr", "crammer_singer"]
		# }
	]


	scores = ["precision_micro", "recall_micro", "f1_micro", "accuracy", None]

	for score in scores:
		print("# Tuning hyper-parameters for {0}".format(score))
		print()

		grd = GridSearchCV(linear_svc_pipeline, param_grid=linear_svc_params, cv=k_fold, scoring=score)
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

	#create_model(x_train, y_train, x_test, y_test)


def create_model(x_train, y_train, x_test, y_test):
	""" Create a trained model using the best parameters. """

	print("\nCREATING FINAL MODEL...")

	vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.9)
	print("vectorizer params:", vectorizer.get_params())

	linear_svc = svm.LinearSVC(C=1.0, dual=True, loss="hinge", penalty="l2")
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

	print(model.score(x_test, y_test))
	print(model.get_params)

	print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
	print()

	print("Confusion matrix for final model:")
	print(confusion_matrix(y_true, y_pred))
	print()
	print()

def analyze_model(x_train, y_train, x_test, y_test):
	print("\nUNPICKLING MODEL...")
	list_unpickle = open("final_model/trained_linear_svc.pkl", "rb")
	model = pickle.load(list_unpickle)
	list_unpickle.close()

	print("Detaiiled classification report for final model:")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	y_true, y_pred = y_test, model.predict(x_test)

	print(model.score(x_test, y_test))
	print(model.get_params)

	print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
	print()

	print("Confusion matrix for final model:")
	print(confusion_matrix(y_true, y_pred))
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	print()
	print()

	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['Negative','Neutral', 'Positive'],
                      title='Confusion matrix, normalized')



#Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()





	# for x, prediction, y in zip(x_val, predictions, y_val):
		# if prediction != y:
			# print("Incorrect:", x, 'has been classified as', prediction, 'and should be ', y)
		# else:
		# if prediction == y:
			# print(prediction)
			# print("Correct:", x, "has been classified as", prediction, "and is correct!")
