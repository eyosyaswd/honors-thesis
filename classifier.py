"""
Script that classifies new data.
"""

import pickle
import preprocessor



def classify():
	pass


def unpickle_model():
	print("\nUNPICKLING MODEL...")
	list_unpickle = open("final_model/trained_linear_svc.pkl", "rb")
	model = pickle.load(list_unpickle)
	list_unpickle.close()
	return model


def main():
	# read csv and covert it to a pandas dataframe
	tweets_df = pd.read_csv("data-set/raw-data-set/analysis-data-set/analysis-data-set-raw.csv")

	# conduct preprocessing
	preprocessor.preprocess_df(tweets_df)

	# unpickle model
	model = unpickle_model()

	classify(tweets_df, model)





if __name__ == '__main__':
	main()


	# for x, prediction, y in zip(x_val, predictions, y_val):
		# if prediction != y:
			# print("Incorrect:", x, 'has been classified as', prediction, 'and should be ', y)
		# else:
		# if prediction == y:
			# print(prediction)
			# print("Correct:", x, "has been classified as", prediction, "and is correct!")
