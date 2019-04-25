"""
Script that classifies new data.
"""

import pandas as pd
import pickle
import preprocessor



def classify(model, tweets_df, original_tweets):
	print("\nCLASSIFYING...")
	tweets = tweets_df["text"].tolist()
	tweets_df["polarity"] = model.predict(tweets)
	tweets_df["text"] = original_tweets
	tweets_df.to_csv("data-set/labelled-data-set/analysis-data-set/analysis-data-set-labelled.csv", index=False)


def unpickle_model():
	print("\nUNPICKLING MODEL...")
	list_unpickle = open("final_model/trained_linear_svc.pkl", "rb")
	model = pickle.load(list_unpickle)
	list_unpickle.close()
	return model


def main():
	# read csv and covert it to a pandas dataframe
	tweets_df = pd.read_csv("data-set/raw-data-set/analysis-data-set/analysis-data-set-raw-pre.csv")

	original_tweets = tweets_df["text"].tolist()

	# conduct preprocessing
	print("\nPREPROCESSING...")
	preprocessor.preprocess_df(tweets_df)

	# unpickle model
	model = unpickle_model()

	classify(model, tweets_df, original_tweets)





if __name__ == '__main__':
	main()
