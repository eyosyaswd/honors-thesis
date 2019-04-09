"""
Main script that will conduct preprocessing, training, classification, and prediction.
"""


from sklearn.model_selection import train_test_split
import pandas as pd
import preprocessor
import classifier


def debug(some_str):
	#print(some_str)
	pass


def split_data(tweets_df):
	""" Splits the data set into training, testing, and validation data sets. """
	tweets = tweets_df["text"].tolist()
	polarities = tweets_df["polarity"].tolist()
	x_train, x_test, y_train, y_test = train_test_split(tweets, polarities, test_size=.20, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
	return x_train, x_test, x_val, y_train, y_test, y_val


def main():
	# read labelled csv and covert it to a pandas dataframe
	tweets_df = pd.read_csv("data-set/labelled-data-set/training-data-set/training-data-set.csv")

	# conduct preprocessing
	preprocessor.preprocess_df(tweets_df)

	# split the dataset into training, testing, and validation data sets
	x_train, x_test, x_val, y_train, y_test, y_val = split_data(tweets_df)

	# create a classifier and train it using the dataset
	classifier.train(x_train, y_train, x_val, y_val, x_test, y_test)

if __name__ == '__main__':
	main()
