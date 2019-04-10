"""
Main script that will conduct preprocessing, training, classification, and prediction.
"""


from sklearn.model_selection import train_test_split
import pandas as pd
import preprocessor
import trainer


def debug(some_str):
	#print(some_str)
	pass


def split_data(tweets_df):
	""" Splits the data set into training, testing, and validation data sets. """
	tweets = tweets_df["text"].tolist()
	polarities = tweets_df["polarity"].tolist()
	x_train, x_test, y_train, y_test = train_test_split(tweets, polarities, test_size=0.2, random_state=1)
	return x_train, x_test, y_train, y_test


def main():
	# read labelled csv and covert it to a pandas dataframe
	tweets_df = pd.read_csv("data-set/labelled-data-set/training-data-set/training-data-set.csv")

	# randomly remove 92.5% of neutral tweets and 31% of negative tweets
	tweets_df = tweets_df.drop(tweets_df.query('polarity == 0').sample(frac=0.925, random_state=1).index)
	tweets_df = tweets_df.drop(tweets_df.query('polarity == -1').sample(frac=0.31, random_state=1).index)

	# reindex dataframe and remove old index column
	tweets_df.reset_index(inplace=True)
	tweets_df = tweets_df.drop(columns=["index"])

	# print(tweets_df.shape)
	print(tweets_df.groupby('polarity').size())

	# conduct preprocessing
	preprocessor.preprocess_df(tweets_df)

	# split the dataset into training, testing, and validation data sets
	x_train, x_test, y_train, y_test = split_data(tweets_df)
	print("\nNumber of training data:", len(x_train),"\nNumber of testing data:", len(x_test))

	# create a classifier and train it using the dataset
	trainer.train(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
	main()
