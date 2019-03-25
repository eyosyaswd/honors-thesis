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
	tweets = tweets_df["text"].tolist()
	polarities = tweets_df["polarity"].tolist()
	x_train, x_test, y_train, y_test = train_test_split(tweets, polarities, test_size=.20, random_state=42)
	return x_train, x_test, y_train, y_test


def main():
	tweets_df = pd.read_csv("data/labelled-dataset/100-example-labelled-tweets.csv")
	preprocessor.preprocess_df(tweets_df)
	# print(tweets_df["polarity"].tolist())
	x_train, x_test, y_train, y_test = split_data(tweets_df)
	classifier.train(x_train, y_train)

if __name__ == '__main__':
	main()
