"""
File for analyzing the data we classified.
"""

import pandas as pd


def main():
	# read csv and covert it to a pandas dataframe
    tweets_df = pd.read_csv("data-set/labelled-data-set/analysis-data-set/analysis-data-set-labelled.csv")
    print(tweets_df.shape)
    print(tweets_df.groupby('polarity').size())



if __name__ == '__main__':
	main()
