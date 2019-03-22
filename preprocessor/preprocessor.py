"""
Script for preprocessing labelled tweets.

Input: csv file of with labelled tweets
Output: preprocessed and labelled tweets

Steps for preprocessing:
1. Basic Operations and Cleaning
    a. Remove URLs, hashtags, mentions
    b. Replace tabs and linebreaks with blanks and "" with ''
    c. Remove all punctuations except for ''
    d. Remove vowels repeated in sequence at least 3 times
    e. Replace sequences of "h" and "a" (e.g. "haha", "ahaha") with a "laugh" tag
    f. Convert emoticons to words
    g. Convert all text to lowercase
    h. Remove extra blank spaces
2. Implement stemming
3. Remove stopwords
"""

import pandas as pd
import re


def preprocess_tweet(tweet):

    # convert all text to lowercase
    tweet = tweet.lower()

    # replace URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)

    # replace #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)

    # replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)

    return tweet


def preprocess_df(tweets_df):
    # iterate through all of the tweet texts in the dataframe and preprocess them
    for index, row in tweets_df.iterrows():
        tweets_df.at[index, "text"] = preprocess_tweet(row["text"])


if __name__ == "__main__":

    tweets_df = pd.read_csv("../data/labelled-dataset/100-example-labelled-tweets.csv")
    print(tweets_df.head())
    preprocess_df(tweets_df)
    print(tweets_df.head())
