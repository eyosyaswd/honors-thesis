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

from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import re

USE_STEMMER = False


def is_valid_word(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def preprocess_word(word):
    # remove punctuations
    word = word.strip('\'"?!,.():;')
    # convert more than 2 letter repetitions to 2 letters
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # remove - & '
    #word = re.sub(r'(-|\')', '', word)
    return word


def preprocess_tweet(tweet):
    # replace URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', 'URL', tweet)

    # replace #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r'\1', tweet)

    # replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)

    # convert all text to lowercase
    tweet = tweet.lower()

    # replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    #
    tweet = tweet.strip('\'"')

    tweet_as_list = tweet.split()
    preprocessed_tweet = []
    for word in tweet_as_list:
        word = preprocess_word(word)
        if is_valid_word(word):
            if USE_STEMMER:
                word = str(SnowballStemmer("english").stem(word))
            preprocessed_tweet.append(word)

    tweet = " ".join(preprocessed_tweet)

    return tweet


def preprocess_df(tweets_df):
    # iterate through all of the tweet texts in the dataframe and preprocess them
    for index, row in tweets_df.iterrows():
        tweets_df.at[index, "text"] = preprocess_tweet(row["text"])


if __name__ == "__main__":
    pass

    # tweets_df = pd.read_csv("data/labelled-dataset/100-example-labelled-tweets.csv")
    # print(tweets_df.head(10))
    # preprocess_df(tweets_df)
    # print(tweets_df.head(10))
    # tweets_df.to_csv("data/labelled-dataset/preprocessed-tweets.csv", index=False)


# NOTES:
# r'\]\n' == '\\]\\n
