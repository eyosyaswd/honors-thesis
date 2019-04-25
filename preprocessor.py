"""
Script for preprocessing labelled tweets.

Input: csv file of with labelled tweets
Output: preprocessed and labelled tweets
"""

# from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re

USE_STEMMER = True


def is_valid_word(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)
    # return word


def preprocess_word(word):
    # remove punctuations
    word = word.strip('\'"?!,.():;')

    # convert more than 2 letter repetitions to 2 letters
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)

    # remove - & '
    word = re.sub(r'(-|\')', '', word)

    # replace negative constructs with "not"
    word = re.sub(r'(cant|dont|isnt|wont|hasnt|arent|aint|never)', 'not', word)

    return word

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    # print(tweet)

    # convert all text to lowercase
    tweet = tweet.lower()

    # replace URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)

    # replace #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r'\1', tweet)

    # replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_HANDLE', tweet)

    # strip away space, \, ', and "
    tweet = tweet.strip(' \'"')

    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)

    # replace emojis with EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)

    # replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)

    tweet_as_list = tweet.split()
    preprocessed_tweet = []
    for word in tweet_as_list:
        word = preprocess_word(word)
        if is_valid_word(word):
            if USE_STEMMER:
                # word = str(SnowballStemmer("english").stem(word))
                word = str(PorterStemmer().stem(word))
            preprocessed_tweet.append(word)

    tweet = " ".join(preprocessed_tweet)

    # print(tweet, "\n")
    return tweet


def preprocess_df(tweets_df):
    # iterate through all of the tweet texts in the dataframe and preprocess them
    for index, row in tweets_df.iterrows():
        tweets_df.at[index, "text"] = preprocess_tweet(row["text"])


if __name__ == "__main__":
    # read labelled csv and covert it to a pandas dataframe
	tweets_df = pd.read_csv("testing_preprocessing.csv")

    # conduct preprocessing
	preprocess_df(tweets_df)
