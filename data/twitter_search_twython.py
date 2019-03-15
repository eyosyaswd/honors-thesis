# NOTE: Currently have decided to just use GetOldTweets3 because it's so much easier and opensource
""" Script to search tweets using Twython, a python Twitter API wrapper. """

from twython import Twython, TwythonRateLimitError, TwythonError
import json
import pandas as pd

# Load credentials from json file
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)

# Instantiate Twython object
twitter = Twython(credentials["CONSUMER_KEY"], credentials["CONSUMER_SECRET"])

# Create the query (parameters: https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets)
query = {"q": "Real Madrid", \
         "lang": "en", \
         "result_type": "recent", \
         "count": 100, \
         "until": "2019-03-06", \
         "since_id": "1103022384892780546", \
         "max_id": "1103051551273304065", \
        }

# Create dictionary to hold tweets
tweets_dict = {"date": [], "id": [], "text": [], "retweet_count": [], "favorite_count": []}

# Search for tweets
try:
    for status in twitter.search(**query)["statuses"]:
        tweets_dict["date"].append(status["created_at"])
        tweets_dict["id"].append(status["id"])
        tweets_dict["text"].append(status["text"])
        tweets_dict["retweet_count"].append(status["retweet_count"])
        tweets_dict["favorite_count"].append(status["favorite_count"])
except TwythonRateLimitError as e:
    print(e)
except TwythonError as e:
    print(e)

# Convert dictionary to pandas DataFrame
tweets_df = pd.DataFrame(tweets_dict)

# Sort tweets by date and time tweeted
tweets_df.sort_values(by="id", inplace=True)

# Output tweets into a csv
tweets_df.to_csv("LEVvsRM-halftime-tweets.csv")
