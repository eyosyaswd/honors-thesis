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
