# Description : This is a sentiment analysis program that parses the tweets fetched from Twitter
# using Python.

import os
import tweepy as tw
import pandas as pd

#These keys have been removed for security purposes.
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = "#Juventus" + " -filter:retweets"
date_since = "2022-06-01"

# Collect tweets
tweets = tw.Cursor(api.search_tweets, tweet_mode='extended', q=search_words, lang="en").items(1000)

# Iterate and print tweets
for tweet in tweets:
    print(tweet.full_text + "\n")


