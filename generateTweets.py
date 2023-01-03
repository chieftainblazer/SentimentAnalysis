# Description : This is a sentiment analysis program that parses the tweets fetched from Twitter
# using Python.

import os
import tweepy as tw
import pandas as pd

consumer_key = "svS2z7eqXtL1tVZSUOvgMeWCK"
consumer_secret = "QB3hTNMku9D0DVjh9JuMsz4DNBCiQjYLYKRezckxgVOX6d94ML"
access_token = "1027293645676769280-625c5mU7CmmvHwyuGtZNZAhB3WF9Hv"
access_token_secret = "ZeZ3mkclUhuBlg51WQaXCyvSEIw3enSPpLy7QFDhDYhXS"

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


