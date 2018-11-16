# -*- coding: utf-8 -*-

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

import json

"""
NOTE: 
401 error can be because of incorrect clock setup
402 error increases timeout exponentially - never forget to close the connection after it appears
"""

class Listener(StreamListener):
    def on_data(self, data):
        print(data)
        return True

    def on_status(self, status):
        print('status received: ', status.text)

    def on_error(self, status):
        if status == 420:
            return False
        print('Error: %r' % status)
        

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth_handler=auth)

l = Listener(api=api)
twitterStream = Stream(auth, l)
twitterStream.filter(track=["car"], async=True)
