# -*- coding: utf-8 -*-

"""
python 2.7.13
"""

import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


"""
num = 0
for page in tweepy.Cursor(api.user_timeline, screen_name="honcharml").pages():
    print 'page n%d' % num
    for item in page:
        print item.id
    num += 1
api.update_status(
    status='found %d pages' % num
)
"""

"""
ids = []
for item in tweepy.Cursor(api.user_timeline, screen_name="honcharml").items(5):
    ids.append(item.id)
to_tweet = ','.join([str(item) for item in ids])[0:140]
api.update_status(status=to_tweet)
"""