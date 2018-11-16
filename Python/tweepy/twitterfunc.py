# -*- coding: utf-8 -*-

import tweepy
from pprint import pprint
from time import sleep

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

"""
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text
"""

"""
status = api.get_status(918230077757894657)
print(status)
"""

"""
res = api.update_status(status='update from twitterfunc module')
"""

"""
res = api.retweet(id=918230077757894657)
"""

"""
get twitter user by id link:
https://twitter.com/intent/user?user_id=216939636

testing_user = api.get_user(
    screen_name='testing'
)
testing_timeline = api.user_timeline(
    id=testing_user.id,
)
for status in testing_timeline:
    api.retweet(id=status.id)
    sleep(0.5)
    print('done', status.id)
"""

"""
10 Pavel Durov retweets
# api.update_status(status='retweeting 10 tweets from Durov')
print(durov_user.id)
durov_retweets = api.retweets(id=durov_user.id)
for retweet in durov_retweets:
    api.retweet(id=retweet.id)
    sleep(0.5)
    print('done', retweet.id)
"""
#durov_user = api.get_user(screen_name='durov')

"""
durov_retweeters = api.retweeters(
    id=924622080355782656,
    count=5,
)
users = []
for id in durov_retweeters:
    cur_user = api.get_user(id=id)
    users.append(cur_user.screen_name)
api.update_status(
    status=','.join(users)
)
"""

"""
api.update_status(
    status="russki текст"
)
"""

"""
me = api.me()
my_timeline = api.user_timeline(
    id=me.id
)
for status in my_timeline:
    source = unicode("во-первых", 'utf-8')
    if source in status.text:
        api.update_status(
            status="что ты мне сделаешь",
            in_reply_to_status_id=status.id
        )
        break
"""

"""
nadia = api.get_user(
    screen_name="nadine__is"
)
nadia = api.create_friendship(
    id=nadia.id,
    follow=True
)
print nadia
"""

"""
api.destroy_friendship(
    id=api.get_user(screen_name="nadine__is").id,
)
"""

print api.rate_limit_status()
