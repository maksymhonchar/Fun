`# -*- coding: utf-8 -*-
import tweepy
import pprint


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

"""Main twitter instance"""
api = tweepy.API(auth)

"""Me myself and I"""
me = api.me()


def user_timeline():
    """Returns: array of statuses"""
    to_return = []
    statuses = api.user_timeline(
        id=me.id
    )
    for status in statuses:
        to_return.append(
            {
                'created_at' : status.created_at,
                'text': status.text,
                'author': status.author.screen_name
            }
        )
    return to_return

def user_search():
    results = api.search_users(
        q="Maxim Honchar"
    )
    to_return = []
    for user in results:
        to_return.append(
            {
                'id': user.id,
                'screen_name': user.screen_name,
                'lang': user.lang,
                'friends_count': user.friends_count,
                'location': user.location,
                'followers_count': user.followers_count,
            }
        )
    return to_return


def tweet_rate(query_term):
    from datetime import datetime

    statuses = api.search(
        q=query_term
    )  # array of statuses

    print(statuses[0].created_at)

    # print(len(statuses))
    # print(statuses[0].id, statuses[-1].id) # they are real, ok

    first_timestamp = statuses[0].created_at
    last_timestamp = statuses[-1].created_at
    total_dt = (first_timestamp - last_timestamp).total_seconds()
    mean_dt = total_dt / len(statuses)

    print "Average tweeting rate for '%s'" % query_term
    print "between %s and %s: %.3fs" % (statuses[-1].created_at, statuses[0].created_at, mean_dt)


def available_trends():
    print api.trends_available()

def trends():
    import json
    from time import sleep

    print "Kiev trends"
    ua_woeid = 23424976
    kiev_woeid = 924938

    kiev_trends = api.trends_place(
        id=kiev_woeid
    )[0][u'trends']

    for trend in kiev_trends:
        print trend[u'query'].decode("utf-8")

    print "###"

    for trend in kiev_trends:
        if len(trend[u'query']) >= 140:
            continue
        api.update_status(
            status=trend[u'query']
        )
        sleep(0.5)
        print('done', trend[u'query'])
