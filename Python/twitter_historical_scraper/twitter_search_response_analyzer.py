#%% [markdown]
# todo: place everything to django modules
# todo: interface for certain user
# todo: max 1000 for each rday - ?
# todo: GH commit & push
# todo: update DB: add tweet_id field
# todo: save to DB certain data + interface for it.

#%% Load libraries
import json
import random
import time
import urllib
from datetime import datetime

import bs4
import requests

#%% Define constants for Twitter search
TWITTER_SEARCH_URL = 'https://twitter.com/search?q={search_query}&src=typd&qf=off&l=en'
TWITTER_SEARCH_MORE_URL = 'https://twitter.com/i/search/timeline?q={search_query}&src=typd&vertical=default&include_available_features=1&include_entities=1&max_position={max_position}&qf=off&l=en'

#%% Define user agent for requests.
user_agent_pool = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0',
    'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393'
]
USER_AGENT = random.choice(user_agent_pool)

#%% Define timeouts pool as a try to overcome bot detection.
timeout_pool_s = [1, 2, 3, 4, 5]

#%% Define class to represent single tweet instance
class Tweet(object):

    def __init__(self, **params):
        self.tweet_id = params['tweet_id']
        self.content = params['tweet_content']
        self.likes = params['favorite_cnt']
        self.retweets = params['retweet_cnt']
        self.replies = params['reply_cnt']
        self.link = params['tweet_link']
        self.date = params['timestamp']

    def jsonify(self):
        tweet_dict = {
            'id': self.tweet_id,
            'content': self.content,
            'likes': self.likes,
            'retweets': self.retweets,
            'replies': self.replies,
            'link': self.link,
            'date': self.date
        }
        return tweet_dict

#%% Define class to represent twitter parser
class TwitterSearchPageParser(object):

    def parse_tweets_timeline(self, timeline_html):
        """Parse tweets from TWITTER_SEARCH_URL or TWITTER_SEARCH_MORE_URL response"""
        tweets = []
        soup = bs4.BeautifulSoup(timeline_html)
        for tweet_tag in soup.find_all("div", class_="tweet"):
            # Find tweet ID.
            tweet_id = tweet_tag['data-tweet-id']
            # Find content of the tweet.
            tweet_content = tweet_tag.find('p', class_='tweet-text').text
            # Find emojis
            emojis_tags = tweet_tag.find('p', class_='tweet-text').find_all(class_='Emoji')
            tweet_content += " " + " ".join(emoji_tag['alt'] for emoji_tag in emojis_tags)
            # Find favorites, likes and replies cnt.
            tweet_tag_footer_div = tweet_tag.find('div', class_='stream-item-footer')
            favorite_cnt = self.find_tweet_cnt_stats(tweet_tag_footer_div, 'favorite')
            retweet_cnt = self.find_tweet_cnt_stats(tweet_tag_footer_div, 'retweet')
            reply_cnt = self.find_tweet_cnt_stats(tweet_tag_footer_div, 'reply')
            # Find tweet link.
            tweet_link = 'twitter.com{0}'.format(tweet_tag['data-permalink-path'])
            # Find tweet posting timestamp.
            timestamp_unixlike = tweet_tag.find('span', class_='_timestamp')['data-time']
            timestamp = datetime.utcfromtimestamp(int(timestamp_unixlike)).strftime('%Y-%m-%d %H:%M:%S')
            # Create Tweet class instance to save.
            tweet_instance = Tweet(tweet_id=tweet_id, tweet_content=tweet_content,
                                   favorite_cnt=favorite_cnt, retweet_cnt=retweet_cnt, reply_cnt=reply_cnt,
                                   tweet_link=tweet_link,
                                   timestamp=timestamp)
            # Save Tweet class instance.
            tweets.append(tweet_instance)
        return tweets

    @staticmethod
    def find_tweet_cnt_stats(tweet_footer_html, stats_type):
        if stats_type not in ['reply', 'retweet', 'favorite']:
            raise ValueError('Incorrect stats_type value')
        stats_span = tweet_footer_html.find(
            'span', class_='ProfileTweet-action--{0}'.format(stats_type))
        stats_span_value = stats_span.find(
            'span', class_='ProfileTweet-actionCount')['data-tweet-stat-count']
        return stats_span_value

search_page_parser = TwitterSearchPageParser()

#%% Request search data from Twitter: 1st page.
search_query = '#bitcoin since:2019-05-28 until:2019-05-29'
search_query = urllib.parse.quote(search_query)

search_url = TWITTER_SEARCH_URL.format(search_query=search_query)

response = requests.get(search_url, headers={'User-agent': USER_AGENT})
response_text = response.text

parsed_tweets = []

first_page_parsed_tweets = search_page_parser.parse_tweets_timeline(response_text)

parsed_tweets.extend(first_page_parsed_tweets)

def find_arg_value(html, value):
    start_pos = html.find(value) + len(value)
    start_pos += 2  # skip = and " characters.
    end_pos = html.find('"', start_pos)
    return html[start_pos:end_pos]

next_position = find_arg_value(response_text, "data-max-position")

#%% Request search data from Twitter: 2-N pages.
has_more_items = True if next_position else False
default_fri_value = None
old_next_position = False

while has_more_items:
    adv_search_url = TWITTER_SEARCH_MORE_URL.format(search_query=search_query, max_position=next_position)

    USER_AGENT = random.choice(user_agent_pool)

    print('Trying to get response from requests.get...')
    response = requests.get(adv_search_url, headers={'User-agent': USER_AGENT})
    response_text = response.text
    response_dict = json.loads(response_text)

    nth_page_parsed_tweets = search_page_parser.parse_tweets_timeline(response_dict['items_html'])
    parsed_tweets.extend(nth_page_parsed_tweets)

    focused_refresh_interval = response_dict['focused_refresh_interval']
    if not default_fri_value:
        default_fri_value = focused_refresh_interval

    next_position = response_dict['min_position']
    # has_more_items = response_dict.get('has_more_items', False)
    has_more_items = old_next_position != next_position
    print('To continue? (stop when old=next) {0}'.format(old_next_position != next_position))

    print('fri is {0}'.format(focused_refresh_interval))
    print('tweets gathered from 1-N pages: {0}'.format(len(parsed_tweets)))

    if not has_more_items:
        break

    old_next_position = next_position
    
    sleep_time = random.choice(timeout_pool_s)
    print('waiting for {0}s\n'.format(sleep_time))
    time.sleep(sleep_time)
