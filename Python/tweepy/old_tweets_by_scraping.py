from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

import json
import datetime
from time import sleep


class OldIdsScrapper(object):

    def __init__(self, chrome_driver_path):
        self.delay = 0.1  # wait on each page load before reading the page
        self.driver = webdriver.Chrome(chrome_driver_path)

    def query(self, query, start, end, lang):
        """Returns found_ids"""
        id_selector = '.time a.tweet-timestamp'
        tweet_selector = 'li.js-stream-item'

        query = query.lower()
        days = (end - start).days + 1
        found_ids = []

        for day in range(days):
            d1 = self.format_day(self.increment_day(start, 0))
            d2 = self.format_day(self.increment_day(start, 1))
            url = self.form_url(lang, query, d1, d2)
            print(url)
            print(d1)
            self.driver.get(url)
            sleep(self.delay)

            try:
                found_tweets = self.driver.find_elements_by_css_selector(tweet_selector)
                increment = 10

                while len(found_tweets) >= increment:
                    print('scrolling down to load more tweets')
                    self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                    sleep(self.delay)
                    found_tweets = self.driver.find_elements_by_css_selector(tweet_selector)
                    increment += 10

                print('{} tweets found, {} total'.format(len(found_tweets), len(found_ids)))

                for tweet in found_tweets:
                    try:
                        status_id = tweet.find_element_by_css_selector(id_selector).get_attribute('href').split('/')[-1]
                        found_ids.append(status_id)
                    except StaleElementReferenceException:
                        print('lost element reference', tweet)  # !!!

            except NoSuchElementException:
                print('no tweets on this day')

            start = self.increment_day(start, 1)

        print 'closing driver'
        self.driver.close()

        return found_ids

    @staticmethod
    def format_day(date):
        """Twitter query friendly date"""
        day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
        month = '0' + str(date.month) if len(str(date.month)) == 1 else str(date.month)
        year = str(date.year)
        return '-'.join([year, month, day])

    @staticmethod
    def form_url(lang, query, since, until):
        p1 = 'https://twitter.com/search?f=tweets&vertical=default'
        p2 = '&l=' + lang + '&q=' + query + '%20since%3A' + since + '%20until%3A' + until
        p3 = '%20include%3Aretweets' + '&src=typd'  # todo: include retweets with or without %20 ?
        return p1 + p2 + p3

    @staticmethod
    def increment_day(date, i):
        return date + datetime.timedelta(days=i)

    @staticmethod
    def write_ids(ids, filename='all_ids.json'):
        try:
            with open(filename) as f:
                all_ids = ids + json.load(f)
                data_to_write = list(set(all_ids))
                print('tweets found on this scrape: ', len(ids))
                print('total tweet count: ', len(data_to_write))
        except EnvironmentError:
            all_ids = ids
            data_to_write = list(set(all_ids))
            print('tweets found on this scrape: ', len(ids))
            print('total tweet count: ', len(data_to_write))

        with open(filename, 'w') as outfile:
            json.dump(data_to_write, outfile)


def testing_oldidsscrapper():
    chrome_driver_path = "C:/Users/Max/Desktop/chromedriver.exe"
    query = 'kyiv'
    lang = 'ja'
    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2017, 1, 10)
    scrapper = OldIdsScrapper(chrome_driver_path)

    import tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    print 'retweeting japanese content from %s to %s' % (start, end)

    api.update_status(
        status='RT all JA content related to __kyiv__ query from %s to %s' % (start, end)
    )

    ids = scrapper.query(
        query=query,
        start=start,
        end=end,
        lang=lang
    )

    for status_id in ids:
        api.retweet(
            id=status_id
        )
        print 'retweeted %r status' % status_id


testing_oldidsscrapper()
