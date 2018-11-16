import urllib.request as urllib2
import http.cookiejar as cookielib
import urllib.parse as urllib
import random


# The biggest problem is, that I don't know how to pass a [select] item -_-
def post_junk(index, to_feed):
    url = "https://progbase.herokuapp.com/profile/update"
    params = {
        'method': 'POST',
        'text': random.choice(to_feed['text']),
        'email': random.choice(to_feed['email']),
        'number': random.choice(to_feed['number'])
    }
    data_to_post = urllib.urlencode(params)
    binary_data_to_post = data_to_post.encode('utf-8')
    print('{0}) Starting to get a response...'.format(index + 1))

    opener = urllib2.build_opener(
        urllib2.HTTPCookieProcessor(cookielib.CookieJar()),
        urllib2.HTTPRedirectHandler()
    )
    response = opener.open(
        url,
        binary_data_to_post
    )
    print("Done!")
    return response

for i in range(10000):
    # Data to pass as a params fields:
    to_feed = {
        'text': ['привет', 'как дела', 'я люблю мороженое', 'hi', 'how are you doing?', 'i like ice cream'],
        'email': ['я_люблю_валидацию', 'я ненавижу валидацию', 'i like validation', 'i hate validation'],
        'number': ['это число', 'и это тоже число', 'this is a number', 'this is a number too', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    # For testing purposes use this:
    # print(response.read().decode("utf-8"))
    response = post_junk(i, to_feed)
