#%% [markdown]
# src: https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/

#%% Load libraries
from html import unescape

import re

import itertools

appos = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
}

#%% Perform preprocessing
original_tweet = "I luv my &lt;3 \" iphone &amp; you're awsm apple. DisplayIsAwesome, sooo happppppy 🙂 http://www.apple.com"

# Escaping HTML characters.
# Get rid of HTML entities liek &lt; or &amp;
# Approach 1: remove using specific regular expressions.
# Use HTMLParser to convert these entities to text. "&lt"->"<"
htmlparsed_tweet = unescape(original_tweet)

# Decodinig data.
# Transform information from complex symbols to simple and easier to understand characters.
# It is necessary to keep the complete data in standard encoding format.
# decoded_tweet = htmlparsed_tweet.decode('utf8').encode('ascii', 'ignore')  # skip this in Python3

# Apostrophe lookup
# To avoid any word sense disambiguation in text, maintain proper structure in it and abide by the rules of context free grammar - convert apostrophes into standard lexicons.
tweet_words = htmlparsed_tweet.split()
tweet_words_no_appos = [
    appos[word] if word in appos
    else word
    for word in tweet_words
]
tweet = " ".join(tweet_words_no_appos)

# Remove stop-words

# Remove punctuations

# Remove expressions
# Textual data (usually speech transcripts) may contain human expressions like [laughing], [Crying], [Audience paused]. These expressions are usually non relevant to content of the speech and hence need to be removed. Simple regular expression can be useful in this case.

# Split attached words
# DisplayIsAwesome -> Display Is Awesome
tweet = " ".join(re.findall('[A-Z][^A-Z]*', tweet))

# Slangs lookup
# awsm -> awesome

# Standardizing words
# "I looooooveeeee you" -> "I love you"
tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

# Remove URLs

# Advanced data cleaning:
# Grammar checking: Grammar checking is majorly learning based, huge amount of proper text data is learned and models are created for the purpose of grammar correction. There are many online tools that are available for grammar correction purposes.
# Spelling correction: In natural language, misspelled errors are encountered. Companies like Google and Microsoft have achieved a decent accuracy level in automated spell correction. One can use algorithms like the Levenshtein Distances, Dictionary Lookup etc. or other modules and packages to fix these errors.
