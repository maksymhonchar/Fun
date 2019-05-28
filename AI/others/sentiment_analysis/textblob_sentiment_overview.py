#%% Import libraries
from textblob import TextBlob

#%% Define "Feedback" examples & check their sentiment 
feedback_text = [
    "The food at restaurant was awesome",
    "The food at restaurant was very good"
]

blob1 = TextBlob(feedback_text[0])
blob2 = TextBlob(feedback_text[1])

print(blob1.sentiment)  # Sentiment(polarity=1.0, subjectivity=1.0) 
print(blob2.sentiment)  # Sentiment(polarity=0.9099999999999999, subjectivity=0.7800000000000001)

#%% Example from https://textblob.readthedocs.io/en/dev/
btc_tweet_text = '''
Updates.
Total Market Cap is up 12% in 2 days.
#Bitcoin is taking a breather at 6950 Resistance.
May attempt 7100/7300 on topside but resistance looks heavy.
Frontline ALTs are posting best daily gains in a month.
Today & tomo daily closing is key to gauge BTC paired ALTs Bottom.
'''

btc_blob = TextBlob(btc_tweet_text)
print(btc_blob.tags)
print(btc_blob.noun_phrases)

# The polarity score is a float within the range [-1.0, 1.0]. [negative, positive]
# The subjectivity is a float within the range [0.0, 1.0]. [0.0 is very objective and 1.0 is very subjective].

for sentence in btc_blob.sentences:
    print(sentence, sentence.sentiment.polarity)
# 0.0
# 0.0
# 0.0
# -0.2
# 0.5
# 0.0

print(btc_blob.translate(to='ru'))

#%% Example from https://medium.com/@imnikhilanand/quick-guide-to-textblob-fbaf9b65c729
attack_text = """
A drone attack that failed to kill President Nicolás Maduro of Venezuela unfolded on live TV and in front of many witnesses
"""

attack_blob = TextBlob(attack_text)

print(attack_blob.noun_phrases)
print(attack_blob.words)

# toNote: pluralize & singularize!
print(attack_blob.words.singularize())
print(attack_blob.words.pluralize())

print(attack_blob.word_counts['of'])

print(attack_blob.ngrams(n=2))
print(attack_blob.ngrams(n=4))

from textblob import Word
for word in attack_blob.words:
    print(Word(word).correct() == word)

#%% Example from https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
av_blob = TextBlob("Analytics Vidhya is a great platform to learn data science. \n It helps community through blogs, hackathons, discussions,etc.")
print(av_blob.tokenize())
print(av_blob.sentences, av_blob.sentences[0])

for phrase in av_blob.noun_phrases:
    print(phrase)  # analytics vidhya; great platform; data science

# toNote: part-of-speech tagging
for words, tag in av_blob.tags:
    print(words, tag)

# inflection - process of word formation in which characters are added to the base form of a word to express grammatical meanings.
# words inflection and lemmatization
print(av_blob.sentences[1].words[1].singularize())  # helps -> help

# pluralize
w = Word('Platform')
print(w.pluralize())  # 'Platforms'

# Use tags to inflect a particular type of words
for word, tag in av_blob.tags:
    if tag == 'NN':
        print(word.pluralize())  # platforms, sciences, communities

# lemmatization
w = Word('running')
print(w.lemmatize('v'))  # 'v' represents verb
# result: 'run'

# n-grams: a combination of multiple words. n-grams can be used as features for language modelling.
for ngram in av_blob.ngrams(4):
    print(ngram)  # ['Analytics', 'Vidhya', 'is', 'a'], ['Vidhya', 'is', 'a', 'great'], ...

# sentiment analysis
# SA is the prorcess of determining the attitude or the emotion of the writer - whether it is positive, negative or neutral.
# the sentiment function returns two properties: polarity and subjectivity
# polarity: float [-1,1] where 1 means positive statement and -1 means negative statement.
# subjectivity: float [0,1] subjective sentences generally refer to personal opinion, emotion or judgement whereas objective refers to factual information.
print(av_blob.sentiment, sep='\t')  # Sentiment(polarity=0.8, subjectivity=0.75)
# 0.8 means the statement is positive; 0.75 subjectivity refers that mostly it is a public opinion and not a factual information.

# spelling correction.
mistake_av_blob = TextBlob('Analytics Vidhya is a gret platfrm to learn data scence')
print(mistake_av_blob.correct())  # Analytics Vidhya is a great platform to learn data science
# Check the list of suggested word and its confidence.
print(mistake_av_blob.words[4].spellcheck())  # suggested words for 'great'
# [('great', 0.5351351351351351), ... ]

# different languages
ua_blob = TextBlob('Усім доброго дня!')
print(ua_blob.detect_language())  # uk
print(ua_blob.translate(from_lang='uk', to='en'))  # Good afternoon! (correct translation was "Good afternoon everyone")

# toNote: classifier
training_set = [
    ('Tom Holland is a bad spiderman.', 'pos'),
    ('a awful Javert (Russell Crowe) ruined Les Miserables for me...', 'pos'),
    ('The Dark Knight Rises is the greatest superhero movie ever!', 'neg'),
    ('Fantastic Four should have never been made.', 'pos'),
    ('Wes Anderson is my favorite director!', 'neg'),
    ('Captain America 2 is pretty awesome.', 'neg'),
    ('Lets pretend "Batman and Robin" never happened..', 'pos'),
]
testing_set = [
    ('Superman was never an interesting character.', 'pos'),
    ('Fantastic Mr Fox is an awesome film!', 'neg'),
    ('Dragonball Evolution is simply terrible!!', 'pos')
]
from textblob import classifiers as tbclassifiers
nb_clf = tbclassifiers.NaiveBayesClassifier(training_set)
print(nb_clf.accuracy(testing_set))  # 1.0
nb_clf.show_informative_features(3)
test_clf_blob = TextBlob('the weather is super!', classifier=nb_clf)
print(test_clf_blob.classify())  # neg

# decision tree classifier is also available
dt_clf = tbclassifiers.DecisionTreeClassifier(training_set)
print(dt_clf.accuracy(testing_set))  # 0.(6)

#%% [markdown]
"""
Pros and Cons  
Pros:  
Since, it is built on the shoulders of NLTK and Pattern, therefore making it simple for beginners by providing an intuitive interface to NLTK.  
It provides language translation and detection which is powered by Google Translate ( not provided with Spacy).  
Cons:  
It is little slower in the comparison to spacy but faster than NLTK. (Spacy > TextBlob > NLTK)  
It does not provide features like dependency parsing, word vectors etc. which is provided by spacy.  
"""