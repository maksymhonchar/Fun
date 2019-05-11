# We discard most of the structure of the input text, like chapters, paragraphs,
# sentences, and formatting, and only count how often each word appears in each text in the corpus.

# Discarding the structure and counting only word occurrences leads to the
# mental image of representing text as a “bag.”

# the order of the words in the original string is completely irrelevant to
# the bag-of-words feature representation.

# 3 steps:
# 1. Tokenization. Split each document into the words that appear in it (called tokens),
# for example by splitting them on whitespace and punctuation.
# 2. Vocabulary building. Collect a vocabulary of all words that appear in any of the
# documents, and number them (say, in alphabetical order).
# 3. Encoding. For each document, count how often each of the words in the vocabulary
# appear in this document.

# sklearn - CountVectorizer for bag-of-words representation.

from sklearn.feature_extraction.text import CountVectorizer


bards_words = [
    "The fool doth think he is wise,",
    "but the wise man knows himself to be a fool"
]

vect = CountVectorizer()
vect.fit(bards_words)  # tokenize; build vocabulary;

print('Vocabulary size: {0}'.format(len(vect.vocabulary_)))  # 13 (13 words)
print('Vocabulary content: {0}'.format(vect.vocabulary_))  # 

# Create the bag-of-words representation for the training data
bag_of_words = vect.transform(bards_words)
print('bag of words: {0}'.format(repr(bag_of_words)))
# <2x13 sparse matrix of type '<class 'numpy.int64'>'
#   with 16 stored elements in Compressed Sparse Row format>


# Don't store zeros - this is a waste of memory.
print('Dense representation of bag_of_words:\n{0}'.format(bag_of_words.toarray()))  # list of 0 or 1

