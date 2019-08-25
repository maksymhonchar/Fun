# BOW - bag-of-words text representation
- only count how often each word appears in each text in the corpus.
- 3 steps:
    - Tokenization. Split each document into the words that appear in it (called tokens), for example by splitting them on whitespace and punctuation.
    - Vocabulary building. Collect a vocabulary of all words that appear in any of the documents, and number them (say, in alphabetical order).
    - Encoding. For each document, count how often each of the words in the vocabulary appear in this document.

- order of the words in the original string is completely irrelevant to the bag-of-words feature representation.

- sklearn: CountVectorizer

# Stopwords
- Fixed lists are mostly helpful for small datasets, which might not contain enough information for the model to determine which words are stopwords from the data itself.

# Others
-  4 kinds of string data you might see:
    - Categorical data
        - "red,” “green,” “blue,” “yellow,” ...
    - Free strings that can be semantically mapped to categories
        - users fill a text field with their own colors "reed", "midnight green" ...
    - Structured string data
        - Address, names of places or people, dates, telephone numbers...
    - Text data
        - phrases, sentences.

- Corpus = the dataset
- Document = Each data point

- IR - information retrieval
- NLP - natural language processing

- LatentDirichletAllocation - identify different topics