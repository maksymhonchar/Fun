# Clean the data

def clean_review(raw_text, to_lower, remove_numbers, remove_stopwords):
    # 1
    text_nohtml = BeautifulSoup(raw_text).get_text()
    # 2
    if remove_numbers:
        re_clean_pattern = "[^a-zA-Z]"
    else:
        re_clean_pattern = "[^a-zA-Z0-9]"
    text_regexclean = re.sub(re_clean_pattern, " ", text_nohtml)
    # 3
    if to_lower:
        text_tokens = text_regexclean.lower().split()
    else:
        text_tokens = text_regexclean.split()
    # 4
    if remove_stopwords:
        nltk_stopwords = set(stopwords.words("english"))
        text_tokens_nostopwords = [
            token for token in text_tokens
            if token not in nltk_stopwords
        ]
        text_tokens = text_tokens_nostopwords
    # 5
    text_cleaned = " ".join(text_tokens)
    return text_cleaned    

# Apply data cleaning to datasets

to_lower = True
remove_numbers = True
remove_stopwords = True

train_df['cleaned_review'] = train_df['review'].apply(
    lambda x: clean_review(
        x, to_lower=to_lower, remove_numbers=remove_numbers, remove_stopwords=remove_stopwords
    )
)

train_df['cln_rev_len'] = train_df['cleaned_review'].apply(
    lambda x: len(' '.join(x))
)

train_df['cln_rev_tokens_len'] = train_df['cleaned_review'].apply(len)

# Vectorize reviews using Bag-of-words algorithm

bow_vectorizer = CountVectorizer()

bow_vectorizer.fit(train_df['cleaned_review'])

display(bow_vectorizer)

display(len(bow_vectorizer.vocabulary_))  # 74047

# Try out bow vectorizer

bow_transfrmd_review = bow_vectorizer.transform(
    train_df['cleaned_review']
)

display(bow_transfrmd_review.shape)  # (25000, 74047)

nonzero_occurences = bow_transfrmd_review.nnz
display(nonzero_occurences)  # 2443820

sparsity = (100 * bow_transfrmd_review.nnz) /            (bow_transfrmd_review.shape[0] * bow_transfrmd_review.shape[1])
display(sparsity)  # 0.13201453131119426

# Transform sparse matrix to tf-idf representation

tfidf_transformer = TfidfTransformer()

tfidf_transformer = tfidf_transformer.fit(bow_transfrmd_review)

tfidf_transfrmd_review = tfidf_transformer.transform(bow_transfrmd_review)

display(tfidf_transfrmd_review.shape)  # (25000, 74047)

# Modelling

X_train, X_valid, y_train, y_valid = train_test_split(
    tfidf_transfrmd_review, train_df['sentiment'],
    test_size=0.22, random_state=101
)

display(type(X_train), type(X_valid))  # scipy.sparse.csr.csr_matrix

# Function to display statistics about predicted values

def display_pred_values_stats(y_pred, y_tocompare):
    cm = pd.crosstab(y_tocompare, y_pred)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print("CONFUSION MATRIX ------->> ")
    print(cm)
    print()
    
    ##check accuracy of model
    print('Classification paradox :------->>')
    print('Accuracy :- ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
    print()
    print('False Negative Rate :- ',round((FN*100)/(FN+TP),2))
    print()
    print('False Postive Rate :- ',round((FP*100)/(FP+TN),2))
    print()
    print(classification_report(y_tocompare, y_pred))

# Train the model

model_logreg = LogisticRegression(random_state=101, solver='lbfgs')
y_pred = model_logreg.predict(X_valid)

display_pred_stats(y_pred, y_valid)model_logreg.fit(X_train, y_train)

# Train set
y_pred = model_logreg.predict(X_train)
display_pred_values_stats(y_pred, y_train)

# Validation set
y_pred = model_logreg.predict(X_valid)
display_pred_values_stats(y_pred, y_valid)

# Create predictions for X_test
test_df = load_tsv_data("data/testData.tsv")

test_df['cleaned_review'] = test_df['review'].apply(
    lambda x: clean_review(
        x, to_lower=to_lower, remove_numbers=remove_numbers, remove_stopwords=remove_stopwords
    )
)

test_df['cln_rev_len'] = test_df['cleaned_review'].apply(
    lambda x: len(' '.join(x))
)

test_df['cln_rev_tokens_len'] = test_df['cleaned_review'].apply(len)

display(test_df.shape)  # (25000, 5)

bow_transfrmd_review = bow_vectorizer.transform(
    test_df['cleaned_review']
)
tfidf_transfrmd_review = tfidf_transformer.transform(bow_transfrmd_review)

display(tfidf_transfrmd_review.shape)  # (25000, 74047)

y_pred = model_logreg.predict(tfidf_transfrmd_review)

display(y_pred)  # 266 mesto / 578 mest

# Submit predictions

test_df = load_tsv_data("data/testData.tsv")

output = pd.DataFrame(
    {'id': test_df['id'], 'sentiment': y_pred}
)

output.to_csv('jul25_15h_54m.csv', index=False, quoting=3)
