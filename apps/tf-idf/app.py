import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Download stopwords if needed
nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = [
    'The cat is on the mat',
    'The dog is in the house',
    'The cat and dog are friends'
]

# Function to preprocess documents (tokenization and stopword removal)
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)  # Join tokens back into a string for CountVectorizer

# Preprocess each document
preprocessed_docs = [preprocess(doc) for doc in documents]

# Step 1: Convert documents into Bag of Words (BoW) using CountVectorizer
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(preprocessed_docs)

# Step 2: Apply TF-IDF Transformation using TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_bow)

# Show the TF-IDF matrix
print(X_tfidf.toarray())

# Show the feature names (i.e., words in the vocabulary)
print(vectorizer.get_feature_names_out())
