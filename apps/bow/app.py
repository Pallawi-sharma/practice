import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')

documents = [
    "The cat is on the mat.",
    "The dog sat on the rug.",
    "Cats and dogs are great pets."
]

# def preprocess(text):
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(text.lower())
#     return [word for word in tokens if word.isalnum() and word not in stop_words]

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            filtered_tokens.append(word)
    return filtered_tokens

vocabulary = set()

preprocessed_docs = []
for doc in documents:
    tokens = preprocess(doc)
    preprocessed_docs.append(tokens)
    vocabulary.update(tokens)

vocabulary = sorted(vocabulary)

def create_bow(doc_tokens, vocabulary):
    bow_vector = [0] * len(vocabulary)
    token_counts = Counter(doc_tokens)
    # Counter return object
    for i, word in enumerate(vocabulary):
        bow_vector[i] = token_counts[word]
    return bow_vector

bow_matrix = []
for doc in preprocessed_docs:
    matrix = create_bow(doc, vocabulary)
    bow_matrix.append(matrix)
    
# bow_matrix = [create_bow(doc, vocabulary) for doc in preprocessed_docs]

print("Vocabulary:", vocabulary)

print("\nBag of Words Matrix:")
for row in bow_matrix:
    print(row)
