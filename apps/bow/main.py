from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "The cat is on the mat.",
    "The dog sat on the rug.",
    "Cats and dogs are great pets."
]
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(bow_matrix.toarray())