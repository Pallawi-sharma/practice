---
author: Kumari Pallawi
title: Introdunction Of NLP
tags: self learning, Introdunction Of NLP, project, documentation.
description: NLP
---

# Introdunction Natural Language Processing (NLP)

## Prerequisites

- Python
- Basic Understanding of Statistics
- Basic Knowledge of Machine Learning
- Data Preprocessing
- ANN
- Optimization

## What is NLP?

- The goal of NLP is to develop algorithms and systems that allow machines to process and interact with natural languages, such as English, Spanish, or Mandarin, just like humans do.
  **Core NLP Tasks:**
  **Named Entity Recognition (NER):** Identifying and classifying entities in text, such as names of people, places, or organizations.
  **Sentiment Analysis:** Determining the emotional tone behind a series of words, often used to analyze opinions in text data.
  **Machine Translation:** Automatically translating text from one language to another (e.g., Google Translate).
  **Speech Recognition:** Converting spoken language into written text.
  **Text Summarization:** Automatically generating concise summaries of longer texts.
  **Question Answering:** Systems that can answer questions posed in natural language.

**Text Preprocessing:**Preparing raw text data for analysis, which includes:
**Tokenization:** Splitting text into smaller units, like words or sentences.
**Stemming/Lemmatization:** Reducing words to their base or root form.
**Stop Word Removal:** Eliminating common words like "the" or "is" that don’t add much meaning to the analysis.
**POS Tagging:** Identifying the part of speech (noun, verb, adjective) for each word in a sentence.

## Why NLP

- Natural Language Processing (NLP) plays a crucial role in modern technology for several compelling reasons.

1. Human-Machine Communication
   **Natural Language Interface:** NLP allows people to interact with machines using natural language (like English, Spanish, etc.), making communication more intuitive compared to traditional commands or code.
   **Voice Assistants:** NLP powers voice-activated systems like Siri, Alexa, and Google Assistant, enabling natural conversations and commands.
   **Customer Service Automation:** Chatbots and virtual assistants, fueled by NLP, handle customer queries in a human-like manner, reducing the need for human intervention and improving efficiency.

2. Text Data Analysis
   **Massive Text Data:** The digital world generates vast amounts of unstructured text data—emails, social media posts, reviews, articles, etc. NLP helps analyze and extract valuable insights from this data.
   **Sentiment Analysis:** Businesses use NLP to understand public sentiment about products, services, or brands from social media, reviews, and surveys. This helps in gauging customer satisfaction and brand reputation.

3. Improving Search Engines
   **Better Search Results:** NLP enhances search engines by enabling them to understand natural language queries more accurately and provide more relevant results (e.g., Google Search, Bing).
   **Semantic Search:** Rather than just matching keywords, NLP helps search engines understand the meaning behind the query, providing more accurate responses to complex questions.

4. Document Processing and Automation
   **Automating Content:** NLP helps automate the summarization of long documents, extraction of important information, or translation of content into multiple languages, saving time and reducing manual effort.
   **Legal and Financial Analysis:** NLP is used to process complex legal, financial, and medical documents, making it easier for professionals to retrieve critical information without manually combing through pages.

5. Personalization
   **Content Recommendations:** NLP helps platforms like Netflix, YouTube, and Amazon recommend personalized content or products by understanding user preferences based on search history or past interactions.
   **Ad Targeting:** NLP helps advertisers analyze user data to show more relevant ads based on user interests, making advertising more effective.

6. Improving Accessibility
   **Assistive Technologies:** NLP helps people with disabilities by enabling speech-to-text, text-to-speech, and real-time captioning. This improves access to technology for people who have visual, hearing, or mobility impairments.
   **Language Learning:** NLP-based tools like Grammarly or Duolingo assist users in improving their language skills by providing grammar corrections, feedback, and language learning assistance.

## Tokenization

- These tokens can be words, sentences, or subwords, depending on the level of tokenization. It is an essential preprocessing step that helps prepare text data for further analysis or for input into machine learning models.
  **Types of Tokenization:**

1. Word Tokenization:
   - In this process, the text is split into individual words.
   - For example:
     - Input: "Natural Language Processing is interesting."
     - Output: ["Natural", "Language", "Processing", "is", "interesting", "."]
   - This is the most common form of tokenization, and it's useful for most NLP tasks.
2. Sentence Tokenization:
   - The text is divided into sentences rather than words.
   - For example:
     - Input: "I love NLP. It's very useful."
     - Output: ["I love NLP.", "It's very useful."]
   - Sentence tokenization is useful when the context of an entire sentence is important.
3. Character Tokenization:
   - This type of tokenization splits the text into individual characters.
   - For example:
     - Input: "NLP"
     - Output: ["N", "L", "P"]

- Character tokenization is useful in tasks like text generation or dealing with languages that don’t have spaces between words (like Chinese or Japanese).

4. Subword Tokenization:
   - Subword tokenization, words are split into subwords, typically using techniques like Byte Pair Encoding (BPE) or WordPiece.
   - For example:
     - Input: "unbelievable"
     - Output: ["un", "believ", "able"]
   - This method is commonly used in modern NLP models (e.g., BERT, GPT) as it helps reduce vocabulary size while allowing the model to handle rare words more efficiently.

- NLTK (Natural Language Toolkit): A popular Python library that provides various tokenization methods for words and sentences.

```python
from nltk.tokenize import word_tokenize, sent_tokenize
text = "Natural Language Processing is interesting. Let's learn it!"
word_tokens = word_tokenize(text)
sent_tokens = sent_tokenize(text)
print(word_tokens)
print(sent_tokens)
```

- spaCy: A fast NLP library that includes built-in tokenizers for different languages.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Natural Language Processing is amazing!")
tokens = [token.text for token in doc]
print(tokens)
```

- Hugging Face's Tokenizers: A library for subword tokenization, particularly for transformer models like BERT and GPT.

## Stemming/Lemmatization

- Stemming and Lemmatization are two key techniques in Natural Language Processing (NLP) for text normalization. Both aim to reduce words to their base or root form, but they do so in different ways.

  1.  Stemming:

- Stemming is the process of reducing a word to its base or root form by chopping off prefixes or suffixes. The result may not be a valid word but is intended to represent the "stem" of the word.
- Example: - "running" → "run", "runner" → "run", "studies" → "studi"
- Stemming algorithms (like Porter Stemmer) apply rules to remove common suffixes such as "ing", "ed", "es", "s", etc., without considering the actual meaning of the word. - Pros - Fast and efficient. - Useful in applications where approximate matches are sufficient (e.g., search engines). - Cons - The output is often not a valid word (e.g., "studies" becomes "studi"). - Can lead to inaccurate results, as it doesn't account for the context or meaning of the word.
- Example

```python
  from nltk.stem import PorterStemmer
  stemmer = PorterStemmer()
  print(stemmer.stem("running")) # Output: run
  print(stemmer.stem("studies")) # Output: studi
```

1.  Lemmatization

- Lemmatization is the process of reducing a word to its base or root form, known as the lemma, while considering its meaning and part of speech. Lemmatization returns a valid word.
- Example:
  - "running" → "run", "better" → "good", "studies" → "study"
- Lemmatization uses a dictionary and considers the context, such as whether the word is a noun, verb, or adjective. This process ensures that the lemma is a valid word.
- Pros
  - More accurate than stemming.
  - Always produces a valid word.
- Cons:
  - Slower than stemming due to the need for dictionary lookup and context analysis.
- Example

```python
  from nltk.stem import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()
  print(lemmatizer.lemmatize("running", pos="v"))  # Output: run
  print(lemmatizer.lemmatize("better", pos="a"))   # Output: good
```

- When to Use:

  **Stemming:** Use when you need a fast and approximate solution, like in search engines or for applications where speed is more important than perfect accuracy.

  **Lemmatization:** Use when you need accuracy, and the exact meaning of words is critical, like in sentiment analysis or text classification tasks.

## Why Remove Stop Words?

- **Reduce Noise:** Stop words often add unnecessary noise to text data, making it harder to focus on important words. For example, in the sentence "The cat is on the mat," the key words are "cat" and "mat", while "the", "is", "on" are considered irrelevant.
- **Efficiency:** Removing stop words reduces the number of tokens that need to be processed, which can speed up computations, especially in large datasets.
- **Improve Model Performance:** For many NLP tasks, removing stop words can enhance model performance by focusing the model on the most meaningful words.
- **Dimensionality Reduction:** In tasks like text classification or sentiment analysis, stop word removal helps reduce the dimensionality of the text, making the data easier to handle.
- Example
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
text = "The quick brown fox jumps over the lazy dog."
words = word_tokenize(text)

filtered_sentence = [w for w in words if not w.lower() in stop_words]
print(filtered_sentence)
```

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The quick brown fox jumps over the lazy dog.")

filtered_sentence = [token.text for token in doc if not token.is_stop]
print(filtered_sentence)

```

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

text = "The quick brown fox jumps over the lazy dog."
words = text.split()
filtered_sentence = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
print(filtered_sentence)
```

```python
custom_stop_words = stop_words.union({"fox", "dog"})
filtered_sentence = [w for w in words if not w.lower() in custom_stop_words]
print(filtered_sentence)
```
## Terminologies used in NLP

1. CORPUS
2. Documents
3. Vocabulary
4. Words
   
