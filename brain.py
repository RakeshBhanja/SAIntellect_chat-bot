# brain.py
# Pure Python NLP matching — no external AI API needed

import string
import nltk

# Download all required NLTK data (safe for Render)
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('omw-1.4',    quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from knowledge import QA_DATA, FALLBACK_ANSWER

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

# Build TF-IDF corpus from all keyword phrases
corpus = []
corpus_index = []

for i, qa in enumerate(QA_DATA):
    combined = ' '.join(qa['keywords'])
    corpus.append(preprocess(combined))
    corpus_index.append(i)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(corpus)

def get_answer(user_input: str) -> str:
    if not user_input.strip():
        return "Please type a question!"

    cleaned = preprocess(user_input)

    # Step 1: Direct keyword match (fast path)
    user_lower = user_input.lower()
    for i, qa in enumerate(QA_DATA):
        for kw in qa['keywords']:
            if kw in user_lower:
                return qa['answer']

    # Step 2: TF-IDF cosine similarity (smart path)
    if cleaned:
        user_vec = vectorizer.transform([cleaned])
        scores = cosine_similarity(user_vec, tfidf_matrix)[0]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score > 0.15:
            return QA_DATA[corpus_index[best_idx]]['answer']

    return FALLBACK_ANSWER
