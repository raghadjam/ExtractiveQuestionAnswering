# retriever.py
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, json_path):
        self.contexts = []
        self._load_contexts(json_path)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.contexts)

    def _load_contexts(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for article in data['data']:
            for para in article['paragraphs']:
                self.contexts.append(para['context'])

    def retrieve(self, question, top_k=1):
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.contexts[i] for i in top_indices]
