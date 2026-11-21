"""
Find the chunks of a document that are most related to the question.

Stage 1 - fast hybrid search over all chunks:
  - semantic score: cosine similarity of sentence embeddings (MiniLM).
    Finds chunks with the same meaning even if the words are different.
  - keyword score: TF-IDF cosine similarity.
    Finds chunks that share rare, important words (names, codes, commands).

Stage 2 - re-ranking of the best candidates with a cross-encoder.
  A cross-encoder reads the question and the chunk *together*, so it judges
  relevance much better than comparing two separate embeddings, but it is
  slower, so it is only run on the top candidates from stage 1.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
RERANK_CANDIDATES = 15

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_reranker():
    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=8)
def embed_chunks(chunks):
    # cached so a document is only embedded once, not for every question
    return load_embedding_model().encode(list(chunks), normalize_embeddings=True)


def keyword_scores(chunks, question):
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    try:
        matrix = vectorizer.fit_transform(chunks + [question])
    except ValueError:
        # happens when the text only has stop words like "the", "is"
        return np.zeros(len(chunks))
    return cosine_similarity(matrix[-1], matrix[:-1])[0]


def hybrid_scores(chunks, question):
    question_embedding = load_embedding_model().encode([question], normalize_embeddings=True)
    semantic = cosine_similarity(question_embedding, embed_chunks(tuple(chunks)))[0]
    keyword = keyword_scores(chunks, question)
    return SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword


def retrieve_relevant_chunks(chunks, question, top_k=5, rerank=True):
    if not chunks or not question.strip():
        return []

    # Stage 1: hybrid search
    scores = hybrid_scores(chunks, question)
    candidates = scores.argsort()[::-1][:RERANK_CANDIDATES if rerank else top_k]

    if not rerank:
        return [{"text": chunks[i], "score": float(scores[i])} for i in candidates]

    # Stage 2: cross-encoder re-ranking (score is a relevance logit;
    # converted to 0-1 with a sigmoid so it is easy to read)
    logits = load_reranker().predict([(question, chunks[i]) for i in candidates])
    relevance = 1 / (1 + np.exp(-np.asarray(logits)))
    order = np.argsort(relevance)[::-1][:top_k]

    return [{"text": chunks[candidates[j]], "score": float(relevance[j])} for j in order]
