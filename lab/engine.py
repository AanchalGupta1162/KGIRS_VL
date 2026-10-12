"""The TF-IDF retrieval pipeline: tokenize -> TF -> DF -> IDF -> TF-IDF -> score -> rank.
Pure Python/standard-library logic, with no Streamlit calls, so it can be unit-tested on its own.
"""
import re
import math
from collections import Counter

from .config import STOPWORDS

def tokenize(text: str) -> list:
    """Lower-cases, strips punctuation, and removes stop words / single-character tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_vocabulary(doc_tokens_list: list) -> list:
    """Vocabulary is derived strictly from the corpus (not the query)."""
    vocab = set()
    for toks in doc_tokens_list:
        vocab.update(toks)
    return sorted(vocab)


def compute_tf(tokens: list, vocab: list, scheme: str) -> dict:
    counts = Counter(tokens)
    n = len(tokens)
    tf_vec = {}
    for term in vocab:
        c = counts.get(term, 0)
        if scheme == "Raw Term Frequency":
            val = float(c)
        elif scheme == "Normalized Term Frequency":
            val = (c / n) if n > 0 else 0.0
        else:  # Log-Normalized Term Frequency
            val = (1.0 + math.log10(c)) if c > 0 else 0.0
        tf_vec[term] = val
    return tf_vec


def compute_df(doc_tokens_list: list, vocab: list) -> dict:
    df = {}
    for term in vocab:
        df[term] = sum(1 for toks in doc_tokens_list if term in toks)
    return df


def compute_idf(df: dict, vocab: list, n_docs: int, scheme: str) -> dict:
    idf = {}
    for term in vocab:
        d = df.get(term, 0)
        if d <= 0:
            idf[term] = 0.0
            continue
        if scheme.startswith("Standard"):
            idf[term] = math.log10(n_docs / d)
        else:
            idf[term] = math.log10(1.0 + (n_docs / d))
    return idf


def tfidf_vector(tf_vec: dict, idf_vec: dict, vocab: list) -> dict:
    return {t: tf_vec[t] * idf_vec[t] for t in vocab}


def cosine_similarity(v1: dict, v2: dict, vocab: list) -> float:
    dot = sum(v1[t] * v2[t] for t in vocab)
    n1 = math.sqrt(sum(v1[t] ** 2 for t in vocab))
    n2 = math.sqrt(sum(v2[t] ** 2 for t in vocab))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def run_retrieval(corpus: list, query: str, tf_scheme: str, idf_scheme: str, use_cosine: bool = True) -> dict:
    """
    Runs the full TF-IDF retrieval pipeline: tokenize -> TF -> DF -> IDF -> TF-IDF -> score -> rank.
    Replaces the generic 'run_simulation' hook from the base template.
    """
    doc_tokens_list = [tokenize(d) for d in corpus]
    vocab = build_vocabulary(doc_tokens_list)

    query_tokens_all = tokenize(query)
    oov_terms = sorted(set(t for t in query_tokens_all if t not in vocab))
    query_tokens = [t for t in query_tokens_all if t in vocab]

    n_docs = len(corpus)
    df = compute_df(doc_tokens_list, vocab)
    idf = compute_idf(df, vocab, n_docs, idf_scheme)

    tf_matrix = [compute_tf(toks, vocab, tf_scheme) for toks in doc_tokens_list]
    tfidf_matrix = [tfidf_vector(tf, idf, vocab) for tf in tf_matrix]

    query_tf = compute_tf(query_tokens, vocab, tf_scheme)
    query_tfidf = tfidf_vector(query_tf, idf, vocab)

    scores = []
    for doc_vec in tfidf_matrix:
        if use_cosine:
            score = cosine_similarity(query_tfidf, doc_vec, vocab)
        else:
            score = sum(query_tfidf[t] * doc_vec[t] for t in vocab)
        scores.append(score)

    ranking = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)

    return {
        "vocab": vocab, "df": df, "idf": idf,
        "tf_matrix": tf_matrix, "tfidf_matrix": tfidf_matrix,
        "query_tfidf": query_tfidf, "scores": scores,
        "ranking": ranking, "oov_terms": oov_terms,
        "query_tokens": query_tokens
    }
