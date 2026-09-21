"""
Virtual Laboratory - Experiment 4: TF-IDF Based Document Retrieval
Course: Knowledge Graphs and Information Retrieval Systems 
Roll Nos: 16-20 | Group No: 4

Built on the standard 4-section Virtual Lab template:
  1. Theory: Concepts, objectives, procedure, and terminology.
  2. Simulation: Interactive TF-IDF retrieval sandbox with a live corpus/query engine.
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded trials, observations, and downloadable PDF report.

Note: No custom CSS is used so that Streamlit's native light and dark themes render seamlessly.
"""

import re
import math
import zlib
from collections import Counter
from html import escape as _html_escape
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Experiment 4: TF-IDF Based Document Retrieval",
    "course": "Knowledge Graphs and Information Retrieval Systems",
    "roll_no": "16-20",
    "group_no": "4",
    "aim": "To represent a document collection and a search query as **TF-IDF weighted vectors** in the "
           "vector space model, rank the documents by their **cosine similarity** to the query, observe how "
           "the choice of TF and IDF weighting scheme changes that ranking, and measure the quality of the "
           "ranking against relevance judgments using standard **information retrieval evaluation "
           "parameters** - Precision@k, Recall@k, F1@k, Average Precision, Reciprocal Rank and nDCG@k.",
    "objectives": [
        "Understand how Term Frequency (TF) and Inverse Document Frequency (IDF) capture local and "
        "global term importance respectively.",
        "Construct TF-IDF weighted vector representations for a document collection and a search query.",
        "Rank documents by similarity between the query vector and each document vector in the vector "
        "space model.",
        "Analyze how different TF and IDF weighting schemes influence retrieval ranking and result quality.",
        "Evaluate retrieval effectiveness using Precision@k, Recall@k, F1@k, Average Precision, "
        "Reciprocal Rank, and nDCG@k against relevance judgments."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles
TF-IDF (**Term Frequency - Inverse Document Frequency**) is a classical term-weighting scheme used in
the **Vector Space Model** for Information Retrieval. It scores every term in a document by combining
two complementary signals:

- **Term Frequency (TF)** - how often a term occurs within a document (local importance).
- **Inverse Document Frequency (IDF)** - how rare a term is across the entire collection (discriminative power).

Multiplying **TF x IDF** favors terms that occur frequently within a specific document but rarely
elsewhere in the corpus, since such terms are the most useful for distinguishing that document from
the rest of the collection. Terms that occur in almost every document (e.g. common words that survived
stop-word removal) end up with an IDF close to zero and therefore contribute little to the score.

### Vector Space Model
Both documents and the query are represented as vectors over a shared vocabulary of terms. The
similarity between the query vector and each document vector - typically measured using **cosine
similarity** - is used to rank documents by relevance. Cosine similarity normalizes for document
length, so a long document is not unfairly favored simply because it contains more words.

### Workflow & System Overview
1. **Tokenization** - corpus and query text are lower-cased, stripped of punctuation, and filtered
   through a stop-word list.
2. **Weighting** - TF and IDF values are computed for every (term, document) pair and multiplied
   together to form TF-IDF weights.
3. **Query Projection** - the query is embedded into the same TF-IDF space using the corpus-derived
   IDF values, producing a query vector.
4. **Similarity Scoring** - the query vector is compared against every document vector (cosine
   similarity or raw dot product).
5. **Ranking** - documents are sorted in descending order of similarity score to produce the final
   retrieval result.
6. **Evaluation** - the ranked list is compared against a set of documents judged relevant to the
   query, and effectiveness is summarized with standard IR evaluation parameters.

### Evaluation Parameters
A document counts as *retrieved* when its similarity score is above zero. Given the set of relevant
documents **R** and the top **k** retrieved documents:

- **Precision@k** - fraction of the top k results that are relevant: |relevant in top k| / k.
- **Recall@k** - fraction of all relevant documents found in the top k: |relevant in top k| / |R|.
- **F1@k** - harmonic mean of Precision@k and Recall@k: 2PR / (P + R).
- **Average Precision (AP)** - mean of the precision values at each rank where a relevant document
  appears, divided over |R|. Averaging AP over several queries gives **MAP**.
- **Reciprocal Rank (RR)** - 1 / rank of the first relevant document. Averaged over queries it is **MRR**.
- **nDCG@k** - Discounted Cumulative Gain, DCG@k = sum of rel_i / log2(i + 1), divided by the DCG of
  an ideal ranking. It rewards placing relevant documents near the top.
    """,
    "procedure": [
        "Step 1: Read the Aim and the Theory page to review TF-IDF weighting and the vector space model.",
        "Step 2: Open the Simulation page from the sidebar menu.",
        "Step 3: Inspect (or edit) the sample document corpus - one document per line.",
        "Step 4: Enter a search query that is relevant to the corpus, or pick one of the sample queries.",
        "Step 5: Choose a Term Frequency scheme and an Inverse Document Frequency scheme.",
        "Step 6: Observe the ranked documents, the similarity bar chart, and the underlying TF, IDF and "
        "TF-IDF matrices.",
        "Step 7: Open the Evaluation tab, mark the documents that are relevant to your query, choose the "
        "cut-off k, and note the Precision, Recall, F1, AP, RR and nDCG values.",
        "Step 8: Click 'Record this trial' after each configuration to log it into your session table.",
        "Step 9: Repeat for at least 3-4 distinct queries / weighting-scheme combinations.",
        "Step 10: Complete the assessment Quiz to test your conceptual understanding.",
        "Step 11: Open the Report page, enter your student information, and download your PDF report."
    ],
    "key_terms": {
        "Term Frequency (TF)": "How often a term occurs within a single document; may be a raw count, "
                                "length-normalized, or log-scaled.",
        "Inverse Document Frequency (IDF)": "How rare a term is across the whole collection; down-weights "
                                             "terms that appear in many documents.",
        "TF-IDF Weight": "Product of TF and IDF for a (term, document) pair - the term's importance in "
                          "that document relative to the corpus.",
        "Vector Space Model": "Representation of documents and queries as vectors over a shared "
                               "term vocabulary.",
        "Cosine Similarity": "Similarity between two vectors based on the cosine of the angle between "
                              "them; commonly used to rank documents by relevance.",
        "Document Frequency (df)": "The number of documents in the corpus that contain a given term "
                                    "at least once.",
        "Vocabulary": "The set of unique terms extracted from the corpus after tokenization and "
                       "stop-word removal.",
        "Out-of-Vocabulary (OOV) Term": "A query term that never occurs in the corpus; it cannot be "
                                         "assigned an IDF weight and is ignored during scoring.",
        "Relevance Judgment": "A human decision on whether a document satisfies the information need "
                               "behind a query; the ground truth for evaluation.",
        "Precision@k": "Share of the top k retrieved documents that are relevant.",
        "Recall@k": "Share of all relevant documents that appear in the top k results.",
        "Average Precision (AP)": "Mean of the precision values at the ranks of each relevant document; "
                                   "its mean over queries is MAP.",
        "nDCG@k": "Normalized Discounted Cumulative Gain; rewards relevant documents ranked near the top, "
                   "scaled so a perfect ranking scores 1."
    },
    "references": [
        "G. Salton and M. J. McGill, *Introduction to Modern Information Retrieval*, McGraw-Hill, 1983.",
        "C. D. Manning, P. Raghavan, and H. Schutze, *Introduction to Information Retrieval*, "
        "Cambridge University Press, 2008.",
        "IIT Kharagpur Virtual Labs - Information Retrieval discipline."
    ]
}

# The six stages of run_retrieval(), paired with the place in the Simulation page where each one is
# visible. Rendered as the flow strip at the top of the Real-world applications page.
PIPELINE_STAGES = [
    {"n": 1, "name": "Corpus",
     "what": "The collection you can retrieve from, one document per line.",
     "where": "Edit document corpus"},
    {"n": 2, "name": "Tokenize",
     "what": "Lower-case, strip punctuation, drop stop words and one-letter tokens.",
     "where": "Summary strip - vocabulary"},
    {"n": 3, "name": "TF and IDF",
     "what": "How often a term occurs in this document; how rare it is across all of them.",
     "where": "Matrices tab - TF, IDF"},
    {"n": 4, "name": "TF-IDF vectors",
     "what": "Multiply the two, so frequent-here-but-rare-elsewhere terms dominate.",
     "where": "Matrices tab - TF-IDF"},
    {"n": 5, "name": "Cosine scoring",
     "what": "Compare the query vector with every document vector, ignoring length.",
     "where": "Ranked results, Score chart"},
    {"n": 6, "name": "Rank and evaluate",
     "what": "Sort by score, then score the ranking itself against relevance judgments.",
     "where": "Evaluation tab"},
]

# Procedure steps grouped into phases: (label, icon, start index, end index) over THEORY_CONTENT["procedure"].
PROCEDURE_PHASES = [
    ("Prepare", ":material/menu_book:", 0, 2),
    ("Run the retrieval", ":material/play_arrow:", 2, 7),
    ("Record and assess", ":material/task_alt:", 7, None),
]

# Real-world uses of the same pipeline. Each carries a small corpus and query so the Applications page can
# run the real retrieval engine on it, rather than describing the result in prose.
APPLICATIONS = [
    {
        "name": "Web and site search",
        "icon": ":material/travel_explore:",
        "doc": "A web or help-centre page",
        "query_is": "What the user types",
        "metric": "nDCG@k",
        "why": "Every page says *account*, so it weighs nothing. Only one says *reset* - and that decides it.",
        "query": "reset my account password",
        "corpus": [
            "Reset your password from the account settings page by choosing Forgot password and confirming "
            "the link we email to you.",
            "Update the billing address and the payment card stored on your account from the same settings page.",
            "Our password policy asks for twelve characters, one number and one symbol on every new account.",
            "Contact the support desk if you still cannot sign in to your account after a password reset.",
            "The company was founded in 2011 and now employs four hundred people across six offices.",
        ],
    },
    {
        "name": "Spam and phishing filtering",
        "icon": ":material/report:",
        "doc": "One email or SMS",
        "query_is": "A profile of spam terms",
        "metric": "Precision@k",
        "why": "Shared vocabulary cancels out. What survives - *urgent*, *suspended*, *verify* - is the signal.",
        "query": "urgent verify your suspended account click this link",
        "corpus": [
            "URGENT your account has been suspended, click this link now to verify your identity before it "
            "is closed permanently.",
            "Verify your bank account immediately or the pending transfer will be cancelled, use the secure "
            "link below.",
            "Team lunch moves to Friday at one o'clock in the fourth floor kitchen, no need to reply.",
            "Your monthly invoice is attached as a PDF, no action is needed from your side.",
            "Congratulations you have won a prize, claim it urgently by clicking the link and paying the "
            "small handling fee.",
        ],
    },
    {
        "name": "More-like-this recommendation",
        "icon": ":material/recommend:",
        "doc": "Another article in the catalogue",
        "query_is": "The article being read",
        "metric": "Precision@k",
        "why": "Cosine divides length out, so a short closely-related piece still beats a long vague one.",
        "query": "knowledge graph entity linking for search",
        "corpus": [
            "Entity linking maps a mention in text to the matching node in a knowledge graph before the "
            "search runs.",
            "A knowledge graph stores entities as nodes and the relationships between them as labelled edges.",
            "Convolutional networks classify images by learning filters over neighbourhoods of pixels.",
            "Query understanding rewrites a search query using the entities recognised inside the text.",
            "The quarterly revenue report shows growth in three of the five regional markets.",
        ],
    },
    {
        "name": "Plagiarism and duplicate detection",
        "icon": ":material/content_copy:",
        "doc": "An earlier submission",
        "query_is": "The passage being checked",
        "metric": "Recall",
        "why": "Two texts on a topic share common words. Sharing the *rare* ones is what looks like copying.",
        "query": "documents and queries are represented as vectors over a shared vocabulary",
        "corpus": [
            "In the vector space model documents and queries are represented as vectors over a shared "
            "vocabulary of terms.",
            "Cosine similarity normalises for length so that long documents are not unfairly favoured.",
            "Boolean retrieval returns the set of documents matching a logical expression, without ranking them.",
            "A good sourdough loaf needs a mature starter and a long cold proof in the refrigerator.",
            "The vector representation of a document is sparse, because most vocabulary terms never occur in it.",
        ],
    },
    {
        "name": "Resume and job matching",
        "icon": ":material/badge:",
        "doc": "One candidate resume",
        "query_is": "The job description",
        "metric": "Recall@k",
        "why": "*Experience* and *team* are on every resume and add nothing. The skill terms carry the score.",
        "query": "python machine learning pipelines and model deployment experience",
        "corpus": [
            "Built and shipped machine learning pipelines in Python, owning model deployment behind a REST API.",
            "Five years of Python backend work on billing services, with occasional exposure to model deployment.",
            "Graphic designer specialising in brand identity, packaging and print production.",
            "Data analyst using SQL and spreadsheets to report on weekly marketing performance.",
            "Machine learning researcher publishing on graph neural networks, working mostly in Python.",
        ],
    },
    {
        "name": "Support ticket routing",
        "icon": ":material/support_agent:",
        "doc": "A support queue",
        "query_is": "An incoming ticket",
        "metric": "Reciprocal Rank",
        "why": "The corpus is one document per queue, so a single term unique to a queue routes the ticket.",
        "query": "wrong tax rate on my invoice",
        "corpus": [
            "Billing queue: invoice and tax rate problems, refunds, failed payments and subscription changes.",
            "Authentication queue: sign-in problems, password resets, two-factor codes and locked accounts.",
            "Performance queue: slow page loads, request timeouts and report generation failures.",
            "Onboarding queue: account setup, data import and user provisioning for new customers.",
            "Hardware queue: shipping, returns and physical replacement of faulty devices.",
        ],
    },
]

# Small, generic English stop-word list used during tokenization.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that", "this", "these",
    "those", "it", "its", "their", "they", "he", "she", "we", "you", "i", "your", "our",
    "his", "her", "them", "which", "who", "whom", "what", "when", "where", "why", "how",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must", "do",
    "does", "did", "have", "has", "had", "not", "no", "if", "than", "then", "so", "such",
    "there", "here", "about", "into", "over", "under", "also", "using", "used", "use"
}

# Default sample corpus: 41 documents, one per line, across five course topics (information retrieval,
# knowledge graphs, machine learning / NLP, databases, semantic search / RAG) plus two long documents.
SAMPLE_CORPUS_PATH = Path(__file__).with_name("sample_corpus.txt")
DEFAULT_CORPUS = [line.strip() for line in SAMPLE_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
DEFAULT_QUERY = "TF-IDF document ranking"

# Relevance judgments for the sample corpus: query -> documents a person judged relevant by topic,
# independent of how TF-IDF happens to rank them. Used as the Evaluation tab's defaults.
SAMPLE_JUDGMENTS = {
    "TF-IDF document ranking": ["D3", "D4", "D40"],
    "evaluation measures for ranked retrieval": ["D5", "D37", "D41"],
    "graph database query language": ["D10", "D27"],
    "semantic search with embeddings": ["D33", "D34", "D36", "D38"],
    "reducing hallucinations in language models": ["D35", "D39"],
    "vocabulary mismatch synonyms": ["D6", "D17", "D33", "D36"],
}

TF_SCHEMES = ["Raw Term Frequency", "Normalized Term Frequency", "Log-Normalized Term Frequency"]
IDF_SCHEMES = ["Standard IDF: log(N / df)", "Smoothed IDF: log(1 + N / df)"]

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What does the acronym TF-IDF stand for?",
        "options": [
            "A) Term Frequency - Inverse Document Frequency",
            "B) Text Format - Index Data Field",
            "C) Total Frequency - Indexed Data File",
            "D) Term Filter - Inverted Document Format"
        ],
        "answer_index": 0,
        "explanation": "TF-IDF combines Term Frequency (local importance) with Inverse Document Frequency "
                        "(global rarity) into a single weight."
    },
    {
        "id": 2,
        "question": "What is the main purpose of the IDF component in TF-IDF weighting?",
        "options": [
            "A) To count how many times a term occurs in a single document",
            "B) To down-weight terms that occur in many documents and up-weight rare, discriminative terms",
            "C) To remove punctuation and stop words from the text",
            "D) To increase the length of every document artificially"
        ],
        "answer_index": 1,
        "explanation": "IDF reduces the weight of common terms that appear across most documents and boosts "
                        "the weight of rarer, more informative terms."
    },
    {
        "id": 3,
        "question": "If a term occurs in every single document of the corpus (df = N), what happens to its "
                     "standard IDF value, log(N / df)?",
        "options": [
            "A) It becomes negative",
            "B) It becomes 0, since log(N/N) = log(1) = 0",
            "C) It becomes infinite",
            "D) It stays equal to the term frequency"
        ],
        "answer_index": 1,
        "explanation": "When df = N, the ratio N/df equals 1, and log(1) = 0, so the term contributes nothing "
                        "to the TF-IDF score."
    },
    {
        "id": 4,
        "question": "Which similarity measure is most commonly used to rank documents against a query in the "
                     "vector space model, since it is unaffected by document length?",
        "options": [
            "A) Euclidean distance",
            "B) Hamming distance",
            "C) Cosine similarity",
            "D) Manhattan distance"
        ],
        "answer_index": 2,
        "explanation": "Cosine similarity measures the angle between two vectors, which normalizes for "
                        "magnitude/length so longer documents are not unfairly favored."
    },
    {
        "id": 5,
        "question": "What is a key drawback of representing documents using raw term counts alone "
                     "(without IDF)?",
        "options": [
            "A) It cannot be computed for large corpora",
            "B) It overweights frequent but largely uninformative words shared across many documents",
            "C) It ignores the order of words completely",
            "D) It requires a knowledge graph to compute"
        ],
        "answer_index": 1,
        "explanation": "Without IDF, common words that appear in most documents can dominate the score even "
                        "though they carry little discriminative meaning."
    },
    {
        "id": 6,
        "question": "In the vector space model, how are a query and each document represented so that they "
                     "can be compared?",
        "options": [
            "A) As raw, unprocessed text strings",
            "B) As nodes in a knowledge graph",
            "C) As numeric vectors over a shared term vocabulary",
            "D) As images encoding word frequency"
        ],
        "answer_index": 2,
        "explanation": "Both the query and every document are projected into the same vector space defined "
                        "by the corpus vocabulary, enabling direct numeric comparison."
    },
    {
        "id": 7,
        "question": "What effect does log-normalized term frequency, 1 + log10(count), have compared to "
                     "using the raw count directly?",
        "options": [
            "A) It amplifies the impact of very high raw counts even further",
            "B) It dampens the effect of very high raw counts, giving diminishing returns for repeated terms",
            "C) It has an identical effect to raw counts in every case",
            "D) It converts the term frequency into a document frequency"
        ],
        "answer_index": 1,
        "explanation": "Log scaling compresses large counts, so a term occurring 20 times does not score "
                        "twenty times higher than a term occurring once."
    },
    {
        "id": 8,
        "question": "What happens when a query contains a term that never appears anywhere in the corpus "
                     "vocabulary (an out-of-vocabulary term)?",
        "options": [
            "A) The retrieval system crashes",
            "B) The whole query is discarded and no results are returned",
            "C) That term has no IDF/TF-IDF weight in the corpus and is effectively ignored during scoring",
            "D) The term is automatically added to every document"
        ],
        "answer_index": 2,
        "explanation": "Since TF-IDF weights are derived from the corpus, a term that never occurs there has "
                        "no document frequency and cannot contribute to any similarity score."
    },
    {
        "id": 9,
        "question": "Which retrieval algorithm improves upon plain TF-IDF by adding term-frequency saturation "
                     "and explicit document-length normalization (tunable via parameters k1 and b)?",
        "options": [
            "A) BM25",
            "B) PageRank",
            "C) k-Nearest Neighbors",
            "D) Breadth-First Search"
        ],
        "answer_index": 0,
        "explanation": "BM25 is a probabilistic ranking function that extends TF-IDF-style weighting with "
                        "saturation and length normalization, and is covered in the next experiment."
    },
    {
        "id": 10,
        "question": "Why can TF-IDF fail to retrieve a genuinely relevant document that uses a synonym of the "
                     "query term (e.g. 'automobile' instead of 'car')?",
        "options": [
            "A) TF-IDF automatically expands queries with synonyms, so this never happens",
            "B) TF-IDF is a purely lexical, exact-match model and has no built-in notion of word meaning",
            "C) TF-IDF only works on numeric data, not text",
            "D) Synonyms always receive an IDF of exactly zero"
        ],
        "answer_index": 1,
        "explanation": "TF-IDF matches on exact surface term overlap; it has no semantic understanding, which "
                        "is one motivation for dense embedding-based semantic search (a later experiment)."
    }
]


# ======================================================================================
# 2. TF-IDF RETRIEVAL ENGINE (CORE SIMULATION LOGIC)
# ======================================================================================

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


def evaluate_ranking(ranking: list, scores: list, relevant: set, k: int) -> dict:
    """
    Computes IR evaluation parameters for one ranked list against a set of relevant document indices.
    Only documents with a score above zero count as retrieved.
    """
    retrieved = [i for i in ranking if scores[i] > 0]
    top_k = retrieved[:k]
    n_rel = len(relevant)
    hits_k = sum(1 for i in top_k if i in relevant)

    precision_k = hits_k / k if k else 0.0
    recall_k = hits_k / n_rel if n_rel else 0.0
    f1_k = (2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k + recall_k) else 0.0

    # Precision and recall after each retrieved document, used for AP and the PR curve.
    per_rank, hits, precisions_at_rel = [], 0, []
    for rank, idx in enumerate(retrieved, start=1):
        is_rel = idx in relevant
        if is_rel:
            hits += 1
            precisions_at_rel.append(hits / rank)
        per_rank.append({"rank": rank, "doc": idx, "relevant": is_rel,
                         "precision": hits / rank, "recall": hits / n_rel if n_rel else 0.0})
    avg_precision = sum(precisions_at_rel) / n_rel if n_rel else 0.0

    first_rel = next((r["rank"] for r in per_rank if r["relevant"]), None)
    reciprocal_rank = 1.0 / first_rel if first_rel else 0.0

    dcg = sum(1.0 / math.log2(rank + 1) for rank, idx in enumerate(top_k, start=1) if idx in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_rel, k) + 1))
    ndcg_k = dcg / idcg if idcg else 0.0

    return {
        "precision_k": precision_k, "recall_k": recall_k, "f1_k": f1_k,
        "avg_precision": avg_precision, "reciprocal_rank": reciprocal_rank, "ndcg_k": ndcg_k,
        "hits_k": hits_k, "n_retrieved": len(retrieved), "per_rank": per_rank
    }


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

PDF_CHAR_MAP = str.maketrans({"—": "-", "–": "-", "‘": "'", "’": "'",
                              "“": '"', "”": '"', "…": "...", "•": "-", " ": " "})


class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                         trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                         student_notes: str) -> bytes:
    """Compiles experiment trial records into a formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, EXPERIMENT_CONFIG["course"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 22, "FD")
    box_top = pdf.get_y()

    pdf.set_xy(14, box_top + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll Nos / Group:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, box_top + 10)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_total and quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score / quiz_total) * 100 if quiz_total else 0)}%)", 1)

    pdf.set_xy(10, box_top + 24)
    pdf.ln(4)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.multi_cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Data", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        from fpdf.fonts import FontFace

        cols = list(trials_df.columns)
        # Relative widths: long text columns (query, schemes) get more room; cells wrap instead of truncating.
        width_weights = {"Trial #": 1.1, "Query": 3.8, "TF Scheme": 3.0, "IDF Scheme": 3.6,
                         "Cosine Norm.": 1.4, "Top Doc": 1.2, "Top Score": 1.5, "k": 0.7,
                         "P@k": 1.4, "R@k": 1.4, "F1@k": 1.4, "AP": 1.4, "nDCG@k": 1.5, "Timestamp": 1.7}
        col_widths = [width_weights.get(c, 2.0) for c in cols]

        def pdf_text(val) -> str:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                text = "-"
            elif isinstance(val, (bool, np.bool_)):
                text = "Yes" if val else "No"
            elif isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            # Built-in Helvetica only covers Latin-1: map common typographic characters, replace the rest.
            text = text.translate(PDF_CHAR_MAP)
            return text.encode("latin-1", "replace").decode("latin-1")

        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        with pdf.table(
            col_widths=col_widths,
            text_align="CENTER",
            line_height=4.2,
            padding=1.2,
            borders_layout="ALL",
            cell_fill_color=(248, 250, 252),
            cell_fill_mode="ROWS",
            headings_style=FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(37, 99, 235)),
            repeat_headings=1,
        ) as table:
            header = table.row()
            for c in cols:
                header.cell(pdf_text(c))
            for _, row in trials_df.iterrows():
                data_row = table.row()
                for c in cols:
                    data_row.cell(pdf_text(row[c]))
    pdf.ln(5)

    # 3. Discussion & Notes
    if pdf.get_y() > pdf.h - 45:  # keep the heading together with its text
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The TF-IDF retrieval trials demonstrated consistent, query-relevant document ranking across "
        "the tested weighting-scheme configurations."
    )
    pdf.multi_cell(0, 5, notes_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Sign-off line
    if pdf.get_y() > pdf.h - 45:  # keep the line and its caption on the same page
        pdf.add_page()
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. SECTION RENDERERS: AIM, THEORY, PROCEDURE, APPLICATIONS, REFERENCES, SIMULATION, QUIZ, REPORT
# ======================================================================================

TF_SHORT = {
    "Raw Term Frequency": "Raw count",
    "Normalized Term Frequency": "Normalized",
    "Log-Normalized Term Frequency": "Log-normalized",
}
IDF_SHORT = {
    "Standard IDF: log(N / df)": "Standard",
    "Smoothed IDF: log(1 + N / df)": "Smoothed",
}
DEFAULT_NOTES = ("The TF-IDF retrieval trials demonstrated consistent, query-relevant document ranking across "
                 "the tested weighting-scheme configurations.")

# Characters that Streamlit markdown would otherwise interpret inside document text.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|~$:])")


def _md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def highlight_terms(text: str, terms: set) -> str:
    """Returns markdown for `text` with every word whose token is a query term highlighted."""
    parts = re.split(r"(\s+)", text)
    out = []
    for part in parts:
        if part.strip() and set(tokenize(part)) & terms:
            out.append(f":orange-background[**{_md_escape(part)}**]")
        else:
            out.append(_md_escape(part))
    return "".join(out)


def render_aim_section():
    """Renders the Aim page: the aim statement, the objectives and what the student hands in."""
    st.header("Aim", icon=":material/flag:")
    st.caption("What this experiment sets out to do, and why it is worth doing.")

    with st.container(border=True):
        st.markdown(EXPERIMENT_CONFIG["aim"])

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Objectives", icon=":material/checklist_rtl:")
        for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"], start=1):
            st.markdown(f"**{i}.** &nbsp; {obj}")
    with right:
        st.subheader("What you hand in", icon=":material/inventory_2:")
        st.markdown(
            "- At least **3 recorded trials** across different queries and weighting schemes.\n"
            "- The **six evaluation parameters** for each trial, measured against relevance judgments.\n"
            "- A completed **quiz** score.\n"
            "- A downloaded **PDF lab report** with your trial table and observations."
        )

    st.divider()
    render_applications_block()


def render_theory_section():
    """Renders the Theory page: the two formulas, the background reading and the glossary."""
    st.header("Theory", icon=":material/menu_book:")
    st.caption("How TF-IDF turns text into vectors and ranks documents against a query.")

    with st.container(border=True):
        st.markdown("**The two formulas behind every score in this lab**")
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"w_{t,d} = \mathrm{tf}_{t,d} \times \log_{10}\frac{N}{\mathrm{df}_t}")
            st.caption("Weight of term *t* in document *d*: frequent here, rare elsewhere.")
        with c2:
            st.latex(r"\cos(\vec q, \vec d) = \frac{\vec q \cdot \vec d}{\lVert \vec q \rVert \, \lVert \vec d \rVert}")
            st.caption("Similarity between the query and a document, independent of length.")

    overview, glossary = st.tabs([
        ":material/article: Overview",
        ":material/dictionary: Key terms",
    ])
    with overview:
        st.markdown(THEORY_CONTENT["background"])
    with glossary:
        var_df = pd.DataFrame(list(THEORY_CONTENT["key_terms"].items()), columns=["Term", "Definition"])
        st.dataframe(
            var_df, hide_index=True, width="stretch",
            column_config={"Term": st.column_config.TextColumn(width="medium"),
                           "Definition": st.column_config.TextColumn(width="large")}
        )


def render_procedure_section():
    """Renders the Procedure page: the ordered steps, grouped into three phases."""
    st.header("Procedure", icon=":material/checklist:")
    st.caption("Follow the steps in order. Steps 3 to 9 all happen on the Simulation page.")

    steps = THEORY_CONTENT["procedure"]
    for label, icon, start, end in PROCEDURE_PHASES:
        st.subheader(label, icon=icon)
        with st.container(border=True):
            for step in steps[start:end]:
                name, _, text = step.partition(": ")
                st.markdown(f"**{name}** &nbsp; {text}")


ACCENT = "#2A9AA4"

# Inline styling for the illustrated blocks on the Real-world applications page. Colors are either the
# shared accent or a neutral gray alpha, so the same markup reads correctly in the light and dark themes.
APP_VISUAL_CSS = """
<style>
.kg-wrap { width: 100%; }
.kg-chips { display: flex; flex-wrap: wrap; align-items: baseline; gap: .35rem; }
.kg-chip {
    display: inline-block; padding: .15em .6em; border-radius: 999px; font-weight: 600;
    background: rgba(42, 154, 164, 0.16); border: 1px solid rgba(42, 154, 164, 0.45); line-height: 1.5;
}
.kg-chip.kg-dead {
    background: rgba(128, 128, 128, 0.10); border-color: rgba(128, 128, 128, 0.30); opacity: .65;
}
.kg-row { display: flex; align-items: center; gap: .75rem; padding: .45rem 0;
          border-bottom: 1px solid rgba(128, 128, 128, 0.22); }
.kg-row:last-child { border-bottom: none; }
.kg-rank { flex: 0 0 1.9rem; height: 1.9rem; border-radius: 50%; display: flex; align-items: center;
           justify-content: center; font-weight: 700; font-size: .85rem;
           background: rgba(128, 128, 128, 0.18); }
.kg-rank.kg-top { background: #2A9AA4; color: #fff; }
.kg-body { flex: 1 1 auto; min-width: 0; }
.kg-bar { height: 8px; border-radius: 4px; background: rgba(128, 128, 128, 0.16); overflow: hidden;
          margin-bottom: .3rem; }
.kg-bar > span { display: block; height: 100%; border-radius: 4px; background: #2A9AA4; }
.kg-bar.kg-zero > span { background: rgba(128, 128, 128, 0.35); }
.kg-text { font-size: .86rem; line-height: 1.45; opacity: .92; }
.kg-score { flex: 0 0 3.4rem; text-align: right; font-variant-numeric: tabular-nums;
            font-weight: 600; font-size: .85rem; }
.kg-hit { background: rgba(224, 146, 58, 0.30); color: inherit; padding: 0 .18em; border-radius: 3px; }
.kg-idfrow { display: flex; align-items: center; gap: .75rem; padding: .3rem 0; }
.kg-idflab { flex: 0 0 11rem; font-size: .82rem; opacity: .75; }
.kg-idfval { flex: 0 0 2.6rem; text-align: right; font-variant-numeric: tabular-nums;
             font-weight: 600; font-size: .85rem; }
</style>
"""


def _esc(text) -> str:
    return _html_escape(str(text), quote=True)


def _highlight_html(text: str, terms: set) -> str:
    """Same idea as highlight_terms(), but emitting HTML for the illustrated result rows."""
    out = []
    for part in re.split(r"(\s+)", text):
        if part.strip() and set(tokenize(part)) & terms:
            out.append(f'<mark class="kg-hit">{_esc(part)}</mark>')
        else:
            out.append(_esc(part))
    return "".join(out)


def _pipeline_svg() -> str:
    """Draws the six retrieval stages as a single illustrated flow strip.

    Every shape uses currentColor or the accent, so the drawing follows the active theme.
    """
    art = []
    for i in range(len(PIPELINE_STAGES)):
        cx = i * 200 + 80
        if i == 0:  # a small shelf of documents
            shapes = []
            for j in range(3):
                dx = cx - 46 + j * 32
                shapes.append(f'<rect x="{dx}" y="56" width="28" height="42" rx="3" fill="currentColor" '
                              f'fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.38"/>')
                for ly in (66, 74, 82):
                    shapes.append(f'<line x1="{dx + 5}" y1="{ly}" x2="{dx + 23}" y2="{ly}" '
                                  f'stroke="currentColor" stroke-opacity="0.38" stroke-width="2"/>')
            art.append("".join(shapes))
        elif i == 1:  # a document broken into loose tokens
            shapes = [f'<rect x="{cx - 58}" y="58" width="26" height="40" rx="3" fill="currentColor" '
                      f'fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.38"/>']
            for ly in (68, 76, 84):
                shapes.append(f'<line x1="{cx - 53}" y1="{ly}" x2="{cx - 37}" y2="{ly}" '
                              f'stroke="currentColor" stroke-opacity="0.38" stroke-width="2"/>')
            for r in range(2):
                for c in range(3):
                    shapes.append(f'<rect x="{cx - 20 + c * 24}" y="{60 + r * 22}" width="18" height="12" '
                                  f'rx="3" fill="{ACCENT}" fill-opacity="0.42"/>')
            art.append("".join(shapes))
        elif i == 2:  # a weight grid with uneven cells
            ops = [0.14, 0.55, 0.22, 0.80, 0.38, 0.10, 0.66, 0.26, 0.18, 0.44, 0.90, 0.30]
            shapes = []
            for idx, op in enumerate(ops):
                r, c = divmod(idx, 4)
                shapes.append(f'<rect x="{cx - 40 + c * 21}" y="{54 + r * 21}" width="17" height="17" '
                              f'rx="3" fill="{ACCENT}" fill-opacity="{op}"/>')
            art.append("".join(shapes))
        elif i == 3:  # vectors leaving the origin
            shapes = [f'<line x1="{cx - 34}" y1="108" x2="{cx + 42}" y2="108" stroke="currentColor" '
                      f'stroke-opacity="0.35" stroke-width="2"/>',
                      f'<line x1="{cx - 34}" y1="108" x2="{cx - 34}" y2="52" stroke="currentColor" '
                      f'stroke-opacity="0.35" stroke-width="2"/>']
            for ex, ey, op in ((cx + 22, 62, 0.9), (cx + 36, 84, 0.55), (cx - 4, 56, 0.55)):
                shapes.append(f'<line x1="{cx - 34}" y1="108" x2="{ex}" y2="{ey}" stroke="{ACCENT}" '
                              f'stroke-opacity="{op}" stroke-width="2.5" marker-end="url(#kgtip)"/>')
            art.append("".join(shapes))
        elif i == 4:  # the angle between query and document
            art.append(
                f'<line x1="{cx - 30}" y1="108" x2="{cx + 42}" y2="108" stroke="currentColor" '
                f'stroke-opacity="0.35" stroke-width="2"/>'
                f'<line x1="{cx - 30}" y1="108" x2="{cx - 30}" y2="52" stroke="currentColor" '
                f'stroke-opacity="0.35" stroke-width="2"/>'
                f'<line x1="{cx - 30}" y1="108" x2="{cx + 34}" y2="54" stroke="{ACCENT}" stroke-width="2.8" '
                f'marker-end="url(#kgtip)"/>'
                f'<line x1="{cx - 30}" y1="108" x2="{cx + 40}" y2="84" stroke="currentColor" '
                f'stroke-opacity="0.6" stroke-width="2.5" marker-end="url(#kgtip)"/>'
                f'<path d="M {cx - 4} 86 A 34 34 0 0 1 {cx + 3} 97" fill="none" stroke="currentColor" '
                f'stroke-opacity="0.65" stroke-width="1.6"/>'
                f'<text x="{cx + 9}" y="90" font-size="12" fill="currentColor" fill-opacity="0.75">&#952;</text>'
            )
        else:  # the ranked list
            shapes = []
            for j, w in enumerate((92, 68, 46, 28)):
                fill = ACCENT if j == 0 else "currentColor"
                op = "0.95" if j == 0 else "0.28"
                shapes.append(f'<rect x="{cx - 46}" y="{56 + j * 16}" width="{w}" height="11" rx="3" '
                              f'fill="{fill}" fill-opacity="{op}"/>')
            art.append("".join(shapes))

    parts = [
        _svg_open(1160, 186, "The six stages of the TF-IDF retrieval pipeline"),
        '<defs><marker id="kgtip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>',
    ]
    for i, stage in enumerate(PIPELINE_STAGES):
        x, cx = i * 200, i * 200 + 80
        parts.append(f'<rect x="{x}" y="20" width="160" height="130" rx="14" fill="currentColor" '
                     f'fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.22"/>')
        parts.append(art[i])
        parts.append(f'<circle cx="{x + 21}" cy="35" r="12" fill="{ACCENT}"/>')
        parts.append(f'<text x="{x + 21}" y="40" text-anchor="middle" font-size="13" font-weight="700" '
                     f'fill="#ffffff">{stage["n"]}</text>')
        parts.append(f'<text x="{cx}" y="137" text-anchor="middle" font-size="14" font-weight="600" '
                     f'fill="currentColor">{_esc(stage["name"])}</text>')
        parts.append(f'<text x="{cx}" y="172" text-anchor="middle" font-size="11" fill="currentColor" '
                     f'fill-opacity="0.55">{_esc(stage["where"])}</text>')
        if i < len(PIPELINE_STAGES) - 1:
            parts.append(f'<line x1="{x + 168}" y1="85" x2="{x + 190}" y2="85" stroke="currentColor" '
                         f'stroke-opacity="0.45" stroke-width="2" marker-end="url(#kgtip)"/>')
    parts.append("</svg>")
    return "".join(parts)


def _weight_grid_svg(terms: list, doc_ids: list, weights: list) -> str:
    """Term-by-document TF-IDF weights drawn as a shaded grid: darker cell, heavier weight."""
    cell_w, cell_h, gutter_l, gutter_t = 78, 30, 46, 26
    width = gutter_l + len(terms) * cell_w
    height = gutter_t + len(doc_ids) * cell_h
    peak = max((w for row in weights for w in row), default=0.0) or 1.0

    parts = [_svg_open(width, height, "TF-IDF weight of each query term in each document")]
    for c, term in enumerate(terms):
        parts.append(f'<text x="{gutter_l + c * cell_w + cell_w / 2}" y="17" text-anchor="middle" '
                     f'font-size="13" font-weight="600" fill="currentColor">{_esc(term)}</text>')
    for r, doc in enumerate(doc_ids):
        y = gutter_t + r * cell_h
        parts.append(f'<text x="{gutter_l - 10}" y="{y + cell_h / 2 + 4}" text-anchor="end" font-size="12" '
                     f'fill="currentColor" fill-opacity="0.7">{_esc(doc)}</text>')
        for c in range(len(terms)):
            w = weights[r][c]
            op = round(max(0.05, w / peak), 3)
            x = gutter_l + c * cell_w
            parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" '
                         f'fill="{ACCENT}" fill-opacity="{op}"><title>{_esc(doc)} · {_esc(terms[c])} '
                         f'= {w:.4f}</title></rect>')
            if w > 0:
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" text-anchor="middle" '
                             f'font-size="11" fill="currentColor" fill-opacity="0.9">{w:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_open(view_w: int, view_h: int, label: str) -> str:
    # A drawing rendered through st.image is an isolated document and cannot reach the page's web fonts,
    # so name a system sans-serif stack here instead of inheriting the serif default.
    return (f'<svg viewBox="0 0 {view_w} {view_h}" role="img" aria-label="{_esc(label)}" '
            f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">')


def _draw(svg: str, width="stretch") -> None:
    """Renders an SVG drawing.

    st.html sanitizes with DOMPurify's html-only profile, which strips <svg> outright, so the drawings go
    through st.image instead - Streamlit inlines them as a data URI. An <img> does not inherit the page's
    text color, so currentColor is resolved against the active theme here before handing it over.
    """
    mode = None
    try:
        mode = st.context.theme["type"]
    except Exception:  # theme is unavailable outside a live script run
        pass
    ink = {"light": "#15232A", "dark": "#E3ECEE"}.get(mode, "#7F8C93")
    st.image(svg.replace("currentColor", ink), width=width)


def _hero_svg() -> str:
    """The one-drawing answer to 'what does this website actually do?'.

    Three scenes - you ask, it weighs every word, you get a ranking you can measure.
    """
    p = [_svg_open(1100, 232, "You ask a question, the lab weighs every word, you get a measured ranking"),
         '<defs><marker id="kghero" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
         'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>']

    # Scene A - the query box
    p.append('<rect x="34" y="66" width="250" height="52" rx="26" fill="currentColor" fill-opacity="0.05" '
             'stroke="currentColor" stroke-opacity="0.30"/>')
    p.append(f'<circle cx="68" cy="92" r="12" fill="none" stroke="{ACCENT}" stroke-width="3"/>')
    p.append(f'<line x1="77" y1="101" x2="88" y2="112" stroke="{ACCENT}" stroke-width="3.5" '
             f'stroke-linecap="round"/>')
    p.append('<text x="104" y="98" font-size="15" fill="currentColor" fill-opacity="0.75">'
             'tf-idf document ranking</text>')

    # Scene B - documents fanned into a weight matrix
    for j in range(3):
        dx = 372 + j * 26
        p.append(f'<rect x="{dx}" y="58" width="22" height="30" rx="3" fill="currentColor" '
                 f'fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.35"/>')
    p.append('<text x="411" y="106" text-anchor="middle" font-size="11" fill="currentColor" '
             'fill-opacity="0.55">documents</text>')
    p.append(f'<line x1="452" y1="80" x2="474" y2="80" stroke="currentColor" stroke-opacity="0.4" '
             f'stroke-width="2" marker-end="url(#kghero)"/>')
    ops = [0.14, 0.62, 0.22, 0.85, 0.30,
           0.40, 0.12, 0.70, 0.20, 0.55,
           0.25, 0.48, 0.16, 0.36, 0.92,
           0.60, 0.18, 0.30, 0.74, 0.22]
    for idx, op in enumerate(ops):
        r, c = divmod(idx, 5)
        p.append(f'<rect x="{492 + c * 30}" y="{44 + r * 26}" width="26" height="22" rx="4" '
                 f'fill="{ACCENT}" fill-opacity="{op}"/>')
    p.append('<text x="561" y="164" text-anchor="middle" font-size="11" fill="currentColor" '
             'fill-opacity="0.55">every word, weighted</text>')

    # Scene C - the ranked answer
    for j, w in enumerate((156, 112, 76, 48)):
        top = j == 0
        p.append(f'<text x="852" y="{62 + j * 24}" text-anchor="end" font-size="12" fill="currentColor" '
                 f'fill-opacity="0.6">{j + 1}</text>')
        p.append(f'<rect x="862" y="{50 + j * 24}" width="{w}" height="16" rx="4" '
                 f'fill="{ACCENT if top else "currentColor"}" fill-opacity="{1 if top else 0.24}"/>')
    p.append(f'<circle cx="1038" cy="58" r="11" fill="{ACCENT}"/>')
    p.append('<path d="M 1032 58 L 1036 63 L 1044 53" fill="none" stroke="#ffffff" stroke-width="2.6" '
             'stroke-linecap="round" stroke-linejoin="round"/>')
    p.append('<text x="862" y="164" font-size="11" fill="currentColor" fill-opacity="0.55">'
             'P@k · R@k · F1 · AP · nDCG</text>')

    # Connectors and captions
    p.append('<line x1="300" y1="92" x2="358" y2="92" stroke="currentColor" stroke-opacity="0.45" '
             'stroke-width="2.5" marker-end="url(#kghero)"/>')
    p.append('<line x1="660" y1="92" x2="846" y2="92" stroke="currentColor" stroke-opacity="0.45" '
             'stroke-width="2.5" marker-end="url(#kghero)"/>')
    for cx, title, sub in ((159, "You ask", "any query, any collection"),
                           (561, "It weighs every word", "rare words count, common ones do not"),
                           (940, "You get a ranking", "and the numbers that judge it")):
        p.append(f'<text x="{cx}" y="198" text-anchor="middle" font-size="16" font-weight="700" '
                 f'fill="currentColor">{_esc(title)}</text>')
        p.append(f'<text x="{cx}" y="219" text-anchor="middle" font-size="12" fill="currentColor" '
                 f'fill-opacity="0.6">{_esc(sub)}</text>')
    p.append("</svg>")
    return "".join(p)


def _mini_svg(inner: str, label: str) -> str:
    return _svg_open(120, 80, label) + inner + "</svg>"


def _benefit_art() -> list:
    """Four small drawings, one per reason to use the lab."""
    # 1. every number is on screen: a weight matrix under a magnifier
    grid = []
    for idx, op in enumerate([0.15, 0.55, 0.25, 0.80, 0.35, 0.12, 0.65, 0.30, 0.20, 0.45, 0.90, 0.28]):
        r, c = divmod(idx, 4)
        grid.append(f'<rect x="{14 + c * 21}" y="{8 + r * 18}" width="18" height="15" rx="3" '
                    f'fill="{ACCENT}" fill-opacity="{op}"/>')
    grid.append(f'<circle cx="88" cy="56" r="14" fill="none" stroke="{ACCENT}" stroke-width="3.2"/>')
    grid.append(f'<line x1="98" y1="66" x2="110" y2="78" stroke="{ACCENT}" stroke-width="3.5" '
                f'stroke-linecap="round"/>')

    # 2. one switch, a different ranking
    knob = [f'<rect x="12" y="8" width="46" height="24" rx="12" fill="{ACCENT}" fill-opacity="0.22" '
            f'stroke="{ACCENT}" stroke-opacity="0.6"/>',
            f'<circle cx="46" cy="20" r="9" fill="{ACCENT}"/>']
    for j, w in enumerate((32, 21, 13)):
        knob.append(f'<rect x="8" y="{44 + j * 12}" width="{w}" height="8" rx="3" fill="currentColor" '
                    f'fill-opacity="0.28"/>')
    # Drawn arrowhead rather than a marker: markers live in another SVG's <defs> and may not be rendered.
    knob.append('<line x1="50" y1="56" x2="60" y2="56" stroke="currentColor" stroke-opacity="0.45" '
                'stroke-width="2"/>')
    knob.append('<polygon points="60,52 67,56 60,60" fill="currentColor" fill-opacity="0.45"/>')
    for j, (w, hot) in enumerate(((21, False), (32, True), (11, False))):
        knob.append(f'<rect x="70" y="{44 + j * 12}" width="{w}" height="8" rx="3" '
                    f'fill="{ACCENT if hot else "currentColor"}" fill-opacity="{1 if hot else 0.28}"/>')

    # 3. a gauge, because the ranking gets scored
    gauge = ['<path d="M 22 62 A 38 38 0 0 1 98 62" fill="none" stroke="currentColor" stroke-opacity="0.18" '
             'stroke-width="11" stroke-linecap="round"/>',
             f'<path d="M 22 62 A 38 38 0 0 1 98 62" fill="none" stroke="{ACCENT}" stroke-width="11" '
             f'stroke-linecap="round" stroke-dasharray="84 126"/>',
             '<line x1="60" y1="62" x2="79" y2="36" stroke="currentColor" stroke-width="3" '
             'stroke-linecap="round"/>',
             '<circle cx="60" cy="62" r="4.5" fill="currentColor"/>',
             '<text x="60" y="78" text-anchor="middle" font-size="12" font-weight="700" '
             'fill="currentColor">nDCG</text>']

    # 4. you leave with a PDF
    report = ['<rect x="22" y="8" width="54" height="64" rx="5" fill="currentColor" fill-opacity="0.07" '
              'stroke="currentColor" stroke-opacity="0.35"/>']
    for j, w in enumerate((36, 36, 26, 36, 20)):
        report.append(f'<rect x="30" y="{18 + j * 10}" width="{w}" height="4" rx="2" fill="currentColor" '
                      f'fill-opacity="0.3"/>')
    report.append(f'<circle cx="90" cy="56" r="16" fill="{ACCENT}"/>')
    report.append('<line x1="90" y1="47" x2="90" y2="63" stroke="#ffffff" stroke-width="3" '
                  'stroke-linecap="round"/>')
    report.append('<path d="M 83 56 L 90 64 L 97 56" fill="none" stroke="#ffffff" stroke-width="3" '
                  'stroke-linecap="round" stroke-linejoin="round"/>')

    return ["".join(grid), "".join(knob), "".join(gauge), "".join(report)]


def render_applications_block():
    """Renders the orientation block at the foot of the Aim page: what the lab does, why it is worth
    using, a tour of its pages, and where the same technique is used in practice - plus a live retrieval
    over the chosen application's own collection. Not a page of its own; called by render_aim_section()."""
    st.header("Real-world applications", icon=":material/public:")
    st.caption("New here? Start with this.")
    st.html(APP_VISUAL_CSS)

    # --- What this lab does -------------------------------------------------------------------------
    st.subheader("What this lab does", icon=":material/lightbulb:")
    _draw(_hero_svg())
    with st.expander("How it works, stage by stage", icon=":material/account_tree:"):
        _draw(_pipeline_svg())
        st.caption("Only the collection and the query change between applications. The grey line under "
                   "each stage is where you watch it happen in the Simulation.")

    # --- Why use it ---------------------------------------------------------------------------------
    st.subheader("Why use it", icon=":material/star:")
    reasons = [
        ("Nothing is hidden", "Every TF, IDF and TF-IDF number that produced the ranking is on screen."),
        ("Change one setting", "Switch a weighting scheme and watch the order of the results move."),
        ("Judge the ranking", "Score it with the same parameters real search systems are judged on."),
        ("Leave with a report", "Your trials and observations export as a PDF you can hand in."),
    ]
    art = _benefit_art()
    cols = st.columns(4)
    for col, (title, text), drawing in zip(cols, reasons, art):
        with col:
            with st.container(border=True):
                _draw(_mini_svg(drawing, title), width=120)
                st.markdown(f"**{title}**")
                st.caption(text)

    st.divider()

    # --- Gallery ------------------------------------------------------------------------------------
    st.subheader("Where it is used", icon=":material/apps:")
    for row_start in (0, 3):
        cols = st.columns(3)
        for col, a in zip(cols, APPLICATIONS[row_start:row_start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(f"## {a['icon']}")
                    st.markdown(f"**{a['name']}**")
                    st.markdown(
                        f":gray-badge[:material/description: {a['doc']}]  \n"
                        f":gray-badge[:material/search: {a['query_is']}]  \n"
                        f":blue-badge[:material/straighten: {a['metric']}]"
                    )

    st.divider()

    # --- Live retrieval over the chosen application -------------------------------------------------
    st.subheader("Watch one run", icon=":material/play_circle:")
    names = [a["name"] for a in APPLICATIONS]
    chosen = st.segmented_control(
        "Application", options=names, default=names[0], key="app_scenario", required=True,
        label_visibility="collapsed"
    )
    app = next((a for a in APPLICATIONS if a["name"] == chosen), APPLICATIONS[0])

    # Normalized TF with standard IDF and cosine: the defaults a student meets in the Simulation.
    result = run_retrieval(app["corpus"], app["query"], TF_SCHEMES[1], IDF_SCHEMES[0], use_cosine=True)
    doc_ids = [f"D{i + 1}" for i in range(len(app["corpus"]))]
    matched = set(result["query_tokens"])
    order = result["ranking"]
    peak = max(result["scores"]) or 1.0

    st.markdown(f"**The query** &nbsp; :material/search: &nbsp; *{app['query']}*")

    # Query terms as chips, sized by how much each one can move the ranking.
    terms_by_idf = sorted(matched, key=lambda t: result["idf"][t], reverse=True)
    top_idf = max((result["idf"][t] for t in terms_by_idf), default=0.0) or 1.0
    chips = []
    for t in terms_by_idf:
        size = 13 + 13 * (result["idf"][t] / top_idf)
        chips.append(f'<span class="kg-chip" style="font-size:{size:.0f}px" title="idf {result["idf"][t]:.3f} '
                     f'· appears in {result["df"][t]} of {len(app["corpus"])} documents">{_esc(t)}</span>')
    for t in result["oov_terms"]:
        chips.append(f'<span class="kg-chip kg-dead" style="font-size:13px" '
                     f'title="not in this collection, ignored">{_esc(t)}</span>')
    if chips:
        st.html(f'<div class="kg-chips">{"".join(chips)}</div>')
        st.caption("Bigger chip, rarer term, more say over the ranking. Faded chips are not in this "
                   "collection and are dropped.")

    st.markdown("**The ranking**")
    rows = []
    for rank, i in enumerate(order, start=1):
        score = result["scores"][i]
        pct = 100 * score / peak
        rows.append(
            f'<div class="kg-row">'
            f'<div class="kg-rank{" kg-top" if rank == 1 and score > 0 else ""}">{rank}</div>'
            f'<div class="kg-body">'
            f'<div class="kg-bar{"" if score > 0 else " kg-zero"}"><span style="width:{max(pct, 1.2):.1f}%"></span></div>'
            f'<div class="kg-text">{_highlight_html(app["corpus"][i], matched)}</div>'
            f'</div>'
            f'<div class="kg-score">{score:.3f}</div>'
            f'</div>'
        )
    st.html(f'<div class="kg-wrap">{"".join(rows)}</div>')
    st.info(app["why"], icon=":material/lightbulb:")

    if matched:
        with st.expander("The weights behind those bars", icon=":material/grid_on:"):
            terms = terms_by_idf
            weights = [[result["tfidf_matrix"][i][t] for t in terms] for i in order]
            _draw(_weight_grid_svg(terms, [doc_ids[i] for i in order], weights))
            st.caption("Rows in ranked order. Same numbers as the Simulation's Matrices tab.")

    st.divider()

    # --- Why a rare term decides the ranking --------------------------------------------------------
    st.subheader("Why the rare word wins", icon=":material/insights:")
    st.markdown("IDF is $\\log_{10}(N / \\mathrm{df}_t)$ - it collapses as a term spreads.")
    idf_rows = []
    for df_val in (1, 2, 5, 20, 50, 100):
        idf_val = math.log10(100 / df_val)
        idf_rows.append(
            f'<div class="kg-idfrow">'
            f'<div class="kg-idflab">in {df_val} of 100 documents</div>'
            f'<div class="kg-bar"><span style="width:{max(100 * idf_val / 2.0, 0.6):.1f}%"></span></div>'
            f'<div class="kg-idfval">{idf_val:.2f}</div>'
            f'</div>'
        )
    st.html(f'<div class="kg-wrap">{"".join(idf_rows)}</div>')


def render_references_section():
    """Renders the References page."""
    st.header("References", icon=":material/library_books:")
    st.caption("Where the theory, the formulas and the evaluation parameters used in this lab come from.")

    with st.container(border=True):
        for ref in THEORY_CONTENT["references"]:
            st.markdown(f"- {ref}")


def _decode_upload(file) -> str:
    raw = file.getvalue()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _one_line(text: str) -> str:
    """The corpus is line-based, so a document's own line breaks are collapsed to spaces."""
    return " ".join(str(text).split())


def set_corpus_text(text: str, query: str | None = None):
    """Queues new corpus (and optionally query) text for the widgets.

    A keyed widget ignores a changed `value=` argument, so the browser would keep showing the old text.
    The text is stored here and written into the widget keys at the start of the next run, before the
    widgets are created.
    """
    st.session_state["corpus_text"] = text
    st.session_state["pending_corpus_text"] = text
    if query is not None:
        st.session_state["query_text"] = query
        st.session_state["pending_query_text"] = query


def render_corpus_upload():
    """File uploader that replaces the corpus with documents read from .txt, .md or .csv files."""
    uploader_key = f"corpus_upload_{st.session_state.get('corpus_upload_nonce', 0)}"
    files = st.file_uploader(
        "Upload documents", type=["txt", "md", "csv"], accept_multiple_files=True, key=uploader_key,
        help="One .txt/.md file: each non-empty line is a document. Several .txt/.md files: each file is "
             "one document. .csv: each row of the chosen column is a document."
    )
    if not files:
        return

    docs, sources = [], []
    csv_files = [f for f in files if f.name.lower().endswith(".csv")]
    text_files = [f for f in files if not f.name.lower().endswith(".csv")]

    if len(text_files) == 1:
        lines = [_one_line(line) for line in _decode_upload(text_files[0]).splitlines()]
        new = [line for line in lines if line]
        docs += new
        sources.append(f"{text_files[0].name} ({len(new)} lines)")
    else:
        for f in text_files:
            doc = _one_line(_decode_upload(f))
            if doc:
                docs.append(doc)
        if text_files:
            sources.append(f"{len(text_files)} text files")

    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding_errors="replace")
        except Exception as exc:
            st.error(f"Could not read {f.name}: {exc}", icon=":material/error:")
            continue
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])] or list(df.columns)
        if not text_cols:
            st.warning(f"{f.name} has no columns.", icon=":material/warning:")
            continue
        preferred = next((c for c in text_cols if str(c).strip().lower()
                          in {"text", "document", "doc", "content", "body"}), text_cols[0])
        col = st.selectbox(f"Text column in {f.name}", text_cols, index=text_cols.index(preferred),
                           key=f"{uploader_key}_{f.name}_col")
        new = [d for d in (_one_line(v) for v in df[col].dropna()) if d]
        docs += new
        sources.append(f"{f.name} ({len(new)} rows)")

    if not docs:
        st.warning("No text found in the uploaded files.", icon=":material/warning:")
        return

    st.caption(f"Found **{len(docs)} documents** in " + ", ".join(sources) + ".")
    with st.container(horizontal=True):
        replace = st.button("Replace corpus", type="primary", icon=":material/upload_file:")
        append = st.button("Add to corpus", icon=":material/playlist_add:")
    if replace or append:
        existing = st.session_state["corpus_text"].strip() if append else ""
        set_corpus_text("\n".join(([existing] if existing else []) + docs))
        st.session_state["corpus_upload_nonce"] = st.session_state.get("corpus_upload_nonce", 0) + 1
        st.session_state["corpus_expanded"] = True
        st.toast(f"{'Added' if append else 'Loaded'} {len(docs)} documents", icon=":material/check_circle:")
        st.rerun()


def _use_sample_query():
    """Copies the picked sample query into the search box, then clears the pick."""
    picked = st.session_state.get("sample_query_pick")
    if picked:
        st.session_state["query_text_input"] = picked
        st.session_state["query_text"] = picked
    st.session_state["sample_query_pick"] = None


def render_simulation_section():
    """Renders Section 2: Interactive TF-IDF Retrieval Sandbox."""
    st.header("Simulation", icon=":material/manage_search:")
    st.caption("Search the corpus, change the weighting scheme, and watch the ranking respond. "
               "Record each configuration you want in your report.")

    # --- Controls ---------------------------------------------------------------------------
    if "pending_corpus_text" in st.session_state:
        st.session_state["corpus_text_area"] = st.session_state.pop("pending_corpus_text")
    if "pending_query_text" in st.session_state:
        st.session_state["query_text_input"] = st.session_state.pop("pending_query_text")
    st.session_state.setdefault("corpus_text_area", st.session_state["corpus_text"])
    st.session_state.setdefault("query_text_input", st.session_state["query_text"])

    with st.container(border=True):
        query = st.text_input(
            "Search query", key="query_text_input",
            icon=":material/search:", placeholder="e.g. TF-IDF document ranking"
        )
        st.session_state["query_text"] = query

        if st.session_state["corpus_text"].strip() == "\n".join(DEFAULT_CORPUS):
            st.pills("Sample queries with relevance judgments", options=list(SAMPLE_JUDGMENTS),
                     key="sample_query_pick", on_change=_use_sample_query)

        with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
            tf_label = st.segmented_control(
                "Term frequency", options=TF_SCHEMES, format_func=TF_SHORT.get,
                default=TF_SCHEMES[0], key="tf_scheme", required=True
            )
            idf_label = st.segmented_control(
                "Inverse document frequency", options=IDF_SCHEMES, format_func=IDF_SHORT.get,
                default=IDF_SCHEMES[0], key="idf_scheme", required=True,
                help="Standard: log(N / df). Smoothed: log(1 + N / df)."
            )
            use_cosine = st.toggle("Cosine normalization", value=True, key="use_cosine",
                                   help="Off = raw dot product, which favors longer documents.")
        tf_scheme, idf_scheme = tf_label, idf_label

        with st.expander("Edit document corpus", icon=":material/edit_note:",
                         expanded=st.session_state.pop("corpus_expanded", False)):
            render_corpus_upload()
            corpus_text = st.text_area(
                "One document per line", height=200, key="corpus_text_area"
            )
            st.session_state["corpus_text"] = corpus_text
            if st.button("Restore sample corpus", icon=":material/restart_alt:"):
                set_corpus_text("\n".join(DEFAULT_CORPUS), DEFAULT_QUERY)
                st.session_state["corpus_expanded"] = True
                st.rerun()

    corpus = [line.strip() for line in st.session_state["corpus_text"].split("\n") if line.strip()]
    if len(corpus) < 2:
        st.warning("Add at least two documents to run retrieval: type them one per line, or upload a file "
                   "and click Replace corpus.",
                   icon=":material/warning:")
        return
    if not query.strip():
        st.info("Type a search query to rank the documents.", icon=":material/search:")
        return

    result = run_retrieval(corpus, query, tf_scheme, idf_scheme, use_cosine)
    vocab = result["vocab"]
    doc_ids = [f"D{i + 1}" for i in range(len(corpus))]
    query_terms = set(result["query_tokens"])
    top_idx = result["ranking"][0]
    matched_docs = sum(1 for s in result["scores"] if s > 0)

    # --- Summary strip ------------------------------------------------------------------------
    with st.container(horizontal=True, gap="small"):
        st.metric("Documents", len(corpus), border=True)
        st.metric("Vocabulary", len(vocab), border=True)
        st.metric("Query terms matched", f"{len(query_terms)} / {len(set(tokenize(query)))}", border=True)
        st.metric("Documents with a match", matched_docs, border=True)

    if result["oov_terms"]:
        st.caption("Not in corpus, ignored: " + " ".join(f":gray-badge[{t}]" for t in result["oov_terms"]))
    if not query_terms:
        st.warning("None of the query terms appear in the corpus, so every score is 0. "
                   "Try another query or edit the corpus.", icon=":material/search_off:")

    score_label = "Cosine similarity" if use_cosine else "Dot product"
    max_score = max(result["scores"]) or 1.0

    results_tab, chart_tab, heat_tab, matrix_tab, eval_tab = st.tabs([
        ":material/format_list_numbered: Ranked results",
        ":material/bar_chart: Score chart",
        ":material/grid_on: Term heatmap",
        ":material/table: Matrices",
        ":material/analytics: Evaluation",
    ])

    # --- Ranked results: search-result cards --------------------------------------------------
    with results_tab:
        def result_card(rank: int, idx: int):
            score = result["scores"][idx]
            with st.container(border=True, gap="xsmall"):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**#{rank}** &nbsp; :gray[{doc_ids[idx]}]")
                    if rank == 1 and score > 0:
                        st.badge("Top match", icon=":material/star:", color="primary")
                    elif score == 0:
                        st.badge("No overlap", color="gray")
                    st.space("stretch")
                    st.markdown(f"`{score:.4f}`")
                st.markdown(highlight_terms(corpus[idx], query_terms))
                st.progress(min(1.0, score / max_score) if score > 0 else 0.0)

        ranked = list(enumerate(result["ranking"], start=1))
        for rank, idx in ranked[:matched_docs]:
            result_card(rank, idx)
        # Unmatched documents all score 0; keep them out of the way on large corpora.
        if matched_docs < len(corpus):
            with st.expander(f"{len(corpus) - matched_docs} documents with no query term",
                             icon=":material/visibility_off:"):
                for rank, idx in ranked[matched_docs:]:
                    result_card(rank, idx)

    with chart_tab:
        # Only documents with a score above 0, capped so the chart stays readable on large corpora.
        order = [i for i in result["ranking"] if result["scores"][i] > 0][:20] or result["ranking"][:20]
        if len(order) < matched_docs:
            st.caption(f"Showing the top {len(order)} of {matched_docs} matching documents.")
        fig = go.Figure(go.Bar(
            x=[result["scores"][i] for i in order][::-1],
            y=[doc_ids[i] for i in order][::-1],
            orientation="h",
            text=[f"{result['scores'][i]:.3f}" for i in order][::-1],
            textposition="outside",
            hovertext=[corpus[i] for i in order][::-1],
            hoverinfo="text+x",
        ))
        fig.update_layout(height=max(280, 36 * len(order)), xaxis_title=score_label,
                          margin=dict(l=10, r=40, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    with heat_tab:
        if query_terms:
            terms = sorted(query_terms)
            st.caption("TF-IDF weight of each query term in each document. Rows follow the ranking.")
        else:
            terms = sorted(vocab, key=lambda t: result["idf"][t], reverse=True)[:15]
            st.caption("No query terms matched. Showing the 15 rarest terms in the corpus instead.")
        order = result["ranking"][:15]
        if len(corpus) > len(order):
            st.caption(f"Showing the top {len(order)} ranked documents of {len(corpus)}.")
        z = [[result["tfidf_matrix"][i][t] for t in terms] for i in order]
        heat = go.Figure(go.Heatmap(
            z=z, x=terms, y=[doc_ids[i] for i in order], texttemplate="%{z:.2f}",
            colorbar=dict(title="w"), hovertemplate="%{y} · %{x}<br>weight %{z:.4f}<extra></extra>"
        ))
        heat.update_yaxes(autorange="reversed")
        heat.update_layout(height=max(280, 36 * len(order)), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, width="stretch", theme="streamlit")

    with matrix_tab:
        which = st.segmented_control(
            "Matrix", options=["TF", "IDF", "TF-IDF"], default="TF-IDF",
            key="matrix_view", required=True, label_visibility="collapsed"
        )
        if which == "TF":
            st.caption(f"Term frequency ({TF_SHORT[tf_scheme].lower()}), {len(vocab)} terms.")
            st.dataframe(pd.DataFrame(result["tf_matrix"], index=doc_ids, columns=vocab).round(3), width="stretch")
        elif which == "IDF":
            idf_df = pd.DataFrame([{"Term": t, "df": result["df"][t], "IDF": result["idf"][t]} for t in vocab])
            st.dataframe(
                idf_df.sort_values("IDF", ascending=False), hide_index=True, width="stretch",
                column_config={"IDF": st.column_config.ProgressColumn(
                    "IDF", format="%.4f", min_value=0.0, max_value=float(idf_df["IDF"].max() or 1.0))}
            )
        else:
            st.dataframe(pd.DataFrame(result["tfidf_matrix"], index=doc_ids, columns=vocab).round(4), width="stretch")

    # --- Evaluation parameters ----------------------------------------------------------------
    with eval_tab:
        st.caption("Mark which documents are actually relevant to the query (your relevance judgments). "
                   "Documents with a score above 0 count as retrieved.")
        # Judgments belong to one query over one corpus, so each pair gets its own widget state.
        judgment_key = "relevant_" + str(zlib.crc32(f"{query.strip().lower()}\n{st.session_state['corpus_text']}".encode()))
        sample_judgment = SAMPLE_JUDGMENTS.get(query.strip(), []) if corpus == DEFAULT_CORPUS else []
        if sample_judgment:
            st.caption(":material/fact_check: Pre-filled with the sample judgments for this query. "
                       "Adjust them if you disagree.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
            relevant_ids = st.multiselect(
                "Relevant documents", options=doc_ids, default=sample_judgment,
                key=judgment_key, placeholder="Choose the documents that answer the query"
            )
        with c2:
            k = st.number_input("Cut-off k", min_value=1, max_value=len(corpus),
                                value=min(5, len(corpus)), step=1, key="eval_k")

        if relevant_ids:
            evaluation = evaluate_ranking(result["ranking"], result["scores"],
                                          {doc_ids.index(d) for d in relevant_ids}, k)
        else:
            evaluation = None
            st.info("Select at least one relevant document to compute the evaluation parameters.",
                    icon=":material/rule:")

        if evaluation:
            with st.container(horizontal=True, gap="small"):
                st.metric(f"Precision@{k}", f"{evaluation['precision_k']:.3f}", border=True,
                          help=f"{evaluation['hits_k']} relevant in the top {k} / {k}")
                st.metric(f"Recall@{k}", f"{evaluation['recall_k']:.3f}", border=True,
                          help=f"{evaluation['hits_k']} relevant in the top {k} / {len(relevant_ids)} relevant")
                st.metric(f"F1@{k}", f"{evaluation['f1_k']:.3f}", border=True)
            with st.container(horizontal=True, gap="small"):
                st.metric("Average Precision", f"{evaluation['avg_precision']:.3f}", border=True,
                          help="Mean precision at the rank of each relevant document (MAP over one query).")
                st.metric("Reciprocal Rank", f"{evaluation['reciprocal_rank']:.3f}", border=True,
                          help="1 / rank of the first relevant document (MRR over one query).")
                st.metric(f"nDCG@{k}", f"{evaluation['ndcg_k']:.3f}", border=True)

            if evaluation["per_rank"]:
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("**Precision and recall at each rank**")
                    rank_df = pd.DataFrame([{
                        "Rank": r["rank"], "Doc": doc_ids[r["doc"]], "Relevant": r["relevant"],
                        "Precision": r["precision"], "Recall": r["recall"],
                    } for r in evaluation["per_rank"]])
                    st.dataframe(
                        rank_df, hide_index=True, width="stretch",
                        column_config={
                            "Relevant": st.column_config.CheckboxColumn(),
                            "Precision": st.column_config.NumberColumn(format="%.3f"),
                            "Recall": st.column_config.NumberColumn(format="%.3f"),
                        }
                    )
                with right:
                    st.markdown("**Precision-recall curve**")
                    pr = go.Figure(go.Scatter(
                        x=[r["recall"] for r in evaluation["per_rank"]],
                        y=[r["precision"] for r in evaluation["per_rank"]],
                        mode="lines+markers", text=[doc_ids[r["doc"]] for r in evaluation["per_rank"]],
                        hovertemplate="%{text}<br>recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>",
                    ))
                    pr.update_layout(height=300, xaxis_title="Recall", yaxis_title="Precision",
                                     xaxis_range=[0, 1.05], yaxis_range=[0, 1.05],
                                     margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(pr, width="stretch", theme="streamlit")
            missed = [d for d in relevant_ids if result["scores"][doc_ids.index(d)] == 0]
            if missed:
                st.caption("Relevant but never retrieved (score 0): " +
                           " ".join(f":red-badge[{d}]" for d in missed))

    # --- Trial logger -------------------------------------------------------------------------
    st.subheader("Trial log", icon=":material/science:")
    with st.container(horizontal=True):
        if st.button("Record this trial", type="primary", icon=":material/add_circle:"):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query": query,
                "TF Scheme": tf_scheme,
                "IDF Scheme": idf_scheme,
                "Cosine Norm.": use_cosine,
                "Top Doc": doc_ids[top_idx],
                "Top Score": round(result["scores"][top_idx], 4),
                "k": k,
                "P@k": round(evaluation["precision_k"], 4) if evaluation else None,
                "R@k": round(evaluation["recall_k"], 4) if evaluation else None,
                "F1@k": round(evaluation["f1_k"], 4) if evaluation else None,
                "AP": round(evaluation["avg_precision"], 4) if evaluation else None,
                "nDCG@k": round(evaluation["ndcg_k"], 4) if evaluation else None,
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial {trial_record['Trial #']} recorded", icon=":material/check_circle:")
        if st.session_state["trials"]:
            if st.button("Clear log", icon=":material/delete_sweep:"):
                st.session_state["trials"] = []
                st.toast("Trial log cleared")
                st.rerun()

    if st.session_state["trials"]:
        df_trials = pd.DataFrame(st.session_state["trials"])
        st.dataframe(
            df_trials, width="stretch", hide_index=True,
            column_config={
                "Trial #": st.column_config.NumberColumn("#", width="small"),
                "Cosine Norm.": st.column_config.CheckboxColumn("Cosine"),
                "Top Score": st.column_config.NumberColumn(format="%.4f"),
                **{c: st.column_config.NumberColumn(format="%.3f")
                   for c in ("P@k", "R@k", "F1@k", "AP", "nDCG@k")},
            }
        )
        st.download_button(
            "Download trials (CSV)", data=df_trials.to_csv(index=False).encode("utf-8"),
            file_name="tfidf_experiment_trials.csv", mime="text/csv", icon=":material/download:"
        )
    else:
        st.caption("No trials yet. Aim for 3-4 with different queries and weighting schemes.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Quiz", icon=":material/quiz:")
    st.caption(f"{len(QUIZ_QUESTIONS)} questions on TF-IDF retrieval. Answers are graded when you submit.")

    if st.session_state.get("quiz_submitted", False):
        score = st.session_state.get("quiz_score", 0)
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.metric("Last score", f"{score} / {len(QUIZ_QUESTIONS)}")
            st.progress(score / len(QUIZ_QUESTIONS), text=f"{score / len(QUIZ_QUESTIONS):.0%}")

    with st.form("lab_quiz_form", border=False):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            with st.container(border=True):
                st.markdown(f":gray[Question {q['id']}]  \n**{q['question']}**")
                selected = st.radio(
                    label=f"Options for question {q['id']}",
                    options=q["options"],
                    index=st.session_state["quiz_answers"].get(q["id"], 0),
                    key=f"quiz_radio_{q['id']}",
                    label_visibility="collapsed"
                )
                user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit answers", type="primary", icon=":material/done_all:")

    if submitted:
        score = sum(1 for q in QUIZ_QUESTIONS if user_responses[q["id"]] == q["answer_index"])
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True
        st.session_state["quiz_score"] = score

        st.subheader("Results", icon=":material/grading:")
        st.progress(score / len(QUIZ_QUESTIONS),
                    text=f"**{score} / {len(QUIZ_QUESTIONS)}** correct ({score / len(QUIZ_QUESTIONS):.0%})")
        for q in QUIZ_QUESTIONS:
            user_ans, correct_ans = user_responses[q["id"]], q["answer_index"]
            if user_ans == correct_ans:
                with st.expander(f"Question {q['id']}: correct", icon=":material/check_circle:"):
                    st.markdown(q["explanation"])
            else:
                with st.expander(f"Question {q['id']}: incorrect", icon=":material/cancel:", expanded=True):
                    st.markdown(f":red[Your answer: {q['options'][user_ans]}]  \n"
                                f":green[Correct answer: {q['options'][correct_ans]}]")
                    st.markdown(q["explanation"])


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.header("Report", icon=":material/description:")
    st.caption("Fill in your details and observations, check the checklist, then download the PDF.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Student details**")
            c1, c2 = st.columns(2)
            with c1:
                student_name = st.text_input(
                    "Student name", value=st.session_state["student_info"].get("name", ""),
                    placeholder="Your full name"
                )
            with c2:
                student_id = st.text_input(
                    "Roll nos / group",
                    value=st.session_state["student_info"].get(
                        "id", f"{EXPERIMENT_CONFIG['roll_no']} / Group {EXPERIMENT_CONFIG['group_no']}"
                    )
                )
            lab_date = st.date_input("Experiment date", value=datetime.now())

            st.session_state["student_info"]["name"] = student_name
            st.session_state["student_info"]["id"] = student_id
            st.session_state["student_info"]["date"] = str(lab_date)

        with st.container(border=True):
            st.markdown("**Observations and conclusions**")
            student_notes = st.text_area(
                "Observations", label_visibility="collapsed",
                value=st.session_state.get("student_notes") or DEFAULT_NOTES, height=160
            )
            st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()
    quiz_score = st.session_state.get("quiz_score", 0)

    with right:
        with st.container(border=True):
            st.markdown("**Report checklist**")
            n_trials = len(trials_df)
            st.markdown(
                f"{':material/check_circle:' if student_name.strip() else ':material/radio_button_unchecked:'} Student name  \n"
                f"{':material/check_circle:' if n_trials >= 3 else ':material/radio_button_unchecked:'} "
                f"Trials recorded :gray[({n_trials}, 3+ recommended)]  \n"
                f"{':material/check_circle:' if st.session_state.get('quiz_submitted') else ':material/radio_button_unchecked:'} "
                f"Quiz submitted :gray[({quiz_score} / {len(QUIZ_QUESTIONS)})]"
            )

        pdf_bytes = generate_pdf_report(
            student_name=student_name,
            student_id=student_id,
            date_str=str(lab_date),
            trials_df=trials_df,
            quiz_score=quiz_score,
            quiz_total=len(QUIZ_QUESTIONS),
            student_notes=student_notes
        )

        # The PDF is only sent to this viewer's browser; it is never written to disk, where every
        # user of a hosted app would share (and could open) the same file.
        st.download_button(
            label="Download lab report (PDF)", data=pdf_bytes, file_name="lab_report.pdf",
            mime="application/pdf", key="stream_pdf_btn", type="primary",
            icon=":material/download:", width="stretch"
        )

    st.subheader("Recorded trials", icon=":material/table_rows:")
    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, width="stretch")
    else:
        st.caption("No trials recorded yet. The report will list 0 trials. Record some on the Simulation page.")


# ======================================================================================
# 5. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "",
            "id": f"{EXPERIMENT_CONFIG['roll_no']} / Group {EXPERIMENT_CONFIG['group_no']}",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "corpus_text" not in st.session_state:
        st.session_state["corpus_text"] = "\n".join(DEFAULT_CORPUS)
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = DEFAULT_QUERY


# Streamlit always renders the navigation widget at the very top of the sidebar, above user content.
# Re-ordering the sidebar's flex children puts the experiment header and progress block above it instead.
SIDEBAR_ORDER_CSS = """
<style>
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarHeader"] { order: 0; }
[data-testid="stSidebarUserContent"] { order: 1; padding-bottom: 0.75rem; }
[data-testid="stSidebarNav"] {
    order: 2;
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    padding-top: 0.5rem;
}
</style>
"""


def render_sidebar():
    with st.sidebar:
        st.markdown(f"### Experiment 4\n**TF-IDF document retrieval**")
        st.caption(f"{EXPERIMENT_CONFIG['course']}  \nRoll nos {EXPERIMENT_CONFIG['roll_no']} · "
                   f"Group {EXPERIMENT_CONFIG['group_no']}")
        st.space("small")
        st.markdown("**Your progress**")
        n_trials = len(st.session_state["trials"])
        st.progress(min(n_trials, 3) / 3, text=f"Trials recorded: {n_trials} of 3+")
        if st.session_state.get("quiz_submitted", False):
            score = st.session_state.get("quiz_score", 0)
            st.progress(score / len(QUIZ_QUESTIONS), text=f"Quiz: {score} / {len(QUIZ_QUESTIONS)}")
        else:
            st.badge("Quiz not submitted", icon=":material/schedule:", color="orange")


def main():
    st.set_page_config(
        page_title="TF-IDF Document Retrieval | KGIRS Virtual Lab",
        page_icon=":material/manage_search:",
        layout="wide"
    )

    init_session_state()
    st.html(SIDEBAR_ORDER_CSS)

    page = st.navigation([
        st.Page(render_aim_section, title="Aim", icon=":material/flag:", url_path="aim", default=True),
        st.Page(render_theory_section, title="Theory", icon=":material/menu_book:", url_path="theory"),
        st.Page(render_procedure_section, title="Procedure", icon=":material/checklist:", url_path="procedure"),
        st.Page(render_references_section, title="References", icon=":material/library_books:",
                url_path="references"),
        st.Page(render_simulation_section, title="Simulation", icon=":material/manage_search:", url_path="simulation"),
        st.Page(render_quiz_section, title="Quiz", icon=":material/quiz:", url_path="quiz"),
        st.Page(render_report_section, title="Report", icon=":material/description:", url_path="report"),
    ], position="sidebar")
    page.run()
    render_sidebar()  # after the page so counts include this run's actions


if __name__ == "__main__":
    main()
