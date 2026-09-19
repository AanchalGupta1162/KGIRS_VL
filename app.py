"""
Virtual Laboratory - Experiment 4: TF-IDF Based Document Retrieval
Course: Knowledge Graphs and Information Retrieval Systems (D17C)
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
    "course": "Knowledge Graphs and Information Retrieval Systems (D17C)",
    "roll_no": "16-20",
    "group_no": "4",
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
        "Step 1: Review the theoretical background on TF-IDF weighting and the vector space model below.",
        "Step 2: Navigate to the Simulation section in the sidebar menu.",
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
        "Step 11: Open Report Generation, enter your student information, and download your PDF report."
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
# 4. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
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


def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, and Procedure."""
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

    overview, objectives, procedure, glossary, refs = st.tabs([
        ":material/article: Overview",
        ":material/flag: Objectives",
        ":material/checklist: Procedure",
        ":material/dictionary: Key terms",
        ":material/library_books: References",
    ])
    with overview:
        st.markdown(THEORY_CONTENT["background"])
    with objectives:
        for obj in EXPERIMENT_CONFIG["objectives"]:
            st.markdown(f"- {obj}")
    with procedure:
        for step in THEORY_CONTENT["procedure"]:
            label, _, text = step.partition(": ")
            st.markdown(f"**{label}** &nbsp; {text}")
    with glossary:
        var_df = pd.DataFrame(list(THEORY_CONTENT["key_terms"].items()), columns=["Term", "Definition"])
        st.dataframe(
            var_df, hide_index=True, width="stretch",
            column_config={"Term": st.column_config.TextColumn(width="medium"),
                           "Definition": st.column_config.TextColumn(width="large")}
        )
    with refs:
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

    page = st.navigation([
        st.Page(render_theory_section, title="Theory", icon=":material/menu_book:", url_path="theory", default=True),
        st.Page(render_simulation_section, title="Simulation", icon=":material/manage_search:", url_path="simulation"),
        st.Page(render_quiz_section, title="Quiz", icon=":material/quiz:", url_path="quiz"),
        st.Page(render_report_section, title="Report", icon=":material/description:", url_path="report"),
    ], position="top")
    page.run()
    render_sidebar()  # after the page so counts include this run's actions


if __name__ == "__main__":
    main()
