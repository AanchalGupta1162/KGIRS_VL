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


def _render_theory_header():
    """Shared header shown at the top of every Theory sub-page."""
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


def render_theory_overview():
    """Renders Theory > Overview."""
    _render_theory_header()
    st.subheader("Overview", icon=":material/article:")
    st.markdown(THEORY_CONTENT["background"])


def render_theory_objectives():
    """Renders Theory > Objectives."""
    _render_theory_header()
    st.subheader("Objectives", icon=":material/flag:")
    for obj in EXPERIMENT_CONFIG["objectives"]:
        st.markdown(f"- {obj}")


def render_theory_procedure():
    """Renders Theory > Procedure."""
    _render_theory_header()
    st.subheader("Procedure", icon=":material/checklist:")
    for step in THEORY_CONTENT["procedure"]:
        label, _, text = step.partition(": ")
        st.markdown(f"**{label}** &nbsp; {text}")


def render_theory_key_terms():
    """Renders Theory > Key terms."""
    _render_theory_header()
    st.subheader("Key terms", icon=":material/dictionary:")
    var_df = pd.DataFrame(list(THEORY_CONTENT["key_terms"].items()), columns=["Term", "Definition"])
    st.dataframe(
        var_df, hide_index=True, width="stretch",
        column_config={"Term": st.column_config.TextColumn(width="medium"),
                       "Definition": st.column_config.TextColumn(width="large")}
    )


def render_theory_references():
    """Renders Theory > References."""
    _render_theory_header()
    st.subheader("References", icon=":material/library_books:")
    for ref in THEORY_CONTENT["references"]:
        st.markdown(f"- {ref}")


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


CUSTOM_QUERY_OPTION = "Type your own..."


def _use_sample_query():
    """Copies the selected sample query into the search box."""
    picked = st.session_state.get("sample_query_pick")
    if picked and picked != CUSTOM_QUERY_OPTION:
        st.session_state["query_text_input"] = picked
        st.session_state["query_text"] = picked


ANIMATION_STEPS = [
    ("Tokenize", "Tokenize & build vocabulary", ":material/splitscreen:"),
    ("TF matrix", "Term frequency (TF) matrix", ":material/table_rows:"),
    ("DF & IDF", "Document frequency & IDF", ":material/query_stats:"),
    ("TF x IDF", "TF x IDF matrix", ":material/grid_on:"),
    ("Query vector", "Query vector", ":material/search:"),
    ("Scoring", "Similarity scoring", ":material/calculate:"),
    ("Ranking", "Rank documents", ":material/sort:"),
    ("Top-k", "Top-k results", ":material/emoji_events:"),
]
ANIMATION_INTERVAL = "1.4s"


def render_simulation_section():
    """Renders Section 2: a step-by-step walk-through of the TF-IDF pipeline for one query."""
    st.header("Simulation", icon=":material/manage_search:")
    st.caption("Set a query and a weighting scheme, then step through the pipeline to watch TF-IDF build "
               "the term-document matrix, weight it by IDF, score it against your query, and rank the results.")

    # --- Controls ---------------------------------------------------------------------------
    if "pending_corpus_text" in st.session_state:
        st.session_state["corpus_text_area"] = st.session_state.pop("pending_corpus_text")
    if "pending_query_text" in st.session_state:
        st.session_state["query_text_input"] = st.session_state.pop("pending_query_text")
    st.session_state.setdefault("corpus_text_area", st.session_state["corpus_text"])
    st.session_state.setdefault("query_text_input", st.session_state["query_text"])

    corpus_preview = [line.strip() for line in st.session_state["corpus_text"].split("\n") if line.strip()]

    with st.container(border=True):
        is_default_corpus = st.session_state["corpus_text"].strip() == "\n".join(DEFAULT_CORPUS)
        query_col, sample_col = st.columns([3, 2], vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Search query", key="query_text_input",
                icon=":material/search:", placeholder="e.g. TF-IDF document ranking"
            )
        with sample_col:
            if is_default_corpus:
                st.selectbox(
                    "Sample queries", options=[CUSTOM_QUERY_OPTION] + list(SAMPLE_JUDGMENTS),
                    key="sample_query_pick", on_change=_use_sample_query,
                    help="Pick one of the sample queries the corpus was written around."
                )
        st.session_state["query_text"] = query

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
            top_k = st.number_input(
                "Top-k results", min_value=1, max_value=max(1, len(corpus_preview)),
                value=min(5, max(1, len(corpus_preview))), step=1, key="top_k",
                help="How many top-ranked documents the animation reveals at the end."
            )
        tf_scheme, idf_scheme = tf_label, idf_label

        with st.expander("Edit document corpus", icon=":material/edit_note:",
                         expanded=st.session_state.pop("corpus_expanded", False)):
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
        st.warning("Add at least two documents to run retrieval: type them one per line.",
                   icon=":material/warning:")
        return
    if not query.strip():
        st.info("Type a search query to run the TF-IDF pipeline.", icon=":material/search:")
        return

    result = run_retrieval(corpus, query, tf_scheme, idf_scheme, use_cosine)
    doc_ids = [f"D{i + 1}" for i in range(len(corpus))]
    top_k = min(top_k, len(corpus))

    # Reset to the first step whenever the underlying pipeline configuration changes, so the
    # walk-through never shows a stale step for a different query/corpus/scheme combination.
    fingerprint = (tuple(corpus), query.strip(), tf_scheme, idf_scheme, use_cosine, top_k)
    if st.session_state["anim_fingerprint"] != fingerprint:
        st.session_state["anim_fingerprint"] = fingerprint
        st.session_state["anim_step"] = 0
        st.session_state["anim_playing"] = False

    ctx = {
        "corpus": corpus, "doc_ids": doc_ids, "query": query, "result": result,
        "tf_scheme": tf_scheme, "idf_scheme": idf_scheme, "use_cosine": use_cosine, "top_k": top_k,
    }

    step = st.session_state["anim_step"]
    n = len(ANIMATION_STEPS)

    _render_animation_controls(step, n)
    _render_animation_display(ctx, n)


def _inject_step_transition_css():
    st.html("""
        <style>
        @keyframes tfidf-step-in {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        div[class*="st-key-anim_step_card_"] {
            animation: tfidf-step-in 0.35s ease-out;
        }
        </style>
    """)


def _render_animation_controls(step: int, n: int):
    """Previous/Play-Pause/Next/Reset. Lives outside the fragment below so every click triggers
    a full rerun, which is required to change the fragment's `run_every` interval."""
    playing = st.session_state["anim_playing"]
    at_end = step >= n - 1

    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if st.button("Previous", icon=":material/chevron_left:", disabled=step <= 0):
            st.session_state["anim_step"] = step - 1
            st.session_state["anim_playing"] = False
            st.rerun()

        if playing:
            if st.button("Pause", icon=":material/pause:", type="primary"):
                st.session_state["anim_playing"] = False
                st.rerun()
        else:
            if st.button("Replay" if at_end else "Play",
                        icon=":material/replay:" if at_end else ":material/play_arrow:",
                        type="primary"):
                if at_end:
                    st.session_state["anim_step"] = 0
                st.session_state["anim_playing"] = True
                st.rerun()

        if st.button("Next", icon=":material/chevron_right:", disabled=step >= n - 1):
            st.session_state["anim_step"] = step + 1
            st.session_state["anim_playing"] = False
            st.rerun()

        if st.button("Reset", icon=":material/restart_alt:", disabled=step == 0 and not playing):
            st.session_state["anim_step"] = 0
            st.session_state["anim_playing"] = False
            st.rerun()


def _render_animation_display(ctx: dict, n: int):
    # run_every is read fresh on every full rerun (this whole entrypoint file re-executes each
    # time), so toggling "anim_playing" via the controls above changes the interval on the very
    # next run: None when paused, a fixed delay while playing.
    interval = ANIMATION_INTERVAL if st.session_state["anim_playing"] else None

    @st.fragment(run_every=interval)
    def _fragment():
        step = st.session_state["anim_step"]
        _, title, icon = ANIMATION_STEPS[step]

        st.progress((step + 1) / n, text=f"Step {step + 1} of {n}: **{title}**")

        _inject_step_transition_css()
        # Keying the card by step forces Streamlit to remount it (rather than patch its
        # children) whenever the step changes, which is what replays the CSS fade/slide.
        with st.container(border=True, key=f"anim_step_card_{step}"):
            st.subheader(title, icon=icon)
            _render_animation_step(step, ctx)

        if st.session_state["anim_playing"]:
            if step < n - 1:
                st.session_state["anim_step"] = step + 1
            else:
                # Reached the end: stop, and force a full rerun so the controls/fragment above
                # are redefined with run_every=None on their next render (a fragment-scoped
                # rerun can't change its own run_every).
                st.session_state["anim_playing"] = False
                st.rerun()

    _fragment()


def _render_animation_step(step: int, ctx: dict):
    corpus, doc_ids, query = ctx["corpus"], ctx["doc_ids"], ctx["query"]
    result, tf_scheme, idf_scheme = ctx["result"], ctx["tf_scheme"], ctx["idf_scheme"]
    use_cosine, top_k = ctx["use_cosine"], ctx["top_k"]
    vocab = result["vocab"]
    query_terms = set(result["query_tokens"])

    if step == 0:  # Tokenize & build vocabulary
        st.markdown("Every document (and the query) is lower-cased, stripped of punctuation, and filtered "
                    "through a stop-word list. The vocabulary is the set of unique terms left in the corpus.")
        tok_df = pd.DataFrame([
            {"Doc": doc_ids[i], "Tokens": ", ".join(tokenize(corpus[i])) or "_(none)_"}
            for i in range(len(corpus))
        ])
        st.dataframe(tok_df, hide_index=True, width="stretch")
        query_tokens_all = tokenize(query)
        if query_tokens_all:
            st.markdown("**Query tokens:** " + " ".join(
                f":green-badge[{t}]" if t in query_terms else f":gray-badge[{t} (out of vocabulary)]"
                for t in query_tokens_all
            ))
        st.caption(f"Vocabulary: **{len(vocab)}** unique terms across **{len(corpus)}** documents.")

    elif step == 1:  # TF matrix
        st.markdown(f"Term frequency ({TF_SHORT[tf_scheme].lower()}) counts how often each term occurs "
                    "in each document — one row per document, one column per vocabulary term.")
        tf_df = pd.DataFrame(result["tf_matrix"], index=doc_ids, columns=vocab).round(3)
        st.dataframe(tf_df, width="stretch")

    elif step == 2:  # DF & IDF
        st.markdown(f"Document frequency (df) counts how many documents contain each term. "
                    f"IDF ({IDF_SHORT[idf_scheme]}) then down-weights terms that appear in many documents.")
        idf_df = pd.DataFrame([{"Term": t, "df": result["df"][t], "IDF": result["idf"][t]} for t in vocab])
        idf_df = idf_df.sort_values("IDF", ascending=False)
        st.dataframe(
            idf_df, hide_index=True, width="stretch",
            column_config={"IDF": st.column_config.ProgressColumn(
                "IDF", format="%.4f", min_value=0.0, max_value=float(idf_df["IDF"].max() or 1.0))}
        )

    elif step == 3:  # TF-IDF matrix
        st.latex(r"w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t")
        st.markdown("Multiplying the TF matrix by the IDF vector (broadcast across every document) gives "
                    "the TF-IDF weight matrix: high where a term is frequent here but rare elsewhere.")
        tfidf_df = pd.DataFrame(result["tfidf_matrix"], index=doc_ids, columns=vocab).round(4)
        st.dataframe(tfidf_df, width="stretch")

    elif step == 4:  # Query vector
        st.markdown("The query is tokenized the same way, then projected into the same TF-IDF space using "
                    "the corpus's IDF values.")
        if query_terms:
            query_tf = compute_tf(result["query_tokens"], vocab, tf_scheme)
            q_df = pd.DataFrame([
                {"Term": t, "TF": query_tf[t], "IDF": result["idf"][t], "Weight": result["query_tfidf"][t]}
                for t in sorted(query_terms)
            ])
            st.dataframe(q_df.round(4), hide_index=True, width="stretch")
        else:
            st.warning("None of the query terms appear in the corpus, so the query vector is all zeros.",
                      icon=":material/search_off:")

    elif step == 5:  # Similarity scoring
        score_label = "Cosine similarity" if use_cosine else "Dot product"
        st.markdown(f"Each document vector is compared against the query vector using **{score_label}**.")
        fig = go.Figure(go.Bar(
            x=doc_ids, y=result["scores"],
            text=[f"{s:.3f}" for s in result["scores"]], textposition="outside",
        ))
        fig.update_layout(height=340, yaxis_title=score_label, showlegend=False,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    elif step == 6:  # Ranking
        st.markdown("Documents are sorted in descending order of similarity score.")
        rank_df = pd.DataFrame([
            {"Rank": r, "Doc": doc_ids[i], "Score": result["scores"][i]}
            for r, i in enumerate(result["ranking"], start=1)
        ])
        st.dataframe(
            rank_df, hide_index=True, width="stretch",
            column_config={"Score": st.column_config.NumberColumn(format="%.4f")}
        )

    else:  # Top-k results
        st.markdown(f"The **top {top_k}** ranked documents for this query.")
        max_score = max(result["scores"]) or 1.0
        for rank, idx in enumerate(result["ranking"][:top_k], start=1):
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
    if "anim_step" not in st.session_state:
        st.session_state["anim_step"] = 0
    if "anim_playing" not in st.session_state:
        st.session_state["anim_playing"] = False
    if "anim_fingerprint" not in st.session_state:
        st.session_state["anim_fingerprint"] = None


def render_sidebar_header():
    with st.sidebar:
        st.markdown(f"### Experiment 4\n**TF-IDF document retrieval**")
        st.caption(f"{EXPERIMENT_CONFIG['course']}  \nRoll nos {EXPERIMENT_CONFIG['roll_no']} · "
                   f"Group {EXPERIMENT_CONFIG['group_no']}")


def render_sidebar_nav(pages: dict):
    """Renders the page links manually, as a list with a sub-list per section.

    st.navigation's built-in sidebar widget always pins itself to the very top of the
    sidebar, so it can't be positioned below other sidebar content. Rendering the links
    ourselves with st.page_link (while st.navigation runs with position="hidden") lets the
    nav sit below the Experiment 4 header instead.
    """
    with st.sidebar:
        for section, section_pages in pages.items():
            if section:
                st.caption(section)
            for p in section_pages:
                st.page_link(p)


def render_sidebar_progress():
    with st.sidebar:
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

    pages = {
        "Theory": [
            st.Page(render_theory_overview, title="Overview", icon=":material/article:",
                    url_path="theory-overview", default=True),
            st.Page(render_theory_objectives, title="Objectives", icon=":material/flag:",
                    url_path="theory-objectives"),
            st.Page(render_theory_procedure, title="Procedure", icon=":material/checklist:",
                    url_path="theory-procedure"),
            st.Page(render_theory_key_terms, title="Key terms", icon=":material/dictionary:",
                    url_path="theory-key-terms"),
            st.Page(render_theory_references, title="References", icon=":material/library_books:",
                    url_path="theory-references"),
        ],
        "Lab": [
            st.Page(render_simulation_section, title="Simulation", icon=":material/manage_search:", url_path="simulation"),
            st.Page(render_quiz_section, title="Quiz", icon=":material/quiz:", url_path="quiz"),
            st.Page(render_report_section, title="Report", icon=":material/description:", url_path="report"),
        ],
    }

    render_sidebar_header()
    render_sidebar_nav(pages)

    page = st.navigation(pages, position="hidden")
    page.run()
    render_sidebar_progress()  # after the page so counts include this run's actions


if __name__ == "__main__":
    main()
