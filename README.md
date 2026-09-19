# TF-IDF Document Retrieval: KGIRS Virtual Lab (Experiment 4)

An interactive virtual lab for **Experiment 4, TF-IDF Based Document Retrieval**, in the course *Knowledge Graphs and Information Retrieval Systems* (D17C, Group 4, Roll Nos. 16–20). Students read the theory, run a live TF-IDF search engine, measure how well it ranks documents, take a quiz, and download a PDF lab report.

- **Live app:** https://kgirsvl-c5u68p2abcwf2lrzng5bfk.streamlit.app/
- **Source code:** https://github.com/IshanJ9/KGIRS_VL

---

## Contents

1. [What the app does](#1-what-the-app-does)
2. [How TF-IDF retrieval works in the app](#2-how-tf-idf-retrieval-works-in-the-app)
3. [How the evaluation works](#3-how-the-evaluation-works)
4. [Code structure](#4-code-structure)
5. [Running it locally](#5-running-it-locally)
6. [Deployment](#6-deployment)
7. [Testing](#7-testing)
8. [What was built, step by step](#8-what-was-built-step-by-step)
9. [Known limitations](#9-known-limitations)

---

## 1. What the app does

The app follows the standard four-section Virtual Lab template. The sections appear as tabs across the top of the page.

| Page | What the student does |
|---|---|
| **Theory** | Reads the TF-IDF and cosine formulas, then the Overview (including the evaluation parameters), Objectives, Procedure, Key terms and References tabs. |
| **Simulation** | Searches a document collection, switches weighting schemes, inspects the scores, evaluates the ranking against relevance judgments, and records trials. |
| **Quiz** | Answers 10 multiple-choice questions and gets each one graded with an explanation. |
| **Report** | Enters their details and observations, checks a checklist, and downloads a PDF lab report. |

The sidebar tracks progress: how many trials are recorded (3 or more are recommended) and the quiz score.

### The Simulation page in detail

**Controls** (the box at the top):
- **Search query:** free text, or one of the six sample-query buttons. The buttons appear only while the built-in corpus is loaded.
- **Term frequency:** Raw count, Normalized or Log-normalized.
- **Inverse document frequency:** Standard or Smoothed.
- **Cosine normalization:** on for cosine similarity, off for the raw dot product.
- **Edit document corpus:** type documents one per line, or upload files (see below). **Restore sample corpus** brings back the built-in documents.

**Summary strip:** number of documents, vocabulary size, query terms matched, and documents with at least one match. Query words that are not in the corpus are listed as "Not in corpus, ignored".

**Result tabs:**

| Tab | Shows |
|---|---|
| Ranked results | One card per matching document, with the rank, score, score bar and the query words highlighted. Documents with no matching word are collapsed at the bottom. |
| Score chart | Horizontal bar chart of the top 20 matching documents. |
| Term heatmap | TF-IDF weight of each query term in the top 15 documents. |
| Matrices | The full TF matrix, the IDF table (df and IDF of every term) and the TF-IDF matrix. |
| Evaluation | Relevance judgments, cut-off k, the six evaluation parameters, a precision/recall table at each rank, and a precision-recall curve (see [section 3](#3-how-the-evaluation-works)). |

**Trial log:** **Record this trial** saves the query, schemes, cosine setting, top document and score, plus k and P@k, R@k, F1@k, AP and nDCG@k when relevance judgments are set. The log can be downloaded as CSV and is included in the PDF report.

### Uploading documents

Uploads are in **Edit document corpus**, and accept `.txt`, `.md` and `.csv` files.

| Upload | What becomes a document |
|---|---|
| One `.txt` or `.md` file | Each non-empty line |
| Several `.txt` / `.md` files | Each whole file |
| A `.csv` file | Each row of a chosen text column. A column named `text`, `document`, `doc`, `content` or `body` is picked automatically. |

After uploading, the app shows how many documents it found. **Replace corpus** swaps them in and **Add to corpus** appends them. Line breaks inside a document are turned into spaces, because the corpus is stored one document per line. Files in UTF-8 or the older Windows encoding (cp1252) both work.

### The PDF report

The Report page builds the PDF in memory and sends it straight to the student's browser. It is never saved on the server, so on the hosted app one student can never open another student's report. The PDF contains:

- student details and the quiz score
- the learning objectives
- the full trial table, where long values wrap instead of being cut off
- the student's observations
- a signature line

The report's built-in font only covers basic Latin characters. Dashes and curly quotes are converted to plain versions, and any other unsupported character prints as "?" instead of breaking the PDF.

---

## 2. How TF-IDF retrieval works in the app

Every time a control changes, the app recalculates the whole pipeline in `run_retrieval()`:

```
Corpus (one document per line)
  → Tokenize: lower-case, strip punctuation, drop stop words and 1-letter tokens
  → Vocabulary V (from the corpus only, never from the query)
  → Document frequency df(t) for every term
  → IDF for every term
  → TF for every (term, document) pair
  → TF-IDF document vectors  w(t, d) = tf(t, d) × idf(t)
  → Query vector, built the same way using the corpus IDF
  → Score every document: cosine similarity or dot product
  → Sort by score (ties keep their original order)
```

**Formulas:**

| Setting | Formula |
|---|---|
| Raw TF | tf = count(t, d) |
| Normalized TF | tf = count(t, d) / number of tokens in d |
| Log-normalized TF | tf = 1 + log₁₀ count(t, d), or 0 when the count is 0 |
| Standard IDF | idf = log₁₀(N / df) |
| Smoothed IDF | idf = log₁₀(1 + N / df) |
| Cosine similarity | (q · d) / (‖q‖ ‖d‖) |
| Dot product | q · d |

**Consequences worth knowing:**
- A term that appears in every document has a standard IDF of log(1) = 0, so it adds nothing to any score. Smoothed IDF keeps it at log(2) ≈ 0.301.
- Query words that never appear in the corpus have no IDF and are ignored.
- There is **no stemming**: "oven" does not match "ovens", and "graph" does not match "graphs".
- Word order and repeated query words do not change cosine scores (bag of words).
- Under cosine similarity, Normalized TF gives exactly the same ranking as Raw TF, because dividing a document vector by its length does not change its direction.

---

## 3. How the evaluation works

Evaluation compares the ranking with **relevance judgments**: the set R of documents a person decided are relevant to the query, independently of how TF-IDF ranks them. A document counts as *retrieved* when its score is above 0.

For the built-in corpus, the six sample queries come with judgments (`SAMPLE_JUDGMENTS` in `app.py`), which are filled in automatically. For any other query or corpus, the student picks the relevant documents in the **Relevant documents** box. The cut-off **k** defaults to 5.

| Parameter | Formula | Meaning |
|---|---|---|
| Precision@k | relevant in top k / k | How much of the first page is useful |
| Recall@k | relevant in top k / \|R\| | How much of what is relevant was found |
| F1@k | 2PR / (P + R) | Balance of the two |
| Average Precision (AP) | mean of Precision@i at each rank i holding a relevant document, divided over \|R\| | Rewards relevant documents early; a relevant document never retrieved counts as a miss |
| Reciprocal Rank (RR) | 1 / rank of the first relevant document | How quickly the first useful result appears |
| nDCG@k | DCG@k / ideal DCG@k, where DCG@k = Σ relᵢ / log₂(i + 1) | Position-weighted gain, 1.0 for a perfect ranking |

Averaged over several queries, AP becomes **MAP** and RR becomes **MRR**.

**Worked example** (checked against the app): the query "TF-IDF document ranking" with R = {D3, D4, D40}, raw TF, standard IDF, cosine and k = 5. The relevant documents are at ranks 1, 2 and 5.

- P@5 = 3/5 = 0.600, R@5 = 3/3 = 1.000, F1@5 = 0.750
- AP = (1/1 + 2/2 + 3/5) / 3 = 0.867, RR = 1.000
- nDCG@5 = (1 + 1/log₂3 + 1/log₂6) / (1 + 1/log₂3 + 1/log₂4) = 2.0178 / 2.1309 = 0.947

---

## 4. Code structure

```
KGIRS_VL/
├── app.py                  # The whole app (single file)
├── sample_corpus.txt       # Built-in corpus: 41 documents, one per line (required at startup)
├── requirements.txt        # Python packages for local runs and for hosting
├── .streamlit/config.toml  # Theme: colours, fonts, light and dark modes
├── .gitignore
├── README.md               # This file
│
│   Local only (ignored by git):
├── test_data/              # Test corpora and TEST_PLAN.md with expected results
├── template.py             # Original blank Virtual Lab template
└── venv/                   # Local Python environment
```

### Inside `app.py`

| Section | Key names | Purpose |
|---|---|---|
| 1. Configuration and content | `EXPERIMENT_CONFIG`, `THEORY_CONTENT`, `STOPWORDS`, `DEFAULT_CORPUS`, `SAMPLE_JUDGMENTS`, `QUIZ_QUESTIONS` | Course details, theory text, stop-word list, built-in corpus, sample judgments and quiz. Edit here to change content. |
| 2. Retrieval engine | `tokenize`, `build_vocabulary`, `compute_tf`, `compute_df`, `compute_idf`, `tfidf_vector`, `cosine_similarity`, `run_retrieval`, `evaluate_ranking` | Plain-Python TF-IDF and evaluation, with no Streamlit code, so they can be tested on their own. |
| 3. PDF exporter | `LabReportPDF`, `generate_pdf_report`, `PDF_CHAR_MAP` | Builds the lab report with fpdf2. |
| 4. Page renderers | `render_theory_section`, `render_simulation_section`, `render_quiz_section`, `render_report_section`, plus upload helpers | One function per page. |
| 5. Entry point | `init_session_state`, `render_sidebar`, `main` | Session setup, sidebar and top navigation (`st.navigation`). |

### Design decisions

- **No custom CSS.** All styling comes from `.streamlit/config.toml`: a teal accent, Space Grotesk headings, IBM Plex Sans body text and IBM Plex Mono for code, with separate light and dark themes. It survives Streamlit upgrades and follows the viewer's light or dark setting.
- **Per-student state.** Trials, quiz answers and the corpus live in `st.session_state`. Every visitor has their own copy, and it resets when the page is refreshed.
- **Text-box updates.** Replace corpus, Add to corpus and Restore sample corpus go through `set_corpus_text()`. It queues the new text and writes it into the text box at the start of the next run. Passing a new value to a keyed widget does not update what the browser shows, and an earlier version of the app showed stale text because of that bug.
- **Evaluation judgments are stored per query and corpus.** Each query/corpus pair gets its own widget key, so changing the query does not carry judgments over to the wrong query.

---

## 5. Running it locally

Requirements: Python 3.12 or newer. The project was built on Python 3.14 with Streamlit 1.64.

```bash
python -m venv venv
```

```bash
venv/Scripts/python -m pip install -r requirements.txt
```

```bash
venv/Scripts/python -m streamlit run app.py
```

The app opens at http://localhost:8501. On macOS or Linux, use `venv/bin/python` instead of `venv/Scripts/python`.

### Agent skill for Streamlit

The official Streamlit agent skill (`developing-with-streamlit`) is installed for Claude Code at user level with `streamlit skills --global`. It loads Streamlit's bundled, version-matched guides when working on this app.

---

## 6. Deployment

The app is hosted on **Streamlit Community Cloud**, connected to the `main` branch of the GitHub repository.

- **Updating:** every push to `main` redeploys automatically within a minute or two.
- **If a push seems ignored:** open share.streamlit.io, go to the app's ⋮ menu and choose **Reboot app**. **Manage app** in the same menu shows the build logs.
- **Required files:** `app.py`, `requirements.txt`, `.streamlit/config.toml` and `sample_corpus.txt`. The app reads `sample_corpus.txt` at startup and will not start without it.
- **Sleeping:** a free app goes to sleep after a period without visitors. The next visitor sees a wake-up button and waits about a minute, so open the link shortly before a demo.

To publish changes:

```bash
git add .
```

```bash
git commit -m "describe your change"
```

```bash
git push
```

---

## 7. Testing

`test_data/` is kept locally (not on GitHub) and contains:

| File | Purpose |
|---|---|
| `corpus.txt` | 6 documents designed to show each weighting effect (keyword repetition, a long document, two meanings of "apple") |
| `corpus.csv` | 4 rows where every document contains "fruit", which tests df = N and IDF = 0 |
| `multi/` | 3 separate files, each loaded as one document |
| `TEST_PLAN.md` | Step-by-step checks with the exact expected scores, computed by the app's own code |

A few expected results on `corpus.txt` (raw TF, standard IDF, cosine on unless stated):

| Query | Expected |
|---|---|
| `apple` | D2 0.8753, D1 0.2375, D3 0.1124, D6 0.0935 |
| `apple` with log-normalized TF | D2 drops to 0.3955 |
| `bread recipe` | D4 0.3296 ahead of D5 0.1523 |
| `bread recipe` with cosine off | D4 and D5 tie at 0.2587 |
| `pizza oven` | All scores 0 and both words listed as not in corpus |

The retrieval, evaluation and PDF functions have no Streamlit code, so they can also be checked directly in Python, or through the whole app with Streamlit's `AppTest` (`from streamlit.testing.v1 import AppTest`).

---

## 8. What was built, step by step

1. **Setup.** Installed the official Streamlit agent skill, then installed Streamlit and the other requirements in the project's `venv`.
2. **UI redesign.** Replaced the sidebar radio menu with top navigation and added the theme in `config.toml`. Rebuilt the Simulation page around a single control box, a summary strip, search-result cards with highlighted query words, and tabs for the chart, heatmap and matrices. Gave the quiz per-question cards and a graded summary. Gave the report page a checklist. All of this used Streamlit's own theming, with no custom CSS.
3. **Course details.** Set the roll numbers to 16–20 for the whole group and the class to D17C, in the app and in the PDF.
4. **File upload.** Added `.txt`, `.md` and `.csv` upload for the corpus, with Replace and Add options. A bug where the text box kept showing old text after an upload was found and fixed with `set_corpus_text()`.
5. **PDF report fixes.** Made the PDF trial table wrap long values instead of cutting them to 16 characters, repeat the header row on new pages, and handle special characters. Also stopped the signature line and section headings from being split across pages.
6. **Hosting.** Committed only the needed files, using `.gitignore` for the rest, and pushed to GitHub. Removed a shared-file download that would have let students open each other's reports, raised the minimum Streamlit version to 1.64, and deployed on Streamlit Community Cloud.
7. **Evaluation features.** Added the Evaluation tab (Precision@k, Recall@k, F1@k, AP, RR, nDCG@k, a per-rank table and a precision-recall curve). Also added the 41-document `sample_corpus.txt` with six judged sample queries and the evaluation columns in the trial log and PDF.
8. **Lab report.** Produced a Word/PDF lab report in the format of the course's sample report. It uses screenshots and numbers taken from real sessions of the app, with worked calculations that match the app's output.

---

## 9. Known limitations

- **Lexical matching only.** There is no stemming, lemmatisation or synonym handling, so documents that use different words for the same idea are missed. This is visible in the evaluation, for example with "vocabulary mismatch synonyms".
- **Small evaluation set.** 41 documents and 6 judged queries are right for a teaching lab, but differences in MAP between weighting schemes are too small to generalise. Adding more judged queries would make comparisons more reliable than adding documents.
- **Speed on large corpora.** The whole pipeline reruns in plain Python on every change. It takes about 0.04 s for 41 documents and about 0.6 s for 500. Much larger collections would need caching or vectorised code.
- **Nothing is saved between visits.** Trials, quiz answers and uploaded corpora are lost when the page is refreshed. Students should download their PDF report in the same session.
