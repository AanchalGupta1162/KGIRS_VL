"""The Simulation page: the interactive, step-by-step TF-IDF walk-through."""
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import DEFAULT_CORPUS, DEFAULT_QUERY, SAMPLE_JUDGMENTS, TF_SCHEMES, IDF_SCHEMES, TF_SHORT, IDF_SHORT
from ..engine import run_retrieval, tokenize, compute_tf
from ..graphics import highlight_terms

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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _pdf_to_documents(file) -> list[str]:
    """Extracts a PDF's text and splits it into one document per sentence.

    Splitting by page would leave a single-page file (a resume, say) as just one document, which
    fails the "at least two documents" minimum below. Sentences give a PDF of any length a corpus
    with real variety to rank across.
    """
    from pypdf import PdfReader

    reader = PdfReader(file)
    full_text = " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(full_text) if s.strip()]


def _text_to_documents(text: str) -> list[str]:
    """One document per non-empty line, the format used by the corpus text box."""
    return [line.strip() for line in text.splitlines() if line.strip()]


CUSTOM_QUERY_OPTION = "Type your own..."


def _use_sample_query():
    """Copies the selected sample query into the search box."""
    picked = st.session_state.get("sample_query_pick")
    if picked and picked != CUSTOM_QUERY_OPTION:
        st.session_state["query_text_input"] = picked
        st.session_state["query_text"] = picked



SIMULATION_VISUAL_CSS = """
<style>
/* --- Simulation-only visual polish: richer graph-palette colors, no layout/functionality changes --- */
/* Readable typography for the full simulation workspace. */
div[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
div[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
div[data-testid="stAppViewContainer"] label,
div[data-testid="stAppViewContainer"] button,
div[data-testid="stAppViewContainer"] input,
div[data-testid="stAppViewContainer"] textarea,
div[data-testid="stAppViewContainer"] [data-baseweb="select"],
div[data-testid="stAppViewContainer"] [data-baseweb="input"] * {
    font-size: 1.2rem !important;
}
div[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li {
    font-size: 1.2rem !important;
}
div[data-testid="stAppViewContainer"] h1,
div[data-testid="stAppViewContainer"] h2,
div[data-testid="stAppViewContainer"] h3 {
    font-size: 1.2em !important;
}
div[data-testid="stAppViewContainer"] [data-testid="stDataFrame"] * {
    font-size: 1.1rem !important;
}

div[class*="st-key-sim_hero_"] {
    background:
        radial-gradient(circle at 8% 18%, rgba(42,154,164,.24), transparent 34%),
        radial-gradient(circle at 92% 12%, rgba(224,146,58,.20), transparent 30%),
        radial-gradient(circle at 52% 100%, rgba(108,99,255,.12), transparent 32%),
        linear-gradient(135deg, rgba(42,154,164,.12), rgba(255,255,255,.02));
    border: 1px solid rgba(42,154,164,.38);
    border-radius: 18px;
    padding: 1rem 1.1rem .9rem;
    margin: .25rem 0 1rem;
    box-shadow: 0 10px 30px rgba(42,154,164,.10), 0 4px 18px rgba(108,99,255,.06);
    animation: tfidf-soft-glow 3.8s ease-in-out infinite alternate;
}
@keyframes tfidf-soft-glow {
    from { box-shadow: 0 8px 24px rgba(42,154,164,.08), 0 3px 14px rgba(91,143,249,.05); }
    to   { box-shadow: 0 12px 34px rgba(224,146,58,.14), 0 5px 20px rgba(108,99,255,.08); }
}
div[class*="st-key-sim_hero_"]::after {
    content: "";
    position: absolute;
    width: 110px;
    height: 110px;
    right: 7%;
    top: 2px;
    border-radius: 50%;
    background:
        radial-gradient(circle, rgba(91,192,190,.22), rgba(91,143,249,.10) 42%,
        rgba(224,146,58,.05) 62%, transparent 74%);
    pointer-events: none;
    animation: tfidf-orb-drift 5s ease-in-out infinite alternate;
}
@keyframes tfidf-orb-drift {
    from { transform: translate3d(0, 2px, 0) scale(.92); opacity: .65; }
    to   { transform: translate3d(-14px, 10px, 0) scale(1.08); opacity: 1; }
}

/* Pipeline pills use the same palette as the similarity graph. */
div[class*="st-key-sim_pipeline_"] .sim_step_pill {
    display: inline-block;
    position: relative;
    overflow: hidden;
    border-radius: 999px;
    border: 1px solid rgba(42,154,164,.30);
    background:
        linear-gradient(135deg, rgba(42,154,164,.15), rgba(91,143,249,.08) 48%, rgba(224,146,58,.10));
    padding: .30rem .68rem;
    font-size: 1rem !important;
    font-weight: 650;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
div[class*="st-key-sim_pipeline_"] .sim_step_pill::after {
    content: "";
    position: absolute;
    left: 10%;
    right: 10%;
    bottom: 0;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, #5B8FF9, #2A9AA4, #6C63FF, #E0923A, #D95F8A, #4CAF7D);
    opacity: .72;
}
div[class*="st-key-sim_pipeline_"] .sim_step_pill:hover {
    transform: translateY(-2px);
    border-color: rgba(224,146,58,.55);
    background:
        linear-gradient(135deg, rgba(42,154,164,.20), rgba(108,99,255,.12), rgba(224,146,58,.14));
    box-shadow: 0 6px 18px rgba(42,154,164,.14), 0 3px 12px rgba(108,99,255,.08);
}
div[class*="st-key-sim_pipeline_"] {
    display: flex;
    flex-wrap: wrap;
    gap: .45rem;
    margin: .2rem 0 .9rem;
}

/* Existing step elements: stronger color pops without changing their structure. */
div[class*="st-key-sim_step_"] {
    border-radius: 999px;
    border: 1px solid rgba(42,154,164,.30);
    background:
        linear-gradient(135deg, rgba(42,154,164,.14), rgba(91,143,249,.07) 45%, rgba(224,146,58,.09));
    padding: .28rem .62rem;
    font-size: 1rem !important;
    font-weight: 650;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[class*="st-key-sim_step_"]:hover {
    transform: translateY(-2px);
    border-color: rgba(224,146,58,.52);
    box-shadow: 0 5px 16px rgba(42,154,164,.12), 0 3px 12px rgba(217,95,138,.07);
}

/* Main animation card: layered palette wash keeps teal dominant but adds the graph colors. */
div[class*="st-key-anim_step_card_"] {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 100% 0%, rgba(224,146,58,.13), transparent 27%),
        radial-gradient(circle at 0% 100%, rgba(91,143,249,.10), transparent 30%),
        radial-gradient(circle at 82% 82%, rgba(108,99,255,.08), transparent 25%),
        linear-gradient(145deg, rgba(42,154,164,.075), rgba(255,255,255,.018));
    border: 1px solid rgba(42,154,164,.34) !important;
    box-shadow:
        0 8px 26px rgba(42,154,164,.09),
        0 4px 18px rgba(91,143,249,.05),
        inset 0 1px 0 rgba(255,255,255,.08);
    border-radius: 16px !important;
    animation: tfidf-step-in .35s ease-out, tfidf-card-pulse 2.6s ease-in-out .35s 1;
}
div[class*="st-key-anim_step_card_"]::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
    background: linear-gradient(
        90deg,
        #5B8FF9 0%,
        #2A9AA4 20%,
        #6C63FF 40%,
        #5BC0BE 58%,
        #E0923A 74%,
        #D95F8A 88%,
        #4CAF7D 100%
    );
    background-size: 220% 100%;
    animation: tfidf-palette-flow 4.5s ease-in-out infinite;
    opacity: .92;
    pointer-events: none;
}
div[class*="st-key-anim_step_card_"]::after {
    content: "";
    position: absolute;
    width: 150px;
    height: 150px;
    right: -55px;
    bottom: -65px;
    border-radius: 50%;
    background: radial-gradient(circle,
        rgba(217,95,138,.11),
        rgba(108,99,255,.07) 42%,
        transparent 72%);
    pointer-events: none;
}
@keyframes tfidf-palette-flow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes tfidf-card-pulse {
    0% { box-shadow: 0 8px 26px rgba(42,154,164,.08), 0 3px 14px rgba(91,143,249,.04); }
    45% { box-shadow: 0 12px 32px rgba(224,146,58,.14), 0 5px 20px rgba(108,99,255,.08); }
    100% { box-shadow: 0 8px 26px rgba(42,154,164,.08), 0 3px 14px rgba(91,143,249,.04); }
}
div[class*="st-key-anim_step_card_"] h2,
div[class*="st-key-anim_step_card_"] h3 {
    background: linear-gradient(90deg, #2A9AA4 0%, #5B8FF9 28%, #6C63FF 52%, #E0923A 76%, #D95F8A 100%);
    background-size: 180% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: tfidf-heading-flow 5s ease-in-out infinite;
}
@keyframes tfidf-heading-flow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Existing data areas get subtle palette framing only. */
div[class*="st-key-anim_step_card_"] .stDataFrame,
div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(42,154,164,.18);
    box-shadow: inset 0 0 0 1px rgba(91,143,249,.035);
}
div[class*="st-key-anim_step_card_"] [data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #5B8FF9, #2A9AA4, #6C63FF, #5BC0BE, #E0923A, #D95F8A, #4CAF7D);
    background-size: 250% 100%;
    animation: tfidf-progress-flow 2.2s linear infinite;
}
@keyframes tfidf-progress-flow {
    0% { background-position: 0% 50%; }
    100% { background-position: 250% 50%; }
}
div[class*="st-key-anim_step_card_"] [data-testid="stMetric"] {
    border-radius: 12px;
    border: 1px solid rgba(42,154,164,.22);
    background:
        linear-gradient(135deg, rgba(42,154,164,.07), rgba(91,143,249,.045) 50%, rgba(224,146,58,.045));
}
div[class*="st-key-sim_controls_"] {
    border-radius: 14px;
    background:
        radial-gradient(circle at 0% 50%, rgba(42,154,164,.07), transparent 30%),
        radial-gradient(circle at 100% 50%, rgba(224,146,58,.06), transparent 30%);
}
div[class*="st-key-sim_controls_"] button {
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
div[class*="st-key-sim_controls_"] button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow:
        0 7px 18px rgba(42,154,164,.18),
        0 3px 10px rgba(91,143,249,.08),
        0 2px 8px rgba(224,146,58,.10);
}
div[class*="st-key-anim_step_card_"] [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(224,146,58,.40);
    box-shadow: 0 7px 18px rgba(42,154,164,.10), 0 3px 12px rgba(217,95,138,.07);
}
div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"]:hover {
    box-shadow: 0 8px 22px rgba(42,154,164,.10), 0 3px 12px rgba(108,99,255,.06);
}
div[class*="st-key-anim_step_card_"] [data-testid="stMetric"],
div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"] {
    transition: transform .20s ease, box-shadow .20s ease, border-color .20s ease;
}

/* Existing tabs: a small multicolor wash, not a new UI element. */
div[class*="st-key-sim_controls_"] [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, rgba(42,154,164,.08), rgba(91,143,249,.06), rgba(108,99,255,.06), rgba(224,146,58,.08));
    border-radius: 12px;
    padding: 3px;
}
div[class*="st-key-sim_controls_"] [data-baseweb="tab"] {
    transition: background .18s ease, transform .18s ease;
}
div[class*="st-key-sim_controls_"] [data-baseweb="tab"]:hover {
    background: linear-gradient(90deg, rgba(91,143,249,.10), rgba(108,99,255,.09), rgba(224,146,58,.10));
    transform: translateY(-1px);
}

@media (prefers-reduced-motion: reduce) {
    div[class*="st-key-sim_hero_"],
    div[class*="st-key-anim_step_card_"],
    div[class*="st-key-anim_step_card_"] [data-testid="stProgress"] > div > div,
    div[class*="st-key-anim_step_card_"]::before,
    div[class*="st-key-anim_step_card_"] h2,
    div[class*="st-key-anim_step_card_"] h3 {
        animation: none !important;
    }
    div[class*="st-key-sim_step_"],
    div[class*="st-key-sim_controls_"] button,
    div[class*="st-key-anim_step_card_"] [data-testid="stMetric"],
    div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"] {
        transition: none !important;
    }
    div[class*="st-key-sim_hero_"]::after {
        animation: none !important;
    }
}

/* Center and bold every progress-bar label on this page (step progress, score bars, etc.),
   without touching the sidebar's own progress bars, which stay as they are. */
[data-testid="stMain"] [data-testid="stProgress"] [data-testid="stMarkdownContainer"] {
    text-align: center;
    font-weight: 700;
}

/* Center and bold every table's headers and cells on this page (tokens, TF matrix, TF-IDF
   matrix, query vector, ranking). */
[data-testid="stMain"] [data-testid="stTable"] th,
[data-testid="stMain"] [data-testid="stTable"] td {
    text-align: center !important;
    font-weight: 700 !important;
}
</style>
"""


ANIMATION_STEPS = [
    ("Tokenize", "Tokenize & build vocabulary", ":material/splitscreen:",
     "The text is cleaned and split into terms. We build the vocabulary from the words that occur in the document collection."),
    ("TF matrix", "Term frequency (TF) matrix", ":material/table_rows:",
     "We count how often each vocabulary term appears in each document. This captures a term's local importance."),
    ("DF & IDF", "Document frequency & IDF", ":material/query_stats:",
     "We check how many documents contain each term, then give rare terms a larger IDF weight than common terms."),
    ("TF x IDF", "TF x IDF matrix", ":material/grid_on:",
     "TF and IDF are multiplied together. Terms that are frequent in one document but rare across the collection now stand out."),
    ("Query vector", "Query vector", ":material/search:",
     "The search query is converted into the same weighted vocabulary space so it can be compared fairly with every document."),
    ("Scoring", "Similarity scoring", ":material/calculate:",
     "Each document vector is compared with the query vector. A higher score means the document shares more important query terms."),
    ("Ranking", "Rank documents", ":material/sort:",
     "Documents are ordered from the highest similarity score to the lowest, producing the retrieval result list."),
    ("Top-k", "Top-k results", ":material/emoji_events:",
     "Only the first k documents are shown as the best matches a user should inspect first."),
]
DEFAULT_ANIMATION_SPEED = 1.4  # seconds each step stays on screen while playing

# Real-text corpora can easily reach 1,000+ vocabulary terms. A plain HTML table (needed for bold,
# centered cells - see _render_animation_step) with that many rows/columns is what caused the
# animation to lag badly, so matrix/term tables are capped to this many rows or columns.
MAX_DISPLAY_TERMS = 25


def render_simulation_section():
    """Renders Section 2: a step-by-step walk-through of the TF-IDF pipeline for one query."""
    st.header("Simulation", icon=":material/manage_search:")
    
    st.html(SIMULATION_VISUAL_CSS)
    with st.container(key="sim_hero_"):
        st.markdown("**Interactive retrieval workspace**")
        st.caption("Explore how tokenization, TF, IDF, TF-IDF weighting and cosine similarity work together.")
    with st.container(horizontal=True, gap="small", key="sim_pipeline_"):
        for _step_i, (_short, _long, _icon, _description) in enumerate(ANIMATION_STEPS, start=1):
            st.markdown(f'<div class="sim_step_pill"><b>{_step_i}</b>&nbsp; {_short}</div>', unsafe_allow_html=True)

    # --- Controls ---------------------------------------------------------------------------
    if "pending_corpus_text" in st.session_state:
        st.session_state["corpus_text_area"] = st.session_state.pop("pending_corpus_text")
    if "pending_query_text" in st.session_state:
        st.session_state["query_text_input"] = st.session_state.pop("pending_query_text")
    st.session_state.setdefault("corpus_text_area", st.session_state["corpus_text"])
    st.session_state.setdefault("query_text_input", st.session_state["query_text"])

    corpus_preview = [line.strip() for line in st.session_state["corpus_text"].split("\n") if line.strip()]

    with st.container(border=True, key="sim_controls_"):
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

            uploaded_files = st.file_uploader(
                "Or upload your own corpus", type=["txt", "md", "pdf"], key="corpus_upload",
                accept_multiple_files=True,
                help="One or more .txt/.md files (one document per line) or .pdf files (split into "
                     "sentences). Replaces the corpus above with all of them combined."
            )
            upload_ids = tuple(f.file_id for f in uploaded_files) if uploaded_files else ()
            if upload_ids and st.session_state.get("corpus_upload_ids") != upload_ids:
                st.session_state["corpus_upload_ids"] = upload_ids
                docs = []
                for f in uploaded_files:
                    if f.name.lower().endswith(".pdf"):
                        docs.extend(_pdf_to_documents(f))
                    else:
                        docs.extend(_text_to_documents(f.read().decode("utf-8", errors="replace")))
                set_corpus_text("\n".join(docs))
                st.session_state["corpus_expanded"] = True
                st.rerun()

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

    if not st.session_state["anim_playing"]:
        st.divider()

        if st.button("Record this trial", icon=":material/save:", type="primary",
                     key="record_simulation_trial"):
            top_index = result["ranking"][0]
            st.session_state["trials"].append({
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query": query.strip(),
                "TF Scheme": tf_scheme,
                "IDF Scheme": idf_scheme,
                "Cosine Norm.": "Yes" if use_cosine else "No",
                "Top Doc": doc_ids[top_index],
                "Top Score": result["scores"][top_index],
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            st.success("Trial recorded. You can review it on the Report page.",
                       icon=":material/check_circle:")


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

        st.slider(
            "Speed", min_value=0.3, max_value=3.0, step=0.1, key="anim_speed",
            format="%.1fs/step", width=200,
            help="How long each step stays on screen while playing.",
        )


def _render_animation_display(ctx: dict, n: int):
    # run_every is read fresh on every full rerun (this whole entrypoint file re-executes each
    # time), so toggling "anim_playing" via the controls above changes the interval on the very
    # next run: None when paused, a fixed delay while playing.
    interval = f"{st.session_state['anim_speed']}s" if st.session_state["anim_playing"] else None

    @st.fragment(run_every=interval)
    def _fragment():
        step = st.session_state["anim_step"]
        _, title, icon, description = ANIMATION_STEPS[step]

        st.progress((step + 1) / n, text=f"Step {step + 1} of {n}: **{title}**")

        _inject_step_transition_css()
        # Keying the card by step forces Streamlit to remount it (rather than patch its
        # children) whenever the step changes, which is what replays the CSS fade/slide.
        with st.container(border=True, key=f"anim_step_card_{step}"):
            st.subheader(title, icon=icon)
            st.info(description, icon=":material/lightbulb:")
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
        tok_df = pd.DataFrame([
            {"Doc": doc_ids[i], "Tokens": ", ".join(tokenize(corpus[i])) or "_(none)_"}
            for i in range(len(corpus))
        ])
        st.table(tok_df.style.hide(axis="index"))
        query_tokens_all = tokenize(query)
        if query_tokens_all:
            st.markdown("**Query tokens:** " + " ".join(
                f":green-badge[{t}]" if t in query_terms else f":gray-badge[{t} (out of vocabulary)]"
                for t in query_tokens_all
            ))
        st.caption(f"Vocabulary: **{len(vocab)}** unique terms across **{len(corpus)}** documents.")

    elif step == 1:  # TF matrix
        ordered_terms = sorted(vocab, key=lambda t: (-result["df"][t], t))[:MAX_DISPLAY_TERMS]
        tf_df = pd.DataFrame(result["tf_matrix"], index=doc_ids, columns=vocab)[ordered_terms].round(3)
        st.caption(f"Showing the {len(ordered_terms)} of {len(vocab)} terms that appear in the most "
                   "documents, so the non-zero weights are grouped on the left.")
        with st.container(height=460):
            st.table(tf_df.replace(0, np.nan))

    elif step == 2:  # DF & IDF
        top_terms = sorted(vocab, key=lambda t: result["idf"][t], reverse=True)[:MAX_DISPLAY_TERMS]
        idf_df = pd.DataFrame([{"Term": t, "df": result["df"][t], "IDF": result["idf"][t]} for t in top_terms])
        idf_df = idf_df.sort_values("IDF", ascending=False).round(4)
        st.caption(f"Showing the {len(top_terms)} highest-IDF terms of {len(vocab)} in the vocabulary.")
        st.table(idf_df.style.hide(axis="index"))

    elif step == 3:  # TF-IDF matrix
        st.latex(r"w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t")
        ordered_terms = sorted(vocab, key=lambda t: (-result["df"][t], t))[:MAX_DISPLAY_TERMS]
        tfidf_df = pd.DataFrame(result["tfidf_matrix"], index=doc_ids, columns=vocab)[ordered_terms].round(4)
        st.caption(f"{len(tfidf_df)} documents x {len(ordered_terms)} of {len(vocab)} terms (most common "
                   "words first, so fewer zero weights show up front).")
        with st.container(height=460):
            st.table(tfidf_df.replace(0, np.nan))

    elif step == 4:  # Query vector
        if query_terms:
            query_tf = compute_tf(result["query_tokens"], vocab, tf_scheme)
            q_df = pd.DataFrame([
                {"Term": t, "TF": query_tf[t], "IDF": result["idf"][t], "Weight": result["query_tfidf"][t]}
                for t in sorted(query_terms)
            ])
            st.table(q_df.round(4).style.hide(axis="index"))
        else:
            st.warning("None of the query terms appear in the corpus, so the query vector is all zeros.",
                      icon=":material/search_off:")

    elif step == 5:  # Similarity scoring
        score_label = "Cosine similarity" if use_cosine else "Dot product"
        _chart_palette = ["#2A9AA4", "#E0923A", "#6C63FF", "#4CAF7D", "#D95F8A", "#5B8FF9"]
        _bar_colors = [
            _chart_palette[i % len(_chart_palette)] if score > 0 else "rgba(128,128,128,0.28)"
            for i, score in enumerate(result["scores"])
        ]
        fig = go.Figure(go.Bar(
            x=doc_ids, y=result["scores"],
            text=[f"{s:.3f}" for s in result["scores"]], textposition="outside",
            marker=dict(color=_bar_colors, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=340, yaxis_title=score_label, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.16)", zeroline=False),
        )
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    elif step == 6:  # Ranking
        rank_df = pd.DataFrame([
            {"Rank": r, "Doc": doc_ids[i], "Score": round(result["scores"][i], 4)}
            for r, i in enumerate(result["ranking"], start=1)
        ])
        st.table(rank_df.style.hide(axis="index"))

    else:  # Top-k results
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