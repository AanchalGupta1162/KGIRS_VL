"""Sidebar chrome: font-size/order CSS, the (currently unused) header helpers, and the
progress block shown under the page nav."""
import streamlit as st

from .config import EXPERIMENT_CONFIG, QUIZ_LENGTH

SIDEBAR_ORDER_CSS = """
<style>
html {
    font-size: 112.5% !important;
}
[data-testid="stAppViewContainer"] {
    font-size: 1.2rem !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] button,
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] [data-baseweb="select"],
[data-testid="stAppViewContainer"] [data-baseweb="input"] * {
    font-size: 1.2rem !important;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2 {
    font-size: 1.55em !important;
}
[data-testid="stAppViewContainer"] h3 {
    font-size: 1.2em !important;
}
[data-testid="stAppViewContainer"] [data-testid="stDataFrame"] * {
    font-size: 1.1rem !important;
}
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] * {
    font-size: 1.15rem !important;
    line-height: 1.15 !important;
}
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] table {
    table-layout: fixed;
    width: 100%;
}
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] th,
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] td {
    padding: .18rem .4rem !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
}
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] td > div {
    width: 100% !important;
    max-width: none !important;
}
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] th:first-child,
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] td:first-child {
    width: 34%;
}
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] th:last-child,
[data-testid="stAppViewContainer"] [class*="st-key-key_terms_table"] [data-testid="stTable"] td:last-child {
    width: 66%;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 1.2rem !important;
}
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarHeader"] { order: 0; }
[data-testid="stSidebarUserContent"] { order: 1; padding-bottom: 0.75rem; }
[data-testid="stSidebarNav"] {
    order: 2;
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    padding-top: 0.5rem;
}
/* Streamlit's wide layout pads the main content 5rem from the sidebar by default; tighten that gap. */
[data-testid="stMainBlockContainer"] {
    padding-left: 1.5rem !important;
}
</style>
"""


def render_sidebar():
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
            st.progress(score / QUIZ_LENGTH, text=f"Quiz: {score} / {QUIZ_LENGTH}")